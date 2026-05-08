#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Michelangelo'))
import argparse
import hashlib
import json
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

torch.set_grad_enabled(False)


def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')


def default_latent_name(config_path: str, ckpt_path: str) -> str:
    config_stem = Path(config_path).stem
    ckpt_stem = Path(ckpt_path).stem
    return sanitize_name(f'michelangelo_{config_stem}_{ckpt_stem}')


class MichelangeloShapeEncoder(torch.nn.Module):
    def __init__(self, shape_model):
        super().__init__()
        self.shape_model = shape_model

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):
        pc = surface[..., 0:3]
        feats = surface[..., 3:6]
        _, shape_zq, _ = self.shape_model.encode(
            pc=pc,
            feats=feats,
            sample_posterior=sample_posterior,
        )
        return shape_zq


def load_michelangelo_model(config_path: str, ckpt_path: str, device: str):
    from michelangelo.utils.misc import get_config_from_file, instantiate_from_config

    model_config = get_config_from_file(config_path)
    if hasattr(model_config, 'model'):
        model_config = model_config.model

    # Michelangelo configs can expose shape_module_cfg either directly under model
    # or nested under model.params.
    if hasattr(model_config, 'shape_module_cfg'):
        shape_module_cfg = model_config.shape_module_cfg
    elif hasattr(model_config, 'params') and hasattr(model_config.params, 'shape_module_cfg'):
        shape_module_cfg = model_config.params.shape_module_cfg
    else:
        raise ValueError(
            f'Could not find shape_module_cfg in config: {config_path}. '
            'Expected model.shape_module_cfg or model.params.shape_module_cfg.'
        )

    shape_model = instantiate_from_config(shape_module_cfg, device=None, dtype=None)

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    shape_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.shape_model.'):
            shape_state_dict[key[len('model.shape_model.'):]] = value
        elif key.startswith('shape_model.'):
            shape_state_dict[key[len('shape_model.'):]] = value

    if not shape_state_dict:
        raise ValueError(
            f'No shape-model weights found in {ckpt_path}. '
            'Expected checkpoint keys prefixed by model.shape_model. or shape_model.'
        )

    missing, unexpected = shape_model.load_state_dict(shape_state_dict, strict=False)
    print(
        f'Restored Michelangelo shape model from {ckpt_path} '
        f'with {len(missing)} missing and {len(unexpected)} unexpected keys'
    )
    if len(missing) > 0:
        print(f'Missing shape keys: {missing}')
    if len(unexpected) > 0:
        print(f'Unexpected shape keys: {unexpected}')

    model = MichelangeloShapeEncoder(shape_model)
    model = model.to(device)
    model.eval()
    return model


def sha_seed(sha256: str, seed: int) -> int:
    digest = hashlib.sha256(f'{seed}:{sha256}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='little', signed=False)


def load_mesh_dump(path: str):
    with open(path, 'rb') as f:
        dump = pickle.load(f)

    vertices = []
    faces = []
    offset = 0
    for obj in dump.get('objects', []):
        v = np.asarray(obj.get('vertices'), dtype=np.float32)
        f = np.asarray(obj.get('faces'), dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            continue
        if len(v) == 0 or len(f) == 0:
            continue
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)

    if not vertices or not faces:
        raise ValueError('mesh dump contains no usable triangle mesh objects')

    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)


def sample_surface(vertices, faces, num_points: int, rng: np.random.Generator, coordinate_scale: float):
    if not np.isfinite(vertices).all():
        raise ValueError('vertices contain non-finite values')

    faces = faces[
        (faces >= 0).all(axis=1) &
        (faces < len(vertices)).all(axis=1)
    ]
    if len(faces) == 0:
        raise ValueError('mesh has no valid faces')

    tris = vertices[faces]
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(normals, axis=1)
    valid = area2 > 1e-12
    if not valid.any():
        raise ValueError('mesh has no non-degenerate faces')

    v0 = v0[valid]
    v1 = v1[valid]
    v2 = v2[valid]
    normals = normals[valid]
    area2 = area2[valid]
    probs = area2 / area2.sum()

    face_idx = rng.choice(len(v0), size=num_points, replace=True, p=probs)
    r1 = rng.random(num_points, dtype=np.float32)
    r2 = rng.random(num_points, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    b0 = 1.0 - sqrt_r1
    b1 = sqrt_r1 * (1.0 - r2)
    b2 = sqrt_r1 * r2

    points = (
        b0[:, None] * v0[face_idx] +
        b1[:, None] * v1[face_idx] +
        b2[:, None] * v2[face_idx]
    ).astype(np.float32)
    points *= np.float32(coordinate_scale)

    sampled_normals = normals[face_idx] / area2[face_idx, None]
    sampled_normals = sampled_normals.astype(np.float32)

    surface = np.concatenate([points, sampled_normals], axis=-1)
    if not np.isfinite(surface).all():
        raise ValueError('sampled surface contains non-finite values')
    return surface


def load_surface_from_mesh_dump(sha256: str, mesh_dump_root: str, num_points: int, seed: int, coordinate_scale: float):
    dump_path = os.path.join(mesh_dump_root, 'mesh_dumps', f'{sha256}.pickle')
    vertices, faces = load_mesh_dump(dump_path)
    rng = np.random.default_rng(sha_seed(sha256, seed))
    surface = sample_surface(vertices, faces, num_points, rng, coordinate_scale)
    return torch.from_numpy(surface)


def parse_instances(instances_arg: str):
    if instances_arg is None:
        return None
    if os.path.exists(instances_arg):
        with open(instances_arg, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return [item.strip() for item in instances_arg.split(',') if item.strip()]


def load_metadata(opt, latent_dir: str):
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')

    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(
            pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256')
        )
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(
            pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256')
        )
    if os.path.exists(os.path.join(latent_dir, 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(latent_dir, 'metadata.csv')).set_index('sha256'))

    metadata = metadata.reset_index()
    instances = parse_instances(opt.instances)
    if instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'mesh_dumped' in metadata.columns:
            metadata = metadata[metadata['mesh_dumped'] == True]
        if 'michelangelo_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['michelangelo_latent_encoded'] != True]
    else:
        metadata = metadata[metadata['sha256'].isin(instances)]

    metadata = metadata.reset_index(drop=True)
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    return metadata[start:end]


def save_latent(path: str, feats: np.ndarray, compress: bool):
    if compress:
        np.savez_compressed(path, feats=feats)
    else:
        np.savez(path, feats=feats)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Encode mesh dumps into pretrained Michelangelo shape latents.'
    )
    parser.add_argument('--root', type=str, required=True, help='Dataset root containing metadata.csv')
    parser.add_argument('--mesh_dump_root', type=str, default=None, help='Root containing mesh_dumps')
    parser.add_argument('--michelangelo_latent_root', type=str, default=None, help='Root to save michelangelo_latents')
    parser.add_argument('--michelangelo_root', type=str, default=str(repo_root / 'Michelangelo'), help='Michelangelo repo root')
    parser.add_argument('--config_path', type=str, default=None, help='Michelangelo shape VAE config YAML')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Michelangelo checkpoint path')
    parser.add_argument('--latent_name', type=str, default=None, help='Output latent model name')
    parser.add_argument('--instances', type=str, default=None, help='Comma-separated sha256s or file with one sha256 per line')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None)
    parser.add_argument('--num_points', type=int, default=4096, help='Surface points sampled per mesh')
    parser.add_argument('--coordinate_scale', type=float, default=2.0, help='Scale applied to TRELLIS-normalized mesh coordinates before Michelangelo encoding')
    parser.add_argument('--batch_size', type=int, default=16, help='GPU encoding batch size')
    parser.add_argument('--max_workers', type=int, default=4, help='CPU mesh loading/surface sampling workers')
    parser.add_argument('--saver_workers', type=int, default=4, help='Concurrent NPZ saver workers')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample_posterior', action='store_true', help='Sample Michelangelo KL posterior instead of using deterministic mode')
    parser.add_argument('--compress', action='store_true', help='Use np.savez_compressed instead of np.savez')
    parser.add_argument('--benchmark', action='store_true', help='Print batch timing')
    opt = edict(vars(parser.parse_args()))

    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.michelangelo_latent_root = opt.michelangelo_latent_root or opt.root
    opt.config_path = opt.config_path or os.path.join(opt.michelangelo_root, 'configs', 'aligned_shape_latents', 'shapevae-256.yaml')
    opt.latent_name = sanitize_name(opt.latent_name or default_latent_name(opt.config_path, opt.ckpt_path))

    sys.path.insert(0, opt.michelangelo_root)
    latent_dir = os.path.join(opt.michelangelo_latent_root, 'michelangelo_latents', opt.latent_name)
    os.makedirs(os.path.join(latent_dir, 'new_records'), exist_ok=True)

    model = load_michelangelo_model(opt.config_path, opt.ckpt_path, opt.device)
    print(f'Loaded Michelangelo model from {opt.ckpt_path}')
    print(f'Writing latents to {latent_dir}')

    metadata = load_metadata(opt, latent_dir)
    records = []

    print(f'Filtering existing objects...')
    existing = []
    for sha256 in tqdm(metadata['sha256'].values, desc='Filtering existing objects'):
        path = os.path.join(latent_dir, f'{sha256}.npz')
        if os.path.exists(path):
            try:
                feats = np.load(path)['feats']
                records.append({
                    'sha256': sha256,
                    'michelangelo_latent_encoded': True,
                    'michelangelo_latent_tokens': feats.shape[0],
                    'michelangelo_latent_dim': feats.shape[1],
                })
                existing.append(sha256)
            except Exception as e:
                print(f'[Existing Error] {sha256}: {e}')
    metadata = metadata[~metadata['sha256'].isin(existing)]
    sha256s = list(metadata['sha256'].values)

    print(f'Found {len(existing)} processed objects')
    print(f'Processing {len(sha256s)} objects...')

    def loader(sha256):
        start_t = time.perf_counter()
        try:
            surface = load_surface_from_mesh_dump(
                sha256,
                opt.mesh_dump_root,
                opt.num_points,
                opt.seed,
                opt.coordinate_scale,
            )
            return sha256, surface, time.perf_counter() - start_t, None
        except Exception as e:
            return sha256, None, time.perf_counter() - start_t, e

    def flush_batch(batch, saver_executor):
        if not batch:
            return
        encode_start = time.perf_counter()
        sha_batch = [item[0] for item in batch]
        surface = torch.stack([item[1] for item in batch], dim=0).to(opt.device, non_blocking=True)
        z = model.encode(surface, sample_posterior=opt.sample_posterior)
        torch.cuda.synchronize() if opt.device.startswith('cuda') else None
        z = z.detach().float().cpu().numpy()
        encode_s = time.perf_counter() - encode_start

        for i, sha256 in enumerate(sha_batch):
            feats = z[i].astype(np.float32)
            if not np.isfinite(feats).all():
                print(f'[Skip] {sha256}: non-finite Michelangelo latent')
                continue
            save_path = os.path.join(latent_dir, f'{sha256}.npz')
            saver_executor.submit(save_latent, save_path, feats, opt.compress)
            records.append({
                'sha256': sha256,
                'michelangelo_latent_encoded': True,
                'michelangelo_latent_tokens': feats.shape[0],
                'michelangelo_latent_dim': feats.shape[1],
            })
        if opt.benchmark:
            load_s = sum(item[2] for item in batch)
            print(f'[Benchmark] batch={len(batch)} load_sum_s={load_s:.3f} encode_s={encode_s:.3f}')

    with ThreadPoolExecutor(max_workers=opt.max_workers) as loader_executor, \
         ThreadPoolExecutor(max_workers=opt.saver_workers) as saver_executor:
        futures = [loader_executor.submit(loader, sha256) for sha256 in sha256s]
        batch = []
        for future in tqdm(as_completed(futures), total=len(futures), desc='Encoding Michelangelo latents'):
            sha256, surface, load_s, error = future.result()
            if error is not None:
                print(f'[Loader Error] {sha256}: {error}')
                continue
            batch.append((sha256, surface, load_s))
            if len(batch) >= opt.batch_size:
                flush_batch(batch, saver_executor)
                batch = []
        flush_batch(batch, saver_executor)
        saver_executor.shutdown(wait=True)

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(latent_dir, 'new_records', f'part_{opt.rank}.csv'), index=False)
    print(f'Wrote {len(records)} records to {os.path.join(latent_dir, "new_records", f"part_{opt.rank}.csv")}')


if __name__ == '__main__':
    main()
