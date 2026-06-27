import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import torch
import numpy as np
import pandas as pd
import o_voxel
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis2.models as models
import trellis2.modules.sparse as sp


torch.set_grad_enabled(False)


def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def truthy(value):
    return str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y'}


def load_one_metadata_filter(metadata_filter_csv):
    if not os.path.exists(metadata_filter_csv):
        raise FileNotFoundError(f'Metadata filter CSV not found: {metadata_filter_csv}')
    filter_metadata = pd.read_csv(metadata_filter_csv)
    if 'sha256' not in filter_metadata.columns:
        raise ValueError(f'{metadata_filter_csv} must contain a sha256 column.')
    if 'local_density_filter_keep' in filter_metadata.columns:
        filter_metadata = filter_metadata[
            filter_metadata['local_density_filter_keep'].map(truthy)
        ]
    elif 'has_local_dense_region' in filter_metadata.columns:
        filter_metadata = filter_metadata[
            ~filter_metadata['has_local_dense_region'].map(truthy)
        ]
    return set(filter_metadata['sha256'].astype(str).values)


def load_metadata_filter(metadata_filter_csv):
    paths = [path.strip() for path in metadata_filter_csv.split(',') if path.strip()]
    if len(paths) == 0:
        return set()
    allowed = load_one_metadata_filter(paths[0])
    for path in paths[1:]:
        allowed = allowed & load_one_metadata_filter(path)
    return allowed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory containing metadata.csv')
    parser.add_argument('--gaussian_distance_voxel_root', type=str, default=None,
                        help='Directory containing gaussian_distance_voxels_<resolution>')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory to save occupancy shape latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Sparse voxel resolution')
    parser.add_argument('--model_root', type=str, required=True,
                        help='Root directory containing the occupancy shape VAE run')
    parser.add_argument('--enc_model', type=str, required=True,
                        help='Occupancy shape VAE run name under model_root')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint to load, e.g. step0090000')
    parser.add_argument('--instances', type=str, default=None,
                        help='Optional instances file or comma-separated sha256 list')
    parser.add_argument('--metadata_filter_csv', type=str, default=None,
                        help='Optional comma-separated metadata CSV filters. Filters are intersected.')
    parser.add_argument('--loader_workers', type=int, default=4,
                        help='Number of background voxel readers')
    parser.add_argument('--read_threads', type=int, default=1,
                        help='o_voxel read threads per file')
    parser.add_argument('--saver_workers', type=int, default=4,
                        help='Number of background latent savers')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.gaussian_distance_voxel_root = opt.gaussian_distance_voxel_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root

    latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
    cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
    encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
    ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
    encoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True), strict=False)
    encoder.eval()
    print(f'Loaded model from {ckpt_path}')

    latent_root = os.path.join(opt.shape_latent_root, 'shape_latents', latent_name)
    os.makedirs(os.path.join(latent_root, 'new_records'), exist_ok=True)

    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256'))
    voxel_metadata_path = os.path.join(opt.gaussian_distance_voxel_root, f'gaussian_distance_voxels_{opt.resolution}', 'metadata.csv')
    if os.path.exists(voxel_metadata_path):
        metadata = metadata.combine_first(pd.read_csv(voxel_metadata_path).set_index('sha256'))
    latent_metadata_path = os.path.join(latent_root, 'metadata.csv')
    if os.path.exists(latent_metadata_path):
        metadata = metadata.combine_first(pd.read_csv(latent_metadata_path).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.metadata_filter_csv is not None and opt.metadata_filter_csv.strip() != '':
        total_before_filter = len(metadata)
        allowed = load_metadata_filter(opt.metadata_filter_csv)
        metadata = metadata[metadata['sha256'].astype(str).isin(allowed)]
        print(
            f'Applied metadata filter: {total_before_filter} -> {len(metadata)} objects '
            f'using {opt.metadata_filter_csv}'
        )

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['gaussian_distance_voxelized'] == True]
        if 'shape_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['shape_latent_encoded'] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(metadata), desc='Filtering existing objects') as pbar:
        def check_sha256(sha256):
            latent_path = os.path.join(latent_root, f'{sha256}.npz')
            if os.path.exists(latent_path):
                coords = np.load(latent_path)['coords']
                records.append({'sha256': sha256, 'shape_latent_encoded': True, 'shape_latent_tokens': coords.shape[0]})
            pbar.update()
        executor.map(check_sha256, metadata['sha256'].values)
        executor.shutdown(wait=True)

    existing_sha256 = set(r['sha256'] for r in records)
    print(f'Found {len(existing_sha256)} processed objects')
    metadata = metadata[~metadata['sha256'].isin(existing_sha256)]
    print(f'Processing {len(metadata)} objects...')

    sha256s = list(metadata['sha256'].values)
    load_queue = Queue(maxsize=32)
    with ThreadPoolExecutor(max_workers=opt.loader_workers) as loader_executor, \
            ThreadPoolExecutor(max_workers=opt.saver_workers) as saver_executor:

        def loader(sha256):
            try:
                coords, _ = o_voxel.io.read_vxz(
                    os.path.join(opt.gaussian_distance_voxel_root, f'gaussian_distance_voxels_{opt.resolution}', f'{sha256}.vxz'),
                    num_threads=opt.read_threads,
                )
                x = sp.SparseTensor(
                    torch.ones((coords.shape[0], 1), dtype=torch.float32),
                    torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
                )
                load_queue.put((sha256, x))
            except Exception as e:
                print(f'[Loader Error] {sha256}: {e}')
                load_queue.put((sha256, None))

        loader_executor.map(loader, sha256s)

        def saver(sha256, pack):
            save_path = os.path.join(latent_root, f'{sha256}.npz')
            np.savez_compressed(save_path, **pack)
            records.append({'sha256': sha256, 'shape_latent_encoded': True, 'shape_latent_tokens': pack['coords'].shape[0]})

        for _ in tqdm(range(len(sha256s)), desc='Extracting occupancy shape latents'):
            try:
                sha256, x = load_queue.get()
                if x is None:
                    print(f'[Skip] {sha256}: Failed to load input')
                    continue

                num_voxels = x.feats.shape[0]
                if not is_valid_sparse_tensor(x):
                    print(f'[Skip] {sha256}: NaN/Inf in input')
                    continue

                z = encoder(x.cuda(), sample_posterior=False)
                torch.cuda.synchronize()

                if not torch.isfinite(z.feats).all():
                    print(f'[Skip] {sha256}: Non-finite latent in z.feats')
                    clear_cuda_error()
                    continue

                pack = {
                    'feats': z.feats.cpu().numpy().astype(np.float32),
                    'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_executor.submit(saver, sha256, pack)

            except Exception as e:
                print(f'[Error] {sha256} ({num_voxels} voxels): {e}')
                clear_cuda_error()
                continue

        saver_executor.shutdown(wait=True)

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(latent_root, 'new_records', f'part_{opt.rank}.csv'), index=False)
