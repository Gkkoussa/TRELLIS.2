import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import trellis2.models as models
import trellis2.modules.sparse as sp
from trellis2.datasets.sparse_voxel_triangle_field import (
    DENSITY_ELONGATION_INPUT_LAYOUT,
    ELONGATION_INPUT_LAYOUT,
    EXTENDED_INPUT_LAYOUT,
    INPUT_LAYOUT,
    TARGET_LAYOUT,
    find_triangle_field_path,
    load_triangle_field_npz,
)

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


def to_cpu_cache(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu_cache(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_cpu_cache(v) for v in value)
    if isinstance(value, list):
        return [to_cpu_cache(v) for v in value]
    return value


def trim_decoder_spatial_cache(spatial_cache):
    """
    Keep only cache entries required by pred_subdiv=False decoder upsampling.

    This mirrors gaussian-distance latent encoding: flow/eval can restore the
    sparse subdivision structure without retaining encoder-only conv caches.
    """
    keep_substrings = ('channel2spatial', 'shape')
    trimmed = {}
    for scale_key, scale_cache in spatial_cache.items():
        if not isinstance(scale_cache, dict):
            continue
        kept_scale_cache = {
            key: value
            for key, value in scale_cache.items()
            if any(keep in str(key) for keep in keep_substrings)
        }
        if kept_scale_cache:
            trimmed[scale_key] = kept_scale_cache
    return trimmed


def read_unique_metadata(path):
    metadata = pd.read_csv(path)
    if 'sha256' not in metadata.columns:
        raise ValueError(f'{path} is missing sha256')
    duplicate_count = metadata['sha256'].duplicated(keep='last').sum()
    if duplicate_count:
        print(f'[Metadata] Dropping {duplicate_count} duplicate rows from {path}', flush=True)
        metadata = metadata.drop_duplicates('sha256', keep='last')
    return metadata.set_index('sha256')


def atomic_save_npz(path, pack):
    tmp_path = f'{path}.tmp.{os.getpid()}'
    try:
        with open(tmp_path, 'wb') as f:
            np.savez_compressed(f, **pack)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_torch_save(path, value):
    tmp_path = f'{path}.tmp.{os.getpid()}'
    try:
        torch.save(value, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def require_triangle_field_dataset_args(cfg, resolution, allow_resolution_mismatch=False):
    if 'dataset' not in cfg:
        raise ValueError('VAE config is missing dataset settings.')
    valid_dataset_names = {
        'SparseVoxelTriangleFieldDataset',
        'MultiResolutionSparseVoxelTriangleFieldDataset',
    }
    if cfg.dataset.name not in valid_dataset_names:
        raise ValueError(
            f"Expected VAE dataset one of {sorted(valid_dataset_names)}, got {cfg.dataset.name}"
        )
    if 'args' not in cfg.dataset:
        raise ValueError('VAE config dataset is missing args.')

    dataset_args = cfg.dataset.args
    required = ['voxel_dirname', 'voxelized_flag_column', 'num_voxels_column', 'distance_transform']
    if cfg.dataset.name == 'SparseVoxelTriangleFieldDataset':
        required.append('resolution')
    else:
        required.append('resolutions')
    missing = [key for key in required if key not in dataset_args]
    if missing:
        raise ValueError(f'VAE config triangle-field dataset args are missing: {missing}')

    if cfg.dataset.name == 'MultiResolutionSparseVoxelTriangleFieldDataset':
        resolutions = [int(r) for r in dataset_args.resolutions]
        if int(resolution) not in resolutions:
            raise ValueError(
                f'--resolution {resolution} is not in VAE config dataset resolutions {resolutions}'
            )
        dataset_args = edict(dict(dataset_args))
        dataset_args.resolution = int(resolution)

    if int(dataset_args.resolution) != int(resolution) and not allow_resolution_mismatch:
        raise ValueError(
            f'--resolution {resolution} does not match VAE config dataset resolution {dataset_args.resolution}'
        )
    if int(dataset_args.resolution) != int(resolution) and allow_resolution_mismatch:
        print(
            f'[Warning] --resolution {resolution} does not match VAE config dataset resolution '
            f'{dataset_args.resolution}; continuing because --allow_resolution_mismatch was set.',
            flush=True,
        )
    if dataset_args.distance_transform not in ('none', 'minus_one_one'):
        raise ValueError(
            f"Unsupported distance_transform {dataset_args.distance_transform}; "
            "expected 'none' or 'minus_one_one'."
        )
    return dataset_args


def get_triangle_field_input_layout(dataset_args):
    include_density = dataset_args.get('include_density_field', False)
    include_elongation = dataset_args.get('include_elongation_field', False)
    if include_density and include_elongation:
        return DENSITY_ELONGATION_INPUT_LAYOUT
    if include_density:
        return EXTENDED_INPUT_LAYOUT
    if include_elongation:
        return ELONGATION_INPUT_LAYOUT
    return INPUT_LAYOUT


def build_triangle_field_sparse_tensor(path, dataset_args):
    input_layout = get_triangle_field_input_layout(dataset_args)
    num_input_channels = max(slc.stop for slc in input_layout.values())
    num_target_channels = max(slc.stop for slc in TARGET_LAYOUT.values())

    with load_triangle_field_npz(path) as data:
        coords = torch.from_numpy(data['coords'].astype(np.int32, copy=False))
        features = torch.from_numpy(data['features'].astype(np.float32, copy=False))

    if features.ndim != 2 or features.shape[1] < num_input_channels:
        raise ValueError(f'{path} has invalid feature shape {tuple(features.shape)}')
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'{path} has invalid coords shape {tuple(coords.shape)}')
    if coords.shape[0] != features.shape[0]:
        raise ValueError(f'{path} coords/features length mismatch: {coords.shape[0]} vs {features.shape[0]}')

    features = features[:, :num_input_channels]
    if dataset_args.distance_transform == 'minus_one_one':
        features = features.clone()
        features[:, :num_target_channels] = features[:, :num_target_channels] * 2.0 - 1.0

    input_feature_scale = dataset_args.get('input_feature_scale', None)
    if input_feature_scale is not None:
        scale = torch.tensor(input_feature_scale, dtype=torch.float32)
        if scale.numel() == 1:
            features = features / scale.clamp_min(1e-12)
        elif scale.numel() == features.shape[1]:
            features = features / scale.reshape(1, -1).clamp_min(1e-12)
        else:
            raise ValueError(
                f'input_feature_scale must be scalar or length {features.shape[1]}, got {scale.numel()}'
            )

    sparse_coords = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
    return sp.SparseTensor(features.float(), sparse_coords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--triangle_field_voxel_root', type=str, default=None,
                        help='Directory containing triangle_field_voxels_<resolution>')
    parser.add_argument('--triangle_field_latent_root', type=str, default=None,
                        help='Directory to save the triangle-field latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Sparse voxel resolution')
    parser.add_argument('--model_root', type=str,
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str,
                        help='Encoder model run name under model_root')
    parser.add_argument('--ckpt', type=str,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--metadata_filter_csv', type=str, default=None,
                        help='Optional comma-separated metadata CSV filters. Filters are intersected.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--loader_workers', type=int, default=4,
                        help='Number of concurrent triangle-field loader threads')
    parser.add_argument('--saver_workers', type=int, default=4,
                        help='Number of concurrent NPZ saver threads')
    parser.add_argument('--queue_size', type=int, default=16,
                        help='Maximum number of loaded sparse tensors waiting for GPU encoding')
    parser.add_argument('--load_timeout_s', type=float, default=300.0,
                        help='Seconds to wait for a loaded item before printing loader status')
    parser.add_argument('--benchmark', action='store_true',
                        help='Print per-object read/encode/save-submit timings')
    parser.add_argument('--allow_resolution_mismatch', action='store_true',
                        help='Allow encoding voxel fields at a resolution different from the VAE training config.')
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.triangle_field_voxel_root = opt.triangle_field_voxel_root or opt.root
    opt.triangle_field_latent_root = opt.triangle_field_latent_root or opt.root

    if opt.enc_model is None:
        raise ValueError('--enc_model must be specified')
    if opt.ckpt is None:
        raise ValueError('--ckpt must be specified')
    if opt.model_root is None:
        raise ValueError('--model_root must be specified')

    latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
    cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
    dataset_args = require_triangle_field_dataset_args(
        cfg,
        opt.resolution,
        allow_resolution_mismatch=opt.allow_resolution_mismatch,
    )
    expected_in_channels = max(
        slc.stop for slc in get_triangle_field_input_layout(dataset_args).values()
    )
    model_in_channels = int(cfg.models.encoder.args.in_channels)
    if model_in_channels != expected_in_channels:
        raise ValueError(
            f'Encoder expects {model_in_channels} channels, but its dataset config builds '
            f'{expected_in_channels} channels.'
        )
    encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
    ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
    encoder.load_state_dict(torch.load(ckpt_path), strict=True)
    encoder.eval()
    print(f'Loaded model from {ckpt_path}')

    latent_root = os.path.join(opt.triangle_field_latent_root, 'triangle_field_latents', latent_name)
    os.makedirs(os.path.join(latent_root, 'new_records'), exist_ok=True)

    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = read_unique_metadata(os.path.join(opt.root, 'metadata.csv'))
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = read_unique_metadata(
            os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')
        ).combine_first(metadata)
    voxel_metadata_path = os.path.join(
        opt.triangle_field_voxel_root,
        f'{dataset_args.voxel_dirname}_{opt.resolution}',
        'metadata.csv',
    )
    if not os.path.exists(voxel_metadata_path):
        raise ValueError(f'Triangle-field metadata not found: {voxel_metadata_path}')
    voxel_metadata = read_unique_metadata(voxel_metadata_path)
    metadata = voxel_metadata.combine_first(metadata)
    latent_metadata_path = os.path.join(latent_root, 'metadata.csv')
    if os.path.exists(latent_metadata_path):
        metadata = metadata.combine_first(read_unique_metadata(latent_metadata_path))
    metadata = metadata.reset_index()
    if opt.metadata_filter_csv is not None and opt.metadata_filter_csv.strip() != '':
        total_before_filter = len(metadata)
        allowed = load_metadata_filter(opt.metadata_filter_csv)
        metadata = metadata[metadata['sha256'].astype(str).isin(allowed)]
        print(
            f'Applied metadata filter: {total_before_filter} -> {len(metadata)} objects '
            f'using {opt.metadata_filter_csv}'
        )

    if opt.filter_low_aesthetic_score is not None:
        metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]

    metadata = metadata[metadata[dataset_args.voxelized_flag_column] == True]
    metadata = metadata[metadata[dataset_args.num_voxels_column] > 0]
    metadata = metadata[
        metadata[dataset_args.num_voxels_column] <= dataset_args.max_active_voxels
    ]

    if dataset_args.get('max_num_faces', None) is not None:
        metadata = metadata[metadata['num_faces'] <= dataset_args.max_num_faces]

    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]
    print(
        f'[Metadata] Eligible resolution-{opt.resolution} instances: {len(metadata)}',
        flush=True,
    )

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
        tqdm(total=len(metadata), desc="Filtering existing objects") as pbar:
        def check_sha256(sha256):
            latent_path = os.path.join(latent_root, f'{sha256}.npz')
            cache_path = os.path.join(latent_root, f'{sha256}.cache.pt')
            if os.path.exists(latent_path) and os.path.exists(cache_path):
                try:
                    with np.load(latent_path) as latent:
                        coords = latent['coords']
                        feats = latent['feats']
                        if coords.ndim != 2 or coords.shape[1] != 3:
                            raise ValueError(f'invalid coords shape {coords.shape}')
                        if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
                            raise ValueError(f'invalid feats shape {feats.shape}')
                    if os.path.getsize(cache_path) == 0:
                        raise ValueError('empty cache')
                    records.append({
                        'sha256': sha256,
                        'triangle_field_latent_encoded': True,
                        'triangle_field_latent_tokens': coords.shape[0],
                    })
                except Exception as e:
                    print(f'[Resume] Reprocessing invalid output {sha256}: {e}', flush=True)
            pbar.update()
        executor.map(check_sha256, metadata['sha256'].values)
        executor.shutdown(wait=True)
    existing_sha256 = set(r['sha256'] for r in records)
    print(f'Found {len(existing_sha256)} processed objects')
    metadata = metadata[~metadata['sha256'].isin(existing_sha256)]

    print(f'Processing {len(metadata)} objects...')

    sha256s = list(metadata['sha256'].values)
    voxel_root = os.path.join(opt.triangle_field_voxel_root, f'{dataset_args.voxel_dirname}_{opt.resolution}')
    load_queue = Queue(maxsize=opt.queue_size)
    with ThreadPoolExecutor(max_workers=opt.loader_workers) as loader_executor, \
         ThreadPoolExecutor(max_workers=opt.saver_workers) as saver_executor:
        saver_futures = []

        def loader(sha256):
            try:
                start_t = time.perf_counter()
                read_start_t = time.perf_counter()
                path = find_triangle_field_path(voxel_root, sha256)
                read_path_s = time.perf_counter() - read_start_t
                tensor_start_t = time.perf_counter()
                x = build_triangle_field_sparse_tensor(path, dataset_args)
                tensor_build_s = time.perf_counter() - tensor_start_t
                queue_put_start_t = time.perf_counter()
                load_queue.put((sha256, x, {
                    'load_total_s': time.perf_counter() - start_t,
                    'read_path_s': read_path_s,
                    'tensor_build_s': tensor_build_s,
                    'queue_put_wait_s': time.perf_counter() - queue_put_start_t,
                }))
            except Exception as e:
                print(f"[Loader Error] {sha256}: {e}")
                load_queue.put((sha256, None, None))

        loader_futures = [loader_executor.submit(loader, sha256) for sha256 in sha256s]

        def saver(sha256, pack, cache_pack):
            save_path = os.path.join(latent_root, f'{sha256}.npz')
            cache_path = os.path.join(latent_root, f'{sha256}.cache.pt')
            save_start_t = time.perf_counter()
            npz_start_t = time.perf_counter()
            atomic_save_npz(save_path, pack)
            save_npz_s = time.perf_counter() - npz_start_t
            cache_start_t = time.perf_counter()
            atomic_torch_save(cache_path, cache_pack)
            save_cache_s = time.perf_counter() - cache_start_t
            records.append({'sha256': sha256, 'triangle_field_latent_encoded': True, 'triangle_field_latent_tokens': pack['coords'].shape[0]})
            if opt.benchmark:
                print(
                    f"[Benchmark:save] {sha256}: "
                    f"save_npz_s={save_npz_s:.3f} "
                    f"save_cache_s={save_cache_s:.3f} "
                    f"save_total_s={time.perf_counter() - save_start_t:.3f}",
                    flush=True,
                )

        for _ in tqdm(range(len(sha256s)), desc="Extracting triangle-field latents"):
            num_voxels = 0
            try:
                while True:
                    try:
                        queue_get_start_t = time.perf_counter()
                        sha256, voxels, load_timing = load_queue.get(timeout=opt.load_timeout_s)
                        queue_get_s = time.perf_counter() - queue_get_start_t
                        break
                    except Empty:
                        done = sum(f.done() for f in loader_futures)
                        print(
                            f"[Wait] No loaded triangle field after {opt.load_timeout_s:.1f}s; "
                            f"loader futures done={done}/{len(loader_futures)} "
                            f"queue_size={load_queue.qsize()}"
                        )
                if voxels is None:
                    print(f"[Skip] {sha256}: Failed to load input")
                    continue

                num_voxels = voxels.feats.shape[0]

                validate_start_t = time.perf_counter()
                if not is_valid_sparse_tensor(voxels):
                    print(f"[Skip] {sha256}: NaN/Inf in input")
                    continue
                validate_s = time.perf_counter() - validate_start_t

                cuda_start_t = time.perf_counter()
                voxels_cuda = voxels.cuda()
                torch.cuda.synchronize()
                cuda_transfer_s = time.perf_counter() - cuda_start_t
                encode_start_t = time.perf_counter()
                z = encoder(voxels_cuda)
                torch.cuda.synchronize()
                encode_s = time.perf_counter() - encode_start_t

                finite_start_t = time.perf_counter()
                if not torch.isfinite(z.feats).all():
                    print(f"[Skip] {sha256}: Non-finite latent in z.feats")
                    clear_cuda_error()
                    continue
                finite_check_s = time.perf_counter() - finite_start_t

                pack_start_t = time.perf_counter()
                pack = {
                    'feats': z.feats.cpu().numpy().astype(np.float32),
                    'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                pack_s = time.perf_counter() - pack_start_t
                cache_pack_start_t = time.perf_counter()
                cache_pack = {
                    'scale': z._scale,
                    'spatial_cache': to_cpu_cache(trim_decoder_spatial_cache(z._spatial_cache)),
                }
                cache_pack_s = time.perf_counter() - cache_pack_start_t
                save_submit_start_t = time.perf_counter()
                saver_futures.append(saver_executor.submit(saver, sha256, pack, cache_pack))
                save_submit_s = time.perf_counter() - save_submit_start_t
                if opt.benchmark:
                    load_timing = load_timing or {}
                    print(
                        f"[Benchmark] {sha256}: voxels={num_voxels} "
                        f"latent_tokens={pack['coords'].shape[0]} "
                        f"queue_get_s={queue_get_s:.3f} "
                        f"load_total_s={load_timing.get('load_total_s', float('nan')):.3f} "
                        f"read_path_s={load_timing.get('read_path_s', float('nan')):.3f} "
                        f"tensor_build_s={load_timing.get('tensor_build_s', float('nan')):.3f} "
                        f"queue_put_wait_s={load_timing.get('queue_put_wait_s', float('nan')):.3f} "
                        f"validate_s={validate_s:.3f} "
                        f"cuda_transfer_s={cuda_transfer_s:.3f} "
                        f"encode_s={encode_s:.3f} "
                        f"finite_check_s={finite_check_s:.3f} "
                        f"pack_s={pack_s:.3f} "
                        f"cache_pack_s={cache_pack_s:.3f} "
                        f"save_submit_s={save_submit_s:.3f}",
                        flush=True,
                    )

            except Exception as e:
                print(f"[Error] {sha256} ({num_voxels} voxels): {e}")
                clear_cuda_error()
                continue

        saver_executor.shutdown(wait=True)
        for future in saver_futures:
            future.result()

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(latent_root, 'new_records', f'part_{opt.rank}.csv'), index=False)
