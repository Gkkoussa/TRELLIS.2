import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import time
import torch
import numpy as np
import pandas as pd
import o_voxel
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)


def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


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

    The decoder restores SparseTensor._spatial_cache from this file and uses
    SparseChannel2Spatial, which looks for channel2spatial_* entries at each
    scale. Shape entries are kept because they are lightweight and describe the
    sparse grid at that scale. Expensive conv/attention/layout caches are
    encoder-side artifacts and are not needed for decoding.
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--gaussian_distance_voxel_root', type=str, default=None,
                        help='Directory to save the gaussian distance voxel files')
    parser.add_argument('--gaussian_distance_latent_root', type=str, default=None,
                        help='Directory to save the gaussian distance latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Sparse voxel resolution')
    parser.add_argument('--enc_pretrained', type=str, default=None,
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str,
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--loader_workers', type=int, default=4,
                        help='Number of concurrent VXZ loader threads')
    parser.add_argument('--saver_workers', type=int, default=4,
                        help='Number of concurrent NPZ saver threads')
    parser.add_argument('--read_threads', type=int, default=1,
                        help='num_threads passed to o_voxel.io.read_vxz for each file')
    parser.add_argument('--queue_size', type=int, default=16,
                        help='Maximum number of loaded sparse tensors waiting for GPU encoding')
    parser.add_argument('--load_timeout_s', type=float, default=300.0,
                        help='Seconds to wait for a loaded item before printing loader status')
    parser.add_argument('--benchmark', action='store_true',
                        help='Print per-object read/encode/save-submit timings')
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.gaussian_distance_voxel_root = opt.gaussian_distance_voxel_root or opt.root
    opt.gaussian_distance_latent_root = opt.gaussian_distance_latent_root or opt.root

    if opt.enc_model is None:
        if opt.enc_pretrained is None:
            raise ValueError('Either --enc_model or --enc_pretrained must be specified')
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}_{opt.resolution}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        if opt.ckpt is None:
            raise ValueError('--ckpt must be specified when --enc_model is used')
        latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    os.makedirs(os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.gaussian_distance_voxel_root, f'gaussian_distance_voxels_{opt.resolution}', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.gaussian_distance_voxel_root, f'gaussian_distance_voxels_{opt.resolution}', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['gaussian_distance_voxelized'] == True]
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

    # filter out objects that are already processed
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
        tqdm(total=len(metadata), desc="Filtering existing objects") as pbar:
        def check_sha256(sha256):
            latent_path = os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, f'{sha256}.npz')
            cache_path = os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, f'{sha256}.cache.pt')
            if os.path.exists(latent_path) and os.path.exists(cache_path):
                coords = np.load(os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, f'{sha256}.npz'))['coords']
                records.append({'sha256': sha256, 'gaussian_distance_latent_encoded': True, 'gaussian_distance_latent_tokens': coords.shape[0]})
            pbar.update()
        executor.map(check_sha256, metadata['sha256'].values)
        executor.shutdown(wait=True)
    existing_sha256 = set(r['sha256'] for r in records)
    print(f'Found {len(existing_sha256)} processed objects')
    metadata = metadata[~metadata['sha256'].isin(existing_sha256)]

    print(f'Processing {len(metadata)} objects...')

    sha256s = list(metadata['sha256'].values)
    load_queue = Queue(maxsize=opt.queue_size)
    with ThreadPoolExecutor(max_workers=opt.loader_workers) as loader_executor, \
         ThreadPoolExecutor(max_workers=opt.saver_workers) as saver_executor:

        def loader(sha256):
            try:
                start_t = time.perf_counter()
                attrs = ['base_color', 'emissive']
                read_start_t = time.perf_counter()
                coords, attr = o_voxel.io.read_vxz(
                    os.path.join(opt.gaussian_distance_voxel_root, f'gaussian_distance_voxels_{opt.resolution}', f'{sha256}.vxz'),
                    num_threads=opt.read_threads
                )
                read_vxz_s = time.perf_counter() - read_start_t
                tensor_start_t = time.perf_counter()
                feats = torch.concat([attr[k] for k in attrs], dim=-1) / 255.0 * 2 - 1
                x = sp.SparseTensor(
                    feats.float(),
                    torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
                )
                tensor_build_s = time.perf_counter() - tensor_start_t
                queue_put_start_t = time.perf_counter()
                load_queue.put((sha256, x, {
                    'load_total_s': time.perf_counter() - start_t,
                    'read_vxz_s': read_vxz_s,
                    'tensor_build_s': tensor_build_s,
                    'queue_put_wait_s': time.perf_counter() - queue_put_start_t,
                }))
            except Exception as e:
                print(f"[Loader Error] {sha256}: {e}")
                load_queue.put((sha256, None, None))

        loader_futures = [loader_executor.submit(loader, sha256) for sha256 in sha256s]

        def saver(sha256, pack, cache_pack):
            save_path = os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, f'{sha256}.npz')
            cache_path = os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, f'{sha256}.cache.pt')
            save_start_t = time.perf_counter()
            npz_start_t = time.perf_counter()
            np.savez_compressed(save_path, **pack)
            save_npz_s = time.perf_counter() - npz_start_t
            cache_start_t = time.perf_counter()
            torch.save(cache_pack, cache_path)
            save_cache_s = time.perf_counter() - cache_start_t
            records.append({'sha256': sha256, 'gaussian_distance_latent_encoded': True, 'gaussian_distance_latent_tokens': pack['coords'].shape[0]})
            if opt.benchmark:
                print(
                    f"[Benchmark:save] {sha256}: "
                    f"save_npz_s={save_npz_s:.3f} "
                    f"save_cache_s={save_cache_s:.3f} "
                    f"save_total_s={time.perf_counter() - save_start_t:.3f}",
                    flush=True,
                )

        for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
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
                            f"[Wait] No loaded VXZ after {opt.load_timeout_s:.1f}s; "
                            f"loader futures done={done}/{len(loader_futures)} "
                            f"queue_size={load_queue.qsize()}"
                        )
                if voxels is None:
                    print(f"[Skip] {sha256}: Failed to load input")
                    continue

                num_voxels = voxels.feats.shape[0]

                # NaN/Inf
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
                saver_executor.submit(saver, sha256, pack, cache_pack)
                save_submit_s = time.perf_counter() - save_submit_start_t
                if opt.benchmark:
                    load_timing = load_timing or {}
                    print(
                        f"[Benchmark] {sha256}: voxels={num_voxels} "
                        f"latent_tokens={pack['coords'].shape[0]} "
                        f"queue_get_s={queue_get_s:.3f} "
                        f"load_total_s={load_timing.get('load_total_s', float('nan')):.3f} "
                        f"read_vxz_s={load_timing.get('read_vxz_s', float('nan')):.3f} "
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

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.gaussian_distance_latent_root, 'gaussian_distance_latents', latent_name, 'new_records', f'part_{opt.rank}.csv'), index=False)
