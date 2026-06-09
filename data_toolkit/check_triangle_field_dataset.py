#!/usr/bin/env python3
import argparse
import concurrent.futures
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


FEATURE_LAYOUT = [
    ('d_tri', 1),
    ('d_vert', 1),
    ('offset_to_v0', 3),
    ('offset_to_v1', 3),
    ('offset_to_v2', 3),
    ('offset_to_centroid', 3),
    ('face_normal', 3),
    ('offset_to_projection', 3),
]


def layout_slices():
    slices = {}
    start = 0
    for name, width in FEATURE_LAYOUT:
        slices[name] = slice(start, start + width)
        start += width
    return slices


def load_npz(path: Path):
    if path.name.endswith('.npz.zst'):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError('Reading .npz.zst requires the zstandard package') from exc
        with open(path, 'rb') as f:
            payload = zstd.ZstdDecompressor().decompress(f.read())
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def find_voxel_file(voxel_root: Path, sha256: str) -> Path | None:
    for suffix in ('.npz.zst', '.npz'):
        path = voxel_root / f'{sha256}{suffix}'
        if path.exists():
            return path
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Check triangle-field voxel dataset feature statistics.')
    parser.add_argument('--voxel_root', type=Path, required=True,
                        help='Directory containing triangle-field .npz.zst/.npz files.')
    parser.add_argument('--metadata', type=Path, default=None,
                        help='Metadata CSV with sha256 column. Defaults to <voxel_root>/metadata.csv.')
    parser.add_argument('--instances', type=Path, default=None,
                        help='Optional instances.txt filter.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional max number of instances to scan.')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of worst instances to print.')
    parser.add_argument('--output_json', type=Path, default=None,
                        help='Optional path to write stats JSON.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker processes for file checks.')
    parser.add_argument('--chunksize', type=int, default=16,
                        help='Multiprocessing chunksize when --num_workers > 1.')
    return parser.parse_args()


def check_one(voxel_root_str: str, sha: str):
    voxel_root = Path(voxel_root_str)
    num_channels = sum(width for _, width in FEATURE_LAYOUT)
    slices = layout_slices()

    path = find_voxel_file(voxel_root, sha)
    if path is None:
        return {'status': 'missing', 'sha256': sha}
    try:
        with load_npz(path) as data:
            coords = data['coords']
            features = data['features'].astype(np.float32, copy=False)
    except Exception as exc:
        return {'status': 'bad', 'sha256': sha, 'path': str(path), 'error': repr(exc)}

    if features.ndim != 2 or features.shape[1] != num_channels:
        return {
            'status': 'bad',
            'sha256': sha,
            'path': str(path),
            'error': f'invalid features shape {features.shape}, expected (*, {num_channels})',
        }
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != features.shape[0]:
        return {
            'status': 'bad',
            'sha256': sha,
            'path': str(path),
            'error': f'invalid coords/features shapes coords={coords.shape}, features={features.shape}',
        }
    if features.shape[0] == 0:
        return {'status': 'bad', 'sha256': sha, 'path': str(path), 'error': 'zero active voxels'}

    finite = np.isfinite(features)
    if not finite.all():
        return {
            'status': 'bad',
            'sha256': sha,
            'path': str(path),
            'error': f'non-finite features count={int((~finite).sum())}',
        }

    f64 = features.astype(np.float64, copy=False)
    return {
        'status': 'ok',
        'sha256': sha,
        'path': str(path),
        'num_voxels': int(features.shape[0]),
        'count': int(features.shape[0]),
        'sum': f64.sum(axis=0),
        'sumsq': (f64 * f64).sum(axis=0),
        'min': f64.min(axis=0),
        'max': f64.max(axis=0),
        'abs_max': float(np.abs(features).max()),
        'group_abs': {
            name: float(np.abs(features[:, slc]).max())
            for name, slc in slices.items()
        },
    }


def main():
    args = parse_args()
    voxel_root = args.voxel_root
    metadata_path = args.metadata or voxel_root / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f'Metadata not found: {metadata_path}')

    metadata = pd.read_csv(metadata_path)
    if 'sha256' not in metadata.columns:
        raise ValueError(f'{metadata_path} must contain a sha256 column')
    sha256s = metadata['sha256'].astype(str).tolist()

    if args.instances is not None:
        allowed = set(line.strip() for line in args.instances.read_text().splitlines() if line.strip())
        sha256s = [sha for sha in sha256s if sha in allowed]
    if args.limit is not None:
        sha256s = sha256s[:args.limit]

    num_channels = sum(width for _, width in FEATURE_LAYOUT)
    count = np.zeros(num_channels, dtype=np.int64)
    sum_ = np.zeros(num_channels, dtype=np.float64)
    sumsq = np.zeros(num_channels, dtype=np.float64)
    min_ = np.full(num_channels, np.inf, dtype=np.float64)
    max_ = np.full(num_channels, -np.inf, dtype=np.float64)

    scanned = 0
    missing = []
    bad = []
    worst_abs = []
    worst_group_abs = {name: [] for name, _ in FEATURE_LAYOUT}
    slices = layout_slices()

    executor = None
    if args.num_workers <= 1:
        results = map(lambda sha: check_one(str(voxel_root), sha), sha256s)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers)
        results = executor.map(
            check_one,
            [str(voxel_root)] * len(sha256s),
            sha256s,
            chunksize=args.chunksize,
        )

    try:
        for result in tqdm(results, total=len(sha256s), desc='Checking triangle fields'):
            if result['status'] == 'missing':
                missing.append(result['sha256'])
                continue
            if result['status'] == 'bad':
                bad.append({
                    'sha256': result['sha256'],
                    'path': result.get('path'),
                    'error': result['error'],
                })
                continue

            count += result['count']
            sum_ += result['sum']
            sumsq += result['sumsq']
            min_ = np.minimum(min_, result['min'])
            max_ = np.maximum(max_, result['max'])
            scanned += 1

            worst_abs.append((
                result['abs_max'],
                result['sha256'],
                result['path'],
                result['num_voxels'],
            ))
            for name, group_abs in result['group_abs'].items():
                worst_group_abs[name].append((
                    group_abs,
                    result['sha256'],
                    result['path'],
                    result['num_voxels'],
                ))
    finally:
        if executor is not None:
            executor.shutdown()

    if (count == 0).any():
        raise RuntimeError('No valid feature rows were scanned')

    mean = sum_ / count
    var = np.maximum(sumsq / count - mean * mean, 0.0)
    std = np.sqrt(var)

    channel_stats = []
    for idx in range(num_channels):
        channel_stats.append({
            'channel': idx,
            'min': float(min_[idx]),
            'max': float(max_[idx]),
            'mean': float(mean[idx]),
            'std': float(std[idx]),
        })

    group_stats = {}
    for name, slc in slices.items():
        group_stats[name] = {
            'channels': list(range(slc.start, slc.stop)),
            'min': [float(v) for v in min_[slc]],
            'max': [float(v) for v in max_[slc]],
            'mean': [float(v) for v in mean[slc]],
            'std': [float(v) for v in std[slc]],
        }

    worst_abs.sort(reverse=True)
    for values in worst_group_abs.values():
        values.sort(reverse=True)

    report = {
        'voxel_root': str(voxel_root),
        'metadata': str(metadata_path),
        'requested_instances': len(sha256s),
        'scanned_instances': scanned,
        'missing_count': len(missing),
        'bad_count': len(bad),
        'total_feature_rows': int(count[0]),
        'channel_stats': channel_stats,
        'group_stats': group_stats,
        'worst_abs': [
            {'abs_max': v, 'sha256': sha, 'path': path, 'num_voxels': n}
            for v, sha, path, n in worst_abs[:args.top_k]
        ],
        'worst_group_abs': {
            name: [
                {'abs_max': v, 'sha256': sha, 'path': path, 'num_voxels': n}
                for v, sha, path, n in values[:args.top_k]
            ]
            for name, values in worst_group_abs.items()
        },
        'missing_first': missing[:args.top_k],
        'bad_first': bad[:args.top_k],
    }

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
