"""Compute per-mesh density summary distributions from voxel payloads."""

import argparse
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm


PERCENTILES = (0, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_dir', required=True)
    parser.add_argument('--instances', required=True)
    parser.add_argument('--metadata_csv', action='append', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--per_mesh_output', default=None)
    parser.add_argument('--feature_index', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def load_features(path):
    if path.endswith('.zst'):
        import zstandard as zstd

        with open(path, 'rb') as file:
            payload = zstd.ZstdDecompressor().decompress(file.read())
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            return data['features']
    with np.load(path, allow_pickle=False) as data:
        return data['features']


def process_instance(item):
    voxel_dir, instance, feature_index = item
    try:
        path = next(
            path
            for path in (
                os.path.join(voxel_dir, f'{instance}.npz.zst'),
                os.path.join(voxel_dir, f'{instance}.npz'),
            )
            if os.path.exists(path)
        )
        features = load_features(path)
        if features.ndim != 2 or features.shape[1] <= feature_index:
            raise ValueError(f'invalid feature shape {features.shape}')
        values = features[:, feature_index].astype(np.float64, copy=False)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError('density is empty or non-finite')
        minimum = float(values.min())
        median = float(np.median(values))
        maximum = float(values.max())
        return instance, minimum, median, maximum, None
    except Exception as error:
        return instance, 0.0, 0.0, 0.0, str(error)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'percentiles': {
            f'{percentile:g}': float(value)
            for percentile, value in zip(PERCENTILES, np.percentile(values, PERCENTILES))
        },
    }


def main():
    args = parse_args()
    with open(args.instances) as file:
        instances = {line.strip() for line in file if line.strip()}
    for path in args.metadata_csv:
        metadata_instances = set(pd.read_csv(path, usecols=['sha256'])['sha256'].astype(str))
        instances.intersection_update(metadata_instances)
    instances = sorted(instances)
    if args.limit is not None:
        instances = instances[:args.limit]

    rows = []
    errors = []
    work = [(args.voxel_dir, instance, args.feature_index) for instance in instances]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for instance, minimum, median, maximum, error in tqdm(
            executor.map(process_instance, work, chunksize=8),
            total=len(work),
            desc='Density scalar statistics',
        ):
            if error is None:
                rows.append((instance, minimum, median, maximum))
            else:
                errors.append(f'{instance}: {error}')
    if not rows:
        raise RuntimeError('No valid density payloads were found')

    values = np.asarray([row[1:] for row in rows], dtype=np.float64)
    stats = {
        'voxel_dir': os.path.abspath(args.voxel_dir),
        'instances_path': os.path.abspath(args.instances),
        'metadata_csv': [os.path.abspath(path) for path in args.metadata_csv],
        'feature_index': args.feature_index,
        'resolution': 128,
        'density_scale': 'base_128_native_payload_values',
        'meshes_requested': len(instances),
        'meshes_valid': len(rows),
        'meshes_skipped': len(errors),
        'minimum': summarize(values[:, 0]),
        'median': summarize(values[:, 1]),
        'maximum': summarize(values[:, 2]),
        'median_minus_minimum': summarize(values[:, 1] - values[:, 0]),
        'maximum_minus_median': summarize(values[:, 2] - values[:, 1]),
        'maximum_minus_minimum': summarize(values[:, 2] - values[:, 0]),
        'errors': errors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as file:
        json.dump(stats, file, indent=2)
    if args.per_mesh_output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.per_mesh_output)), exist_ok=True)
        pd.DataFrame(
            rows,
            columns=(
                'sha256',
                'density_field_min_128',
                'density_field_median_128',
                'density_field_max_128',
            ),
        ).to_csv(args.per_mesh_output, index=False)
    print(json.dumps({key: value for key, value in stats.items() if key != 'errors'}, indent=2))
    print(f'Wrote {args.output}; skipped {len(errors)} meshes.')
    if args.per_mesh_output is not None:
        print(f'Wrote {args.per_mesh_output}.')


if __name__ == '__main__':
    main()
