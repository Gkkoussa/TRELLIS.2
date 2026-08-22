"""Estimate global filtered-training density normalization for point flow."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from trellis2.datasets.point_density_mesh import load_normalized_mesh, sample_surface_field


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--metadata_filter_csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--field_name', choices=['density', 'elongation'], default='density')
    parser.add_argument('--samples_per_mesh', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def filtered_metadata(root: str, filter_csv: str) -> pd.DataFrame:
    metadata = pd.read_csv(os.path.join(root, 'metadata.csv')).set_index('sha256')
    keep = None
    for path in [item.strip() for item in filter_csv.split(',') if item.strip()]:
        current = set(pd.read_csv(path, usecols=['sha256'])['sha256'].astype(str))
        keep = current if keep is None else keep.intersection(current)
    if keep is not None:
        metadata = metadata[metadata.index.astype(str).isin(keep)]
    return metadata[metadata['local_path'].notna()].reset_index()


def process_mesh(args):
    root, sha256, local_path, field_name, resolution, samples_per_mesh, seed = args
    path = local_path if os.path.isabs(local_path) else os.path.join(root, local_path)
    try:
        mesh = load_normalized_mesh(path)
        mesh_seed = int.from_bytes(
            hashlib.sha256(f'{seed}:{sha256}'.encode()).digest()[:4],
            byteorder='little',
            signed=False,
        )
        _, _, density = sample_surface_field(
            mesh,
            samples_per_mesh,
            field_name=field_name,
            resolution=resolution,
            seed=mesh_seed,
        )
        density = density.astype(np.float64)
        return float(density.sum()), float(np.square(density).sum()), density.size, None
    except Exception as error:
        return 0.0, 0.0, 0, f'{sha256}: {error}'


def main():
    args = parse_args()
    metadata = filtered_metadata(args.root, args.metadata_filter_csv)
    work = [
        (
            args.root,
            row.sha256,
            row.local_path,
            args.field_name,
            args.resolution,
            args.samples_per_mesh,
            args.seed,
        )
        for row in metadata.itertuples(index=False)
    ]
    total = 0.0
    total_sq = 0.0
    count = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for value_sum, value_sq_sum, value_count, error in tqdm(
            executor.map(process_mesh, work),
            total=len(work),
            desc='Density statistics',
        ):
            total += value_sum
            total_sq += value_sq_sum
            count += value_count
            if error is not None:
                errors.append(error)
    if count == 0:
        raise RuntimeError('No valid density samples were collected')
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    stats = {
        'mean': mean,
        'std': variance ** 0.5,
        'count': count,
        'resolution': args.resolution,
        'field_name': args.field_name,
        'samples_per_mesh': args.samples_per_mesh,
        'meshes_total': len(work),
        'meshes_valid': len(work) - len(errors),
        'mesh_normalization': 'bbox_center_unit_max_extent_0.99999',
        'field_formula': (
            '-log(vertex_mean_incident_area_barycentric / voxel_size^2 + 1e-8)'
            if args.field_name == 'density'
            else 'vertex_mean_incident(-log(clamp(4*sqrt(3)*area/sum(edge_length^2),1e-8,1)))_barycentric'
        ),
        'metadata_filter_csv': args.metadata_filter_csv,
        'errors': errors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as file:
        json.dump(stats, file, indent=2)
    print(json.dumps({key: value for key, value in stats.items() if key != 'errors'}, indent=2))
    print(f'Wrote {args.output}; skipped {len(errors)} meshes.')


if __name__ == '__main__':
    main()
