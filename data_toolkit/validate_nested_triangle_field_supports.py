#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from voxelize_triangle_field import load_triangle_field_npz


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate lower-resolution triangle-field supports derived from a source support.'
    )
    parser.add_argument('--nested_root', type=Path, required=True)
    parser.add_argument('--source_root', type=Path, required=True)
    parser.add_argument('--instances', type=Path, required=True)
    parser.add_argument('--resolutions', type=str, default='16,32,64,128,256,512')
    parser.add_argument('--source_resolution', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--output_json', type=Path, default=None)
    return parser.parse_args()


def find_payload(root: Path, resolution: int, sha256: str) -> Path:
    directory = root / f'triangle_field_voxels_{resolution}'
    for suffix in ('.npz.zst', '.npz'):
        path = directory / f'{sha256}{suffix}'
        if path.exists():
            return path
    raise FileNotFoundError(f'missing resolution-{resolution} payload for {sha256} under {directory}')


def load_coords(root: Path, resolution: int, sha256: str) -> np.ndarray:
    with load_triangle_field_npz(str(find_payload(root, resolution, sha256))) as data:
        return np.asarray(data['coords'], dtype=np.int32)


def linear_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = coords.astype(np.int64, copy=False)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def check_instance(
    sha256: str,
    nested_root: Path,
    source_root: Path,
    resolutions: tuple[int, ...],
    source_resolution: int,
):
    source_coords = load_coords(source_root, source_resolution, sha256)
    coords_by_resolution = {source_resolution: source_coords}
    exact_mismatches = {}

    for resolution in resolutions:
        if resolution == source_resolution:
            continue
        actual = load_coords(nested_root, resolution, sha256)
        expected = np.unique(source_coords // (source_resolution // resolution), axis=0)
        actual_keys = np.unique(linear_keys(actual, resolution))
        expected_keys = np.unique(linear_keys(expected, resolution))
        missing = int(np.setdiff1d(expected_keys, actual_keys, assume_unique=True).size)
        extra = int(np.setdiff1d(actual_keys, expected_keys, assume_unique=True).size)
        exact_mismatches[str(resolution)] = {
            'missing': missing,
            'extra': extra,
        }
        coords_by_resolution[resolution] = actual

    parent_stats = {}
    for low_resolution, high_resolution in zip(resolutions[:-1], resolutions[1:]):
        low_keys = np.unique(linear_keys(coords_by_resolution[low_resolution], low_resolution))
        high_coords = coords_by_resolution[high_resolution]
        parent = high_coords // (high_resolution // low_resolution)
        missing = int((~np.isin(linear_keys(parent, low_resolution), low_keys)).sum())
        parent_stats[f'{low_resolution}->{high_resolution}'] = {
            'missing_high_voxels': missing,
            'high_voxels': int(high_coords.shape[0]),
        }

    return {
        'sha256': sha256,
        'exact_mismatches': exact_mismatches,
        'parent_stats': parent_stats,
    }


def main():
    args = parse_args()
    resolutions = tuple(sorted(int(value) for value in args.resolutions.split(',')))
    if resolutions[-1] != args.source_resolution:
        raise ValueError('The highest --resolutions entry must equal --source_resolution')
    for low, high in zip(resolutions[:-1], resolutions[1:]):
        if high % low != 0:
            raise ValueError(f'resolution {high} must be divisible by {low}')

    instances = [
        line.strip()
        for line in args.instances.read_text().splitlines()
        if line.strip()
    ]
    totals = {
        f'{low}->{high}': {'missing_high_voxels': 0, 'high_voxels': 0}
        for low, high in zip(resolutions[:-1], resolutions[1:])
    }
    exact_missing = {str(resolution): 0 for resolution in resolutions[:-1]}
    exact_extra = {str(resolution): 0 for resolution in resolutions[:-1]}
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                check_instance,
                sha256,
                args.nested_root,
                args.source_root,
                resolutions,
                args.source_resolution,
            ): sha256
            for sha256 in instances
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc='Validating nested supports',
        ):
            sha256 = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({'sha256': sha256, 'error': repr(exc)})
                continue
            for resolution, mismatch in result['exact_mismatches'].items():
                exact_missing[resolution] += mismatch['missing']
                exact_extra[resolution] += mismatch['extra']
            for pair, stats in result['parent_stats'].items():
                totals[pair]['missing_high_voxels'] += stats['missing_high_voxels']
                totals[pair]['high_voxels'] += stats['high_voxels']

    for stats in totals.values():
        stats['missing_fraction'] = (
            stats['missing_high_voxels'] / stats['high_voxels']
            if stats['high_voxels']
            else None
        )
    report = {
        'instances_requested': len(instances),
        'failures': len(failures),
        'failure_examples': failures[:20],
        'exact_support_missing': exact_missing,
        'exact_support_extra': exact_extra,
        'high_voxel_parent_coverage': totals,
        'valid': (
            not failures
            and not any(exact_missing.values())
            and not any(exact_extra.values())
            and not any(stats['missing_high_voxels'] for stats in totals.values())
        ),
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    if not report['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
