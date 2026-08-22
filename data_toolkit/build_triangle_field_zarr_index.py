#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from triangle_field_zarr import open_zipstore


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build verified train/holdout indices for packed triangle-field shards.'
    )
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--train_instances', type=Path, required=True)
    parser.add_argument('--holdout_instances', type=Path, required=True)
    parser.add_argument('--excluded_instances', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--resolutions', default='32,64,128,256,512')
    return parser.parse_args()


def read_instances(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main():
    args = parse_args()
    resolutions = [int(value) for value in args.resolutions.split(',') if value.strip()]
    train = set(read_instances(args.train_instances))
    holdout = set(read_instances(args.holdout_instances))
    excluded = set(read_instances(args.excluded_instances))
    if train & holdout:
        raise ValueError('Train and holdout manifests overlap.')

    rows = []
    shard_paths = sorted((args.dataset_root / 'shards').glob('shard_*/triangle_field_shard_*.zarr.zip'))
    for shard_path in shard_paths:
        if not (shard_path.parent / 'COMPLETE').is_file():
            raise ValueError(f'Shard lacks COMPLETE marker: {shard_path}')
        store, root = open_zipstore(shard_path)
        try:
            if [int(value) for value in root.attrs['resolutions']] != resolutions:
                raise ValueError(f'Resolution mismatch in {shard_path}')
            instances = [value.decode('ascii') for value in np.asarray(root['sha256'])]
            offsets = {
                resolution: np.asarray(root[f'r{resolution}/offsets'], dtype=np.int64)
                for resolution in resolutions
            }
            for local_index, sha256 in enumerate(instances):
                if sha256 in excluded:
                    continue
                split = 'train' if sha256 in train else 'holdout' if sha256 in holdout else None
                if split is None:
                    raise ValueError(f'Packed instance is absent from both splits: {sha256}')
                row = {
                    'sha256': sha256,
                    'shard_path': str(shard_path),
                    'local_index': local_index,
                    'split': split,
                }
                for resolution in resolutions:
                    row[f'num_voxels_{resolution}'] = int(
                        offsets[resolution][local_index + 1] - offsets[resolution][local_index]
                    )
                rows.append(row)
        finally:
            store.close()

    index = pd.DataFrame(rows)
    if index['sha256'].duplicated().any():
        duplicates = index.loc[index['sha256'].duplicated(), 'sha256'].head().tolist()
        raise ValueError(f'Duplicate packed instances: {duplicates}')
    expected_train = train - excluded
    expected_holdout = holdout - excluded
    actual_train = set(index.loc[index['split'] == 'train', 'sha256'])
    actual_holdout = set(index.loc[index['split'] == 'holdout', 'sha256'])
    if actual_train != expected_train or actual_holdout != expected_holdout:
        raise ValueError(
            'Packed split membership mismatch: '
            f'train missing={len(expected_train - actual_train)} extra={len(actual_train - expected_train)}, '
            f'holdout missing={len(expected_holdout - actual_holdout)} '
            f'extra={len(actual_holdout - expected_holdout)}'
        )
    if (index[[f'num_voxels_{resolution}' for resolution in resolutions]] <= 0).any().any():
        raise ValueError('Index contains a sample with no active voxels.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output_dir / 'all.csv', index=False)
    for split in ('train', 'holdout'):
        subset = index[index['split'] == split].reset_index(drop=True)
        subset.to_csv(args.output_dir / f'{split}.csv', index=False)
        (args.output_dir / f'{split}_instances.txt').write_text(
            '\n'.join(subset['sha256']) + '\n'
        )
    print(
        f'Indexed {len(index)} samples from {len(shard_paths)} shards: '
        f'train={len(actual_train)}, holdout={len(actual_holdout)}, excluded={len(excluded)}'
    )


if __name__ == '__main__':
    main()
