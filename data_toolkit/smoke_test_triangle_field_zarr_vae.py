#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trellis2 import models
from trellis2.datasets.sparse_voxel_triangle_field import (
    MultiResolutionZarrSparseVoxelTriangleFieldDataset,
)
from trellis2.trainers.vae.triangle_field_vae import TriangleFieldVaeTrainer
from trellis2.utils.data_utils import recursive_to_device


def parse_map(value: str) -> dict[int, int]:
    return {
        int(item.split(':', 1)[0]): int(item.split(':', 1)[1])
        for item in value.split(',')
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Smoke-test and benchmark the packed all-resolution triangle-field VAE path.'
    )
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--batch_sizes', default='32:16,64:16,128:8,256:4,512:2')
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--timed_steps', type=int, default=2)
    parser.add_argument('--loader_batches', type=int, default=20)
    parser.add_argument('--load_quantile', type=float, default=0.95)
    return parser.parse_args()


def representative_indices(
    dataset,
    resolution: int,
    count: int,
    quantile: float,
) -> list[int]:
    group = dataset.resolution_groups[resolution]
    loads = np.asarray(dataset.loads[group.start:group.stop])
    target = np.quantile(loads, quantile)
    selected = np.argsort(np.abs(loads - target))[:count]
    return [group.start + int(index) for index in selected]


def load_batch(dataset, indices):
    return recursive_to_device(
        dataset.collate_fn([dataset[index] for index in indices]),
        torch.device('cuda'),
        non_blocking=False,
    )


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    batch_sizes = parse_map(args.batch_sizes)
    dataset = MultiResolutionZarrSparseVoxelTriangleFieldDataset(
        '{}', **cfg['dataset']['args']
    )
    if set(batch_sizes) != set(dataset.resolutions):
        raise ValueError(
            f'Batch sizes must cover {dataset.resolutions}, got {sorted(batch_sizes)}'
        )

    model_dict = {
        name: getattr(models, model_cfg['name'])(**model_cfg['args']).cuda()
        for name, model_cfg in cfg['models'].items()
    }
    trainer_args = dict(cfg['trainer']['args'])
    trainer_args.update({
        'batch_size_per_gpu_by_resolution': batch_sizes,
        'resolution_sampling_weights': {resolution: 1.0 for resolution in batch_sizes},
        'max_steps': 100,
        'i_log': 1000000,
        'i_sample': 1000000,
        'i_save': 1000000,
        'prefetch_data': False,
        'debug_nans': True,
    })
    with tempfile.TemporaryDirectory(prefix='trifield-zarr-vae-smoke-') as output_dir:
        trainer = TriangleFieldVaeTrainer(
            model_dict,
            dataset,
            output_dir=output_dir,
            load_dir=None,
            step=None,
            **trainer_args,
        )

        print('\nGPU forward/backward benchmark')
        results = {}
        for resolution in sorted(batch_sizes, reverse=True):
            batch_size = batch_sizes[resolution]
            indices = representative_indices(
                dataset,
                resolution,
                batch_size,
                args.load_quantile,
            )
            elapsed = []
            peak_gib = []
            last_log = None
            for iteration in range(args.warmup_steps + args.timed_steps):
                batch = load_batch(dataset, indices)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                last_log = trainer.run_step([batch])
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                if iteration >= args.warmup_steps:
                    elapsed.append(duration)
                    peak_gib.append(torch.cuda.max_memory_allocated() / 2**30)
            loss = float(last_log['loss']['loss'])
            grad_norm = float(last_log['status']['grad_norm'])
            if not np.isfinite(loss) or not np.isfinite(grad_norm):
                raise RuntimeError(
                    f'Non-finite result at resolution {resolution}: '
                    f'loss={loss}, grad_norm={grad_norm}'
                )
            results[resolution] = {
                'batch_size': batch_size,
                'seconds': float(np.mean(elapsed)),
                'peak_gib': float(max(peak_gib)),
                'loss': loss,
                'grad_norm': grad_norm,
            }
            print(f'r{resolution}: {results[resolution]}', flush=True)

        print('\nMulti-worker loader benchmark')
        loader_start = time.perf_counter()
        sample_count = 0
        voxel_count = 0
        resolution_counts = {resolution: 0 for resolution in dataset.resolutions}
        for batch_index, batch in enumerate(trainer.dataloader):
            resolutions = batch['resolution'].tolist()
            sample_count += len(resolutions)
            voxel_count += int(batch['x'].feats.shape[0])
            for resolution in resolutions:
                resolution_counts[int(resolution)] += 1
            if batch_index + 1 == args.loader_batches:
                break
        loader_seconds = time.perf_counter() - loader_start
        loader_result = {
            'batches': args.loader_batches,
            'samples': sample_count,
            'voxels': voxel_count,
            'seconds': loader_seconds,
            'samples_per_second': sample_count / loader_seconds,
            'million_voxels_per_second': voxel_count / loader_seconds / 1e6,
            'resolution_samples': resolution_counts,
        }
        print(loader_result, flush=True)
        print('\nSMOKE_TEST_PASSED', flush=True)


if __name__ == '__main__':
    main()
