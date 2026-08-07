"""Benchmark latent SR training with the configured or disabled checkpointing."""

import argparse
import copy
import gc
import json
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch


RESULT_PREFIX = 'BENCHMARK_RESULT='


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--warmup_steps', type=int, default=2)
    parser.add_argument('--measure_steps', type=int, default=8)
    parser.add_argument(
        '--variant',
        choices=('configured', 'no_checkpoint'),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_variant(args):
    from trellis2 import datasets, models, trainers

    set_seed()
    with open(args.config) as file:
        cfg = json.load(file)
    if args.variant == 'no_checkpoint':
        block_args = cfg['models']['encoder']['args']['block_args']
        for stage_args in block_args:
            stage_args['use_checkpoint'] = False

    dataset_cfg = cfg['dataset']
    dataset = getattr(datasets, dataset_cfg['name'])(
        args.data_dir,
        **dataset_cfg['args'],
    )
    model_dict = {
        name: getattr(models, model_cfg['name'])(**model_cfg['args']).cuda()
        for name, model_cfg in cfg['models'].items()
    }
    trainer_args = copy.deepcopy(cfg['trainer']['args'])
    trainer_args.update(
        batch_size_per_gpu=args.batch_size,
        batch_split=1,
        num_workers=2,
        max_steps=args.warmup_steps + args.measure_steps,
        prefetch_data=False,
        i_sample=10**9,
        i_save=10**9,
        i_log=10**9,
        i_print=10**9,
    )
    output_dir = tempfile.mkdtemp(prefix=f'checkpoint-benchmark-{args.variant}-')
    trainer = getattr(trainers, cfg['trainer']['name'])(
        model_dict,
        dataset,
        output_dir=output_dir,
        load_dir=None,
        step=None,
        **trainer_args,
    )
    data_list = trainer.load_data()
    batch = data_list[0]
    workload = {
        'batch_size': int(batch['z_0'].shape[0]),
        'latent_tokens': int(batch['z_0'].feats.shape[0]),
        'field_voxels': int(batch['cond'].feats.shape[0]),
    }

    for _ in range(args.warmup_steps):
        trainer.run_step(data_list)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    durations = []
    losses = []
    for _ in range(args.measure_steps):
        start = time.perf_counter()
        step_log = trainer.run_step(data_list)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        losses.append(float(step_log['loss']['loss']))

    result = {
        'variant': args.variant,
        'warmup_steps': args.warmup_steps,
        'measure_steps': args.measure_steps,
        **workload,
        'mean_step_seconds': float(np.mean(durations)),
        'median_step_seconds': float(np.median(durations)),
        'steps_per_hour': float(3600.0 / np.mean(durations)),
        'samples_per_second': float(args.batch_size / np.mean(durations)),
        'baseline_allocated_gib': baseline_allocated / 2**30,
        'baseline_reserved_gib': baseline_reserved / 2**30,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'last_loss': losses[-1],
    }
    trainer.writer.close()
    del data_list, batch, trainer, model_dict, dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def run_parent(args):
    results = []
    for variant in ('configured', 'no_checkpoint'):
        command = [
            sys.executable,
            os.path.abspath(__file__),
            '--config', args.config,
            '--data_dir', args.data_dir,
            '--batch_size', str(args.batch_size),
            '--warmup_steps', str(args.warmup_steps),
            '--measure_steps', str(args.measure_steps),
            '--variant', variant,
        ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        print(completed.stdout, end='')
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end='')
        result_line = next(
            line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)
        )
        results.append(json.loads(result_line[len(RESULT_PREFIX):]))
    print('BENCHMARK_COMPARISON=' + json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.variant is None:
        run_parent(arguments)
    else:
        run_variant(arguments)
