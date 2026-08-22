#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from triangle_field_zarr import (
    find_payload,
    load_payload,
    open_zipstore,
    pack_zipstore,
    read_zarr_sample,
    source_roots,
    verify_zipstore,
)


_WORKER_MODE = None
_WORKER_ROOTS = None
_WORKER_ZARR_STORE = None
_WORKER_ZARR_ROOT = None


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark triangle-field NPZ-Zstd against a Zarr ZipStore shard.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_instances", type=int, default=128)
    parser.add_argument("--resolutions", default="32,64,128,256,512")
    parser.add_argument("--lower_suffix", default="avg_from_512")
    parser.add_argument("--chunk_voxels", type=int, default=65536)
    parser.add_argument("--compression_level", type=int, default=5)
    parser.add_argument("--num_random_reads", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_instances(path, roots, resolutions, count):
    selected = []
    for instance in (line.strip() for line in path.read_text().splitlines()):
        if not instance:
            continue
        try:
            for resolution in resolutions:
                find_payload(roots[resolution], instance)
        except FileNotFoundError:
            continue
        selected.append(instance)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Found only {len(selected)} complete instances; requested {count}")
    return selected


def sample_checksum(coords, features):
    return len(coords), int(coords.astype(np.int64).sum()), float(features.astype(np.float64).sum())


def init_worker(mode, roots, zip_path):
    global _WORKER_MODE, _WORKER_ROOTS, _WORKER_ZARR_STORE, _WORKER_ZARR_ROOT
    _WORKER_MODE = mode
    _WORKER_ROOTS = roots
    if mode == "zarr":
        _WORKER_ZARR_STORE, _WORKER_ZARR_ROOT = open_zipstore(zip_path)


def worker_read(task):
    resolution, index, instance = task
    if _WORKER_MODE == "npz":
        arrays = load_payload(find_payload(_WORKER_ROOTS[resolution], instance))
    else:
        arrays = read_zarr_sample(_WORKER_ZARR_ROOT, resolution, index)
    return sample_checksum(*arrays)


def benchmark_reads(mode, tasks, roots, zip_path, workers):
    start = time.perf_counter()
    if workers == 1:
        init_worker(mode, roots, zip_path)
        checksums = [worker_read(task) for task in tasks]
        if _WORKER_ZARR_STORE is not None:
            _WORKER_ZARR_STORE.close()
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            workers,
            initializer=init_worker,
            initargs=(mode, roots, zip_path),
        ) as pool:
            checksums = pool.map(worker_read, tasks, chunksize=1)
    elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "samples_per_second": len(tasks) / elapsed,
        "voxels": int(sum(item[0] for item in checksums)),
        "checksum_coords": int(sum(item[1] for item in checksums)),
        "checksum_features": float(sum(item[2] for item in checksums)),
    }


def main():
    args = parse_args()
    resolutions = [int(value) for value in args.resolutions.split(",") if value.strip()]
    roots = source_roots(args.dataset_root, resolutions, args.lower_suffix)
    instances = select_instances(args.instances, roots, resolutions, args.num_instances)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / f"triangle_field_{len(instances)}.zarr.zip"
    manifest_path = args.output_dir / "instances.txt"
    manifest_path.write_text("\n".join(instances) + "\n")

    source_paths = [find_payload(roots[resolution], instance) for resolution in resolutions for instance in instances]
    source_bytes = sum(path.stat().st_size for path in source_paths)
    start = time.perf_counter()
    pack_zipstore(
        zip_path, roots, instances, resolutions, args.chunk_voxels, args.compression_level
    )
    write_seconds = time.perf_counter() - start
    checked_counts = verify_zipstore(zip_path, roots, instances, resolutions)
    checked_voxels = sum(checked_counts.values())

    rng = np.random.default_rng(args.seed)
    tasks = []
    for _ in range(args.num_random_reads):
        index = int(rng.integers(len(instances)))
        resolution = int(rng.choice(resolutions))
        tasks.append((resolution, index, instances[index]))

    benchmarks = {}
    for workers in (1, args.num_workers):
        npz_result = benchmark_reads("npz", tasks, roots, str(zip_path), workers)
        zarr_result = benchmark_reads("zarr", tasks, roots, str(zip_path), workers)
        if (
            npz_result["voxels"] != zarr_result["voxels"]
            or npz_result["checksum_coords"] != zarr_result["checksum_coords"]
            or not np.isclose(
                npz_result["checksum_features"], zarr_result["checksum_features"], rtol=0, atol=1e-5
            )
        ):
            raise ValueError(f"Read benchmark checksums differ for {workers} workers")
        benchmarks[str(workers)] = {"npz_zstd": npz_result, "zarr_zipstore": zarr_result}

    report = {
        "instances": len(instances),
        "resolutions": resolutions,
        "checked_voxels": checked_voxels,
        "chunk_voxels": args.chunk_voxels,
        "source": {
            "files": len(source_paths),
            "bytes": source_bytes,
        },
        "zarr_zipstore": {
            "files": 1,
            "bytes": zip_path.stat().st_size,
            "write_seconds": write_seconds,
        },
        "benchmarks": benchmarks,
    }
    report["zarr_zipstore"]["size_ratio_vs_npz"] = report["zarr_zipstore"]["bytes"] / source_bytes
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Benchmark report: {report_path}")


if __name__ == "__main__":
    main()
