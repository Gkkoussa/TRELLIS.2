#!/usr/bin/env python3
import argparse
import concurrent.futures
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average-downsample triangle-field voxel payloads from a high-resolution source."
    )
    parser.add_argument("--source_root", type=Path, required=True, help="Directory containing source .npz/.npz.zst files.")
    parser.add_argument("--output_root", type=Path, required=True, help="Dataset root where output dirs will be created.")
    parser.add_argument("--source_resolution", type=int, default=512)
    parser.add_argument("--resolutions", type=str, default="32,64,128,256")
    parser.add_argument("--instances", type=Path, default=None, help="Optional instances.txt. Defaults to source metadata order.")
    parser.add_argument("--source_metadata", type=Path, default=None, help="Defaults to <source_root>/metadata.csv.")
    parser.add_argument("--output_prefix", type=str, default="triangle_field_voxels")
    parser.add_argument("--output_suffix", type=str, default="avg_from_512")
    parser.add_argument("--feature_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compression", choices=("zstd", "npz"), default="zstd")
    parser.add_argument("--zstd_level", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_npz(path: Path):
    if path.name.endswith(".npz.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("Reading .npz.zst requires zstandard") from exc
        with open(path, "rb") as f:
            payload = zstd.ZstdDecompressor().decompress(f.read())
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def find_source_file(source_root: Path, sha256: str) -> Path:
    for suffix in (".npz.zst", ".npz"):
        path = source_root / f"{sha256}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No source triangle-field voxel file found for {sha256} in {source_root}")


def average_downsample(coords: np.ndarray, features: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    if factor == 1:
        return coords.astype(np.int32, copy=False), features.astype(np.float32, copy=False)
    parent = np.floor_divide(coords, factor).astype(np.int32, copy=False)
    unique_parent, inverse = np.unique(parent, axis=0, return_inverse=True)
    sums = np.zeros((unique_parent.shape[0], features.shape[1]), dtype=np.float32)
    counts = np.zeros((unique_parent.shape[0], 1), dtype=np.float32)
    np.add.at(sums, inverse, features.astype(np.float32, copy=False))
    np.add.at(counts, inverse, 1.0)
    return unique_parent.astype(np.int32, copy=False), sums / np.maximum(counts, 1.0)


def average_downsample_torch(
    coords: np.ndarray,
    features: np.ndarray,
    factor: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if factor == 1:
        return coords.astype(np.int32, copy=False), features.astype(np.float32, copy=False)
    coords_t = torch.from_numpy(coords.astype(np.int32, copy=False)).to(device=device, non_blocking=True)
    feats_t = torch.from_numpy(features.astype(np.float32, copy=False)).to(device=device, non_blocking=True)
    parent = torch.div(coords_t, factor, rounding_mode="floor").int()
    unique_parent, inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    sums = torch.zeros(
        (unique_parent.shape[0], feats_t.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.zeros((unique_parent.shape[0], 1), dtype=torch.float32, device=device)
    sums.index_add_(0, inverse, feats_t)
    counts.index_add_(0, inverse, torch.ones((feats_t.shape[0], 1), dtype=torch.float32, device=device))
    out_features = sums / counts.clamp_min(1.0)
    return unique_parent.cpu().numpy().astype(np.int32, copy=False), out_features.cpu().numpy()


def save_triangle_field(path: Path, coords: np.ndarray, features: np.ndarray, compression: str, zstd_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "npz":
        np.savez(path, coords=coords, features=features)
        return

    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError("Writing .npz.zst requires zstandard") from exc

    payload = io.BytesIO()
    np.savez(payload, coords=coords, features=features)
    compressed = zstd.ZstdCompressor(level=zstd_level).compress(payload.getvalue())
    with open(path, "wb") as f:
        f.write(compressed)


def process_one(
    sha256: str,
    source_root: str,
    output_dirs: dict[int, str],
    source_resolution: int,
    feature_dtype: str,
    compression: str,
    zstd_level: int,
    device: str,
    skip_existing: bool,
) -> dict:
    source_path = find_source_file(Path(source_root), sha256)
    extension = ".npz.zst" if compression == "zstd" else ".npz"
    output_paths = {
        res: Path(output_dir) / f"{sha256}{extension}"
        for res, output_dir in output_dirs.items()
    }
    if skip_existing and all(path.exists() for path in output_paths.values()):
        counts = {}
        for res, path in output_paths.items():
            with load_npz(path) as data:
                counts[str(res)] = int(data["coords"].shape[0])
        return {"sha256": sha256, "status": "skipped", "counts": counts}

    with load_npz(source_path) as data:
        coords = data["coords"].astype(np.int32, copy=False)
        features = data["features"].astype(np.float32, copy=False)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{source_path} has invalid coords shape {coords.shape}")
    if features.ndim != 2 or features.shape[0] != coords.shape[0]:
        raise ValueError(f"{source_path} has invalid features shape {features.shape}")

    counts = {}
    out_dtype = np.float16 if feature_dtype == "float16" else np.float32
    for res, output_path in output_paths.items():
        if source_resolution % res != 0:
            raise ValueError(f"source_resolution={source_resolution} is not divisible by resolution={res}")
        factor = source_resolution // res
        if device == "cuda":
            out_coords, out_features = average_downsample_torch(coords, features, factor, device)
        else:
            out_coords, out_features = average_downsample(coords, features, factor)
        out_features = out_features.astype(out_dtype, copy=False)
        save_triangle_field(output_path, out_coords, out_features, compression, zstd_level)
        counts[str(res)] = int(out_coords.shape[0])
    return {"sha256": sha256, "status": "ok", "counts": counts}


def main():
    args = parse_args()
    source_metadata = args.source_metadata or args.source_root / "metadata.csv"
    if not source_metadata.exists():
        raise FileNotFoundError(f"Source metadata not found: {source_metadata}")
    metadata = pd.read_csv(source_metadata)
    if "sha256" not in metadata.columns:
        raise ValueError(f"{source_metadata} must contain a sha256 column")

    if args.instances is None:
        instances = metadata["sha256"].astype(str).tolist()
    else:
        instances = [line.strip() for line in args.instances.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        instances = instances[:args.limit]

    resolutions = [int(item) for item in args.resolutions.split(",") if item.strip()]
    if len(resolutions) == 0:
        raise ValueError("--resolutions must contain at least one resolution")

    output_dirs = {
        res: args.output_root / f"{args.output_prefix}_{res}_{args.output_suffix}"
        for res in resolutions
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    errors = []
    kwargs = {
        "source_root": str(args.source_root),
        "output_dirs": {res: str(path) for res, path in output_dirs.items()},
        "source_resolution": args.source_resolution,
        "feature_dtype": args.feature_dtype,
        "compression": args.compression,
        "zstd_level": args.zstd_level,
        "device": args.device,
        "skip_existing": args.skip_existing,
    }
    if args.device == "cuda" and args.max_workers != 1:
        print("Warning: --device cuda uses a single worker to avoid multiple processes contending for one GPU.")
        args.max_workers = 1

    if args.max_workers <= 1:
        iterator = (process_one(sha, **kwargs) for sha in instances)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers)
        iterator = executor.map(
            process_one,
            instances,
            [kwargs["source_root"]] * len(instances),
            [kwargs["output_dirs"]] * len(instances),
            [kwargs["source_resolution"]] * len(instances),
            [kwargs["feature_dtype"]] * len(instances),
            [kwargs["compression"]] * len(instances),
            [kwargs["zstd_level"]] * len(instances),
            [kwargs["device"]] * len(instances),
            [kwargs["skip_existing"]] * len(instances),
            chunksize=args.chunksize,
        )

    try:
        for result in tqdm(iterator, total=len(instances), desc="Downsampling triangle fields"):
            records.append(result)
    except Exception as exc:
        errors.append(repr(exc))
        raise
    finally:
        if "executor" in locals():
            executor.shutdown(cancel_futures=True)

    by_sha = {record["sha256"]: record for record in records}
    for res, output_dir in output_dirs.items():
        rows = []
        for sha in instances:
            if sha not in by_sha:
                continue
            rows.append(
                {
                    "sha256": sha,
                    "triangle_field_voxelized": True,
                    "num_triangle_field_voxels": by_sha[sha]["counts"][str(res)],
                }
            )
        pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)
        with open(output_dir / "instances.txt", "w") as f:
            for row in rows:
                f.write(f"{row['sha256']}\n")
        summary = {
            "source_root": str(args.source_root),
            "source_resolution": args.source_resolution,
            "resolution": res,
            "num_instances": len(rows),
            "feature_dtype": args.feature_dtype,
            "compression": args.compression,
            "zstd_level": args.zstd_level if args.compression == "zstd" else None,
            "device": args.device,
            "output_dir": str(output_dir),
        }
        with open(output_dir / "downsample_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps({
        "num_instances": len(records),
        "resolutions": resolutions,
        "output_dirs": {str(res): str(path) for res, path in output_dirs.items()},
        "errors": errors,
    }, indent=2))


if __name__ == "__main__":
    main()
