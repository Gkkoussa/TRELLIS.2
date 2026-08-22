#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

from triangle_field_zarr import (
    find_payload,
    pack_zipstore,
    source_roots,
    verify_zipstore,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack and exactly verify one immutable triangle-field ZipStore shard."
    )
    parser.add_argument("--staging_root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", default="32,64,128,256,512")
    parser.add_argument("--lower_suffix", default="hybrid_avg_from_512")
    parser.add_argument("--chunk_voxels", type=int, default=65536)
    parser.add_argument("--compression_level", type=int, default=5)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--verify_existing", action="store_true")
    parser.add_argument(
        "--staging-retained",
        dest="staging_retained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record whether source staging survives after this command.",
    )
    return parser.parse_args()


def read_instances(path: Path) -> list[str]:
    instances = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not instances:
        raise ValueError(f"No instances in {path}")
    if len(instances) != len(set(instances)):
        raise ValueError(f"Duplicate instances in {path}")
    return instances


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, instances: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sha256", "index"))
        writer.writeheader()
        for index, sha256 in enumerate(instances):
            writer.writerow({"sha256": sha256, "index": index})


def main():
    args = parse_args()
    resolutions = [int(value) for value in args.resolutions.split(",") if value.strip()]
    instances = read_instances(args.instances)
    roots = source_roots(args.staging_root, resolutions, args.lower_suffix)
    source_paths = [
        find_payload(roots[resolution], instance)
        for resolution in resolutions
        for instance in instances
    ]
    source_bytes = sum(path.stat().st_size for path in source_paths)
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}
    attrs = {
        "lower_resolution_method": "active-child average d_tri/d_vert; target-resolution recomputed aux",
        "source_resolution": 512,
        "base_feature_channels": 20,
        "density_and_elongation_included": False,
        "provenance": provenance,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.output.exists():
        if not args.verify_existing:
            raise FileExistsError(f"Output already exists: {args.output}")
        verified_counts = verify_zipstore(args.output, roots, instances, resolutions)
        pack_seconds = None
    else:
        temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
        packed_counts = pack_zipstore(
            temporary,
            roots,
            instances,
            resolutions,
            args.chunk_voxels,
            args.compression_level,
            attrs=attrs,
        )
        verified_counts = verify_zipstore(temporary, roots, instances, resolutions)
        if packed_counts != verified_counts:
            raise ValueError(f"Packed and verified voxel counts differ: {packed_counts} != {verified_counts}")
        pack_seconds = time.perf_counter() - started
        if args.output.exists():
            raise FileExistsError(f"Output appeared while packing: {args.output}")
        os.replace(temporary, args.output)

    digest_started = time.perf_counter()
    output_digest = file_sha256(args.output)
    digest_seconds = time.perf_counter() - digest_started
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.csv")
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    write_manifest(manifest_path, instances)
    report = {
        "format": "triangle_field_zarr_shard_report",
        "version": 1,
        "output": str(args.output),
        "output_sha256": output_digest,
        "output_bytes": args.output.stat().st_size,
        "instances": len(instances),
        "resolutions": resolutions,
        "voxel_counts": {str(key): value for key, value in verified_counts.items()},
        "total_voxels": sum(verified_counts.values()),
        "source_files": len(source_paths),
        "source_bytes": source_bytes,
        "size_ratio_vs_staging_npz": args.output.stat().st_size / source_bytes,
        "pack_and_verify_seconds": pack_seconds,
        "sha256_seconds": digest_seconds,
        "exact_coordinate_and_feature_verification": True,
        "staging_retained": args.staging_retained,
        "provenance": provenance,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
