#!/usr/bin/env python3
"""Build a targeted QEM recovery list from validation failures and missing outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--qem_root", type=Path, default=None)
    parser.add_argument("--triangle_field_voxel_root", type=Path, default=None)
    parser.add_argument("--invalid_instances", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--require_empty", action="store_true")
    return parser.parse_args()


def truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def read_sha_lines(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation instance list: {path}")
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    }


def main() -> None:
    args = parse_args()
    qem_root = args.qem_root or (
        args.root / f"qem_edge_collapsed_meshes_{args.resolution}"
    )
    voxel_root = args.triangle_field_voxel_root or (
        args.root / f"triangle_field_voxels_{args.resolution}"
    )
    invalid_path = args.invalid_instances or qem_root / "invalid_instances.txt"
    output_path = args.output or qem_root / "recovery_instances.txt"
    summary_path = args.summary_json or qem_root / "recovery_summary.json"

    base_path = args.root / "metadata.csv"
    voxel_metadata_path = voxel_root / "metadata.csv"
    qem_metadata_path = qem_root / "metadata.csv"
    for path in (base_path, voxel_metadata_path, qem_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required metadata: {path}")

    base = pd.read_csv(base_path)
    voxel = pd.read_csv(voxel_metadata_path)
    qem = pd.read_csv(qem_metadata_path)
    if "sha256" not in base or "sha256" not in voxel or "sha256" not in qem:
        raise ValueError("Base, voxel, and QEM metadata must contain sha256")

    if "triangle_field_voxelized" in voxel:
        voxel = voxel[truthy_series(voxel["triangle_field_voxelized"])]
    if "num_triangle_field_voxels" in voxel:
        voxel = voxel[
            pd.to_numeric(voxel["num_triangle_field_voxels"], errors="coerce")
            .fillna(0)
            .gt(0)
        ]

    available = set(base["sha256"].astype(str)) & set(voxel["sha256"].astype(str))
    if "pbr_dumped" in base:
        available &= set(
            base.loc[truthy_series(base["pbr_dumped"]), "sha256"].astype(str)
        )
    available = {
        sha
        for sha in available
        if (args.root / "pbr_dumps" / f"{sha}.pickle").is_file()
    }

    if "qem_edge_collapsed" in qem:
        completed = set(
            qem.loc[truthy_series(qem["qem_edge_collapsed"]), "sha256"].astype(str)
        )
    else:
        completed = set(qem["sha256"].astype(str))

    invalid_reported = read_sha_lines(invalid_path)
    invalid_available = invalid_reported & available
    absent_from_metadata = available - completed
    recovery = sorted(invalid_available | absent_from_metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{sha}\n" for sha in recovery))
    summary = {
        "resolution": args.resolution,
        "available": len(available),
        "metadata_completed": len(completed & available),
        "validation_invalid_reported": len(invalid_reported),
        "validation_invalid_available": len(invalid_available),
        "absent_from_metadata": len(absent_from_metadata),
        "recovery_instances": len(recovery),
        "invalid_not_available": len(invalid_reported - available),
        "output": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.require_empty and recovery:
        raise SystemExit(
            f"QEM recovery is not clean: {len(recovery)} instance(s) still require work"
        )


if __name__ == "__main__":
    main()
