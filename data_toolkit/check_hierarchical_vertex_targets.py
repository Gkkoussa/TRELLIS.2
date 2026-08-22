#!/usr/bin/env python3
"""Validate complete-chain hierarchical vertex target payloads."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from check_triangle_field_vertex_support import (
    coord_keys,
    find_payload,
    load_npz,
    load_support_coords,
    load_worklist,
    read_instances,
    truthy,
)
from generate_hierarchical_vertex_targets import target_dir, unique_coords


DEFAULT_RESOLUTIONS = (32, 64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hierarchical_vertex_target_root", type=Path, required=True)
    parser.add_argument("--triangle_field_voxel_root", type=Path, required=True)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument(
        "--skip_support_check",
        action="store_true",
        help="Skip re-reading triangle-field payloads; structural checks still run.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--invalid_instances", type=Path, default=None)
    parser.add_argument("--failures_csv", type=Path, default=None)
    return parser.parse_args()


def validate_resolutions(values: list[int]) -> tuple[int, ...]:
    resolutions = tuple(sorted(int(value) for value in values))
    if not resolutions:
        raise ValueError("--resolutions must not be empty")
    for coarse, fine in zip(resolutions, resolutions[1:]):
        if fine != coarse * 2:
            raise ValueError("--resolutions must form a consecutive 2x hierarchy")
    return resolutions


def metadata_instances(
    target_root: Path,
    resolutions: tuple[int, ...],
) -> tuple[list[str], dict[int, set[str]]]:
    stage_sets: dict[int, set[str]] = {}
    for resolution in resolutions:
        metadata_path = target_dir(target_root, resolution) / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        if "sha256" not in metadata:
            raise ValueError(f"{metadata_path} must contain sha256")
        if "hierarchical_vertex_targets_generated" in metadata:
            metadata = metadata[
                metadata["hierarchical_vertex_targets_generated"]
                .astype(str)
                .map(truthy)
            ]
        stage_sets[resolution] = set(metadata["sha256"].astype(str))
    common = set.intersection(*(stage_sets[resolution] for resolution in resolutions))
    return sorted(common), stage_sets


def validate_coords(
    value: np.ndarray,
    resolution: int,
    label: str,
    *,
    allow_empty: bool,
) -> np.ndarray:
    coords = np.asarray(value)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{label} has invalid shape {coords.shape}")
    if not allow_empty and len(coords) == 0:
        raise ValueError(f"{label} is empty")
    if not np.issubdtype(coords.dtype, np.integer):
        if not np.equal(coords, np.floor(coords)).all():
            raise ValueError(f"{label} contains non-integer values")
    coords = coords.astype(np.int32, copy=False)
    if (coords < 0).any() or (coords >= resolution).any():
        raise ValueError(f"{label} has coordinates outside [0, {resolution})")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"{label} contains duplicate coordinates")
    return coords


def check_one(task: tuple) -> dict[str, object]:
    (
        target_root_string,
        triangle_field_root_string,
        sha256,
        resolutions,
        check_support,
    ) = task
    target_root = Path(target_root_string)
    triangle_field_root = Path(triangle_field_root_string)
    positives: dict[int, np.ndarray] = {}
    per_resolution: dict[int, dict[str, int]] = {}
    source_counts: list[int] = []
    retained_counts: list[int] = []
    excluded_counts: list[int] = []
    try:
        for resolution in resolutions:
            directory = target_dir(target_root, resolution)
            path = find_payload(directory, sha256)
            if path is None:
                raise FileNotFoundError(
                    f"missing R{resolution} hierarchical target for {sha256}"
                )
            with load_npz(path) as payload:
                required = {
                    "vertices",
                    "vertex_voxel_coords",
                    "ignored_vertex_voxel_coords",
                    "resolution",
                    "hierarchy_resolutions",
                    "num_source_vertices",
                    "num_retained_source_vertices",
                    "num_excluded_source_vertices",
                    "metadata_json",
                }
                missing = required - set(payload.files)
                if missing:
                    raise ValueError(f"{path} missing arrays {sorted(missing)}")
                vertices = np.asarray(payload["vertices"])
                positive = validate_coords(
                    payload["vertex_voxel_coords"],
                    resolution,
                    f"R{resolution} vertex_voxel_coords",
                    allow_empty=False,
                )
                ignored = validate_coords(
                    payload["ignored_vertex_voxel_coords"],
                    resolution,
                    f"R{resolution} ignored_vertex_voxel_coords",
                    allow_empty=True,
                )
                saved_resolution = int(np.asarray(payload["resolution"]).item())
                saved_hierarchy = tuple(
                    int(value) for value in np.asarray(payload["hierarchy_resolutions"])
                )
                source_count = int(np.asarray(payload["num_source_vertices"]).item())
                retained_count = int(
                    np.asarray(payload["num_retained_source_vertices"]).item()
                )
                excluded_count = int(
                    np.asarray(payload["num_excluded_source_vertices"]).item()
                )
                metadata = json.loads(
                    str(np.asarray(payload["metadata_json"]).item())
                )

            if saved_resolution != resolution:
                raise ValueError(
                    f"{path} says resolution {saved_resolution}, expected {resolution}"
                )
            if saved_hierarchy != tuple(resolutions):
                raise ValueError(
                    f"{path} hierarchy {saved_hierarchy} != {tuple(resolutions)}"
                )
            if source_count <= 0 or retained_count <= 0 or excluded_count < 0:
                raise ValueError(f"{path} contains invalid source/retained counts")
            if retained_count + excluded_count != source_count:
                raise ValueError(f"{path} source counts do not add up")
            if vertices.shape != positive.shape or not np.isfinite(vertices).all():
                raise ValueError(f"{path} has invalid vertices shape/values")
            expected_centers = (
                (positive.astype(np.float64) + 0.5) / resolution - 0.5
            )
            if not np.allclose(vertices, expected_centers, atol=2e-6):
                raise ValueError(f"{path} vertices are not voxel centers")
            if len(ignored) and np.isin(
                coord_keys(positive, resolution),
                coord_keys(ignored, resolution),
            ).any():
                raise ValueError(f"{path} positive and ignore targets overlap")
            if int(metadata["num_positive_vertex_voxels"]) != len(positive):
                raise ValueError(f"{path} metadata positive count mismatch")
            if int(metadata["num_ignored_vertex_voxels"]) != len(ignored):
                raise ValueError(f"{path} metadata ignore count mismatch")

            if check_support:
                support_root = (
                    triangle_field_root / f"triangle_field_voxels_{resolution}"
                )
                support_path = find_payload(support_root, sha256)
                if support_path is None:
                    raise FileNotFoundError(
                        f"missing R{resolution} triangle-field support for {sha256}"
                    )
                support = load_support_coords(support_path, resolution)
                support_keys = coord_keys(support, resolution)
                if not np.isin(
                    coord_keys(positive, resolution), support_keys
                ).all():
                    raise ValueError(f"{path} positive target outside support")
                if len(ignored) and not np.isin(
                    coord_keys(ignored, resolution), support_keys
                ).all():
                    raise ValueError(f"{path} ignore target outside support")

            positives[resolution] = positive
            source_counts.append(source_count)
            retained_counts.append(retained_count)
            excluded_counts.append(excluded_count)
            per_resolution[resolution] = {
                "positive_voxels": int(len(positive)),
                "ignored_voxels": int(len(ignored)),
            }

        if len(set(source_counts)) != 1:
            raise ValueError("num_source_vertices differs across resolutions")
        if len(set(retained_counts)) != 1:
            raise ValueError("num_retained_source_vertices differs across resolutions")
        if len(set(excluded_counts)) != 1:
            raise ValueError("num_excluded_source_vertices differs across resolutions")
        for coarse, fine in zip(resolutions, resolutions[1:]):
            parent = unique_coords(positives[fine] // 2)
            if not np.array_equal(parent, positives[coarse]):
                raise ValueError(
                    f"R{coarse} targets do not exactly equal parents of R{fine} targets"
                )

        return {
            "status": "ok",
            "sha256": sha256,
            "source_vertices": source_counts[0],
            "retained_source_vertices": retained_counts[0],
            "excluded_source_vertices": excluded_counts[0],
            "per_resolution": per_resolution,
        }
    except Exception as exc:
        return {"status": "bad", "sha256": sha256, "error": repr(exc)}


def main() -> None:
    args = parse_args()
    resolutions = validate_resolutions(args.resolutions)
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")
    common_instances, stage_sets = metadata_instances(
        args.hierarchical_vertex_target_root, resolutions
    )
    expected_instances = load_worklist(args.root, resolutions, args.instances, None)
    if args.instances is not None:
        allowed = read_instances(args.instances)
        common_instances = [
            sha256 for sha256 in common_instances if sha256 in allowed
        ]
    instances = sorted(set(common_instances) & set(expected_instances))
    print(
        f"Validating {len(instances)} common hierarchical target meshes at "
        f"resolutions {list(resolutions)}; support_check={not args.skip_support_check}",
        flush=True,
    )

    tasks = [
        (
            str(args.hierarchical_vertex_target_root),
            str(args.triangle_field_voxel_root),
            sha256,
            resolutions,
            not args.skip_support_check,
        )
        for sha256 in instances
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        results = list(
            tqdm(
                executor.map(check_one, tasks, chunksize=args.chunksize),
                total=len(tasks),
                desc="Checking hierarchical vertex targets",
            )
        )

    ok = [result for result in results if result["status"] == "ok"]
    bad = [result for result in results if result["status"] == "bad"]
    union = set.union(*(stage_sets[resolution] for resolution in resolutions))
    expected_set = set(expected_instances)
    metadata_incomplete = sorted(
        sha256
        for sha256 in expected_set | union
        if sha256 in expected_set
        and any(sha256 not in stage_sets[resolution] for resolution in resolutions)
    )
    failures = bad + [
        {
            "status": "metadata_incomplete",
            "sha256": sha256,
            "error": "instance is not present in successful metadata at every resolution",
        }
        for sha256 in metadata_incomplete
    ]

    validation_root = (
        args.hierarchical_vertex_target_root
        / "hierarchical_vertex_targets_validation"
    )
    validation_root.mkdir(parents=True, exist_ok=True)
    invalid_path = args.invalid_instances or validation_root / "invalid_instances.txt"
    failures_path = args.failures_csv or validation_root / "validation_failures.csv"
    output_path = args.output_json or validation_root / "validation_stats.json"
    invalid_path.write_text(
        "".join(
            f'{result["sha256"]}\n'
            for result in sorted(failures, key=lambda item: item["sha256"])
        )
    )
    pd.DataFrame(
        [
            {
                "sha256": result["sha256"],
                "status": result["status"],
                "error": result.get("error", ""),
            }
            for result in failures
        ],
        columns=["sha256", "status", "error"],
    ).to_csv(failures_path, index=False)

    source_vertices = sum(int(result["source_vertices"]) for result in ok)
    retained_vertices = sum(
        int(result["retained_source_vertices"]) for result in ok
    )
    summary = {
        "resolutions": list(resolutions),
        "metadata_rows_by_resolution": {
            str(resolution): len(stage_sets[resolution])
            for resolution in resolutions
        },
        "common_metadata_instances": len(instances),
        "expected_instances": len(expected_instances),
        "valid": len(ok),
        "bad": len(bad),
        "metadata_incomplete": len(metadata_incomplete),
        "support_membership_rechecked": not args.skip_support_check,
        "source_vertices": source_vertices,
        "retained_source_vertices": retained_vertices,
        "excluded_source_vertices": source_vertices - retained_vertices,
        "source_vertex_retention_fraction": (
            retained_vertices / source_vertices if source_vertices else 0.0
        ),
        "per_resolution": {
            str(resolution): {
                "positive_voxels": sum(
                    int(result["per_resolution"][resolution]["positive_voxels"])
                    for result in ok
                ),
                "ignored_voxels": sum(
                    int(result["per_resolution"][resolution]["ignored_voxels"])
                    for result in ok
                ),
            }
            for resolution in resolutions
        },
        "invalid_instances": str(invalid_path),
        "failures_csv": str(failures_path),
        "bad_examples": bad[:100],
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
