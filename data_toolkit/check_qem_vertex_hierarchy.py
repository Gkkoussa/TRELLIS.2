#!/usr/bin/env python3
"""Measure whether independently generated QEM vertex targets form a 2x hierarchy."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_RESOLUTIONS = [32, 64, 128, 256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For every QEM vertex at 2R, test whether floor(coord / 2) is "
            "(1) in the R triangle-field sparse support and (2) in the R QEM vertex set."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help="Ordered 2x resolutions to compare (default: 32 64 128 256 512).",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=None,
        help="Optional instances.txt restriction.",
    )
    parser.add_argument(
        "--metadata_filter_csv",
        default=None,
        help="Optional comma-separated metadata CSVs to intersect by sha256.",
    )
    parser.add_argument(
        "--limit_per_pair",
        type=int,
        default=None,
        help="Optional sorted-instance limit for a quick audit.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument(
        "--max_missing_examples",
        type=int,
        default=1000,
        help="Maximum detailed unreachable-child examples saved per resolution pair.",
    )
    parser.add_argument(
        "--examples_per_mesh",
        type=int,
        default=4,
        help="Maximum unreachable-child coordinates returned by one mesh worker.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <root>/outputs/qem_vertex_hierarchy_audit.",
    )
    return parser.parse_args()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def metadata_instances(path: Path, success_column: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}")
    metadata = pd.read_csv(path)
    if "sha256" not in metadata:
        raise ValueError(f"{path} must contain a sha256 column")
    if success_column in metadata:
        metadata = metadata[truthy(metadata[success_column])]
    return set(metadata["sha256"].astype(str))


def filter_instances(path: Path) -> set[str]:
    metadata = pd.read_csv(path)
    if "sha256" not in metadata:
        raise ValueError(f"{path} must contain a sha256 column")
    if "local_density_filter_keep" in metadata:
        metadata = metadata[truthy(metadata["local_density_filter_keep"])]
    elif "has_local_dense_region" in metadata:
        metadata = metadata[~truthy(metadata["has_local_dense_region"])]
    return set(metadata["sha256"].astype(str))


def find_payload(root: Path, instance: str) -> Path | None:
    for suffix in (".npz.zst", ".npz"):
        path = root / f"{instance}{suffix}"
        if path.is_file():
            return path
    return None


def load_npz(path: Path):
    if path.name.endswith(".npz.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("Reading .npz.zst requires zstandard") from exc
        with path.open("rb") as handle:
            payload = zstd.ZstdDecompressor().decompress(handle.read())
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def validate_coords(
    coords: np.ndarray,
    resolution: int,
    label: str,
    *,
    require_unique: bool,
) -> np.ndarray:
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{label} has invalid shape {coords.shape}")
    if len(coords) == 0:
        raise ValueError(f"{label} is empty")
    if not np.issubdtype(coords.dtype, np.integer):
        if not np.equal(coords, np.floor(coords)).all():
            raise ValueError(f"{label} contains non-integer coordinates")
    coords = coords.astype(np.int64, copy=False)
    if (coords < 0).any() or (coords >= resolution).any():
        raise ValueError(f"{label} contains coordinates outside [0, {resolution})")
    if require_unique and len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"{label} contains duplicate coordinates")
    return coords


def coord_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = coords.astype(np.int64, copy=False)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def check_one(task: tuple) -> dict:
    (
        instance,
        coarse_resolution,
        fine_resolution,
        coarse_qem_root,
        fine_qem_root,
        coarse_support_root,
        examples_per_mesh,
    ) = task
    roots = {
        "coarse_qem": Path(coarse_qem_root),
        "fine_qem": Path(fine_qem_root),
        "coarse_support": Path(coarse_support_root),
    }
    paths = {name: find_payload(root, instance) for name, root in roots.items()}
    missing_files = [name for name, path in paths.items() if path is None]
    if missing_files:
        return {
            "status": "error",
            "sha256": instance,
            "error": f"missing payload(s): {missing_files}",
        }

    try:
        with load_npz(paths["coarse_qem"]) as payload:
            coarse_vertices = validate_coords(
                payload["vertex_voxel_coords"],
                coarse_resolution,
                "coarse QEM vertex_voxel_coords",
                require_unique=True,
            )
        with load_npz(paths["fine_qem"]) as payload:
            fine_vertices = validate_coords(
                payload["vertex_voxel_coords"],
                fine_resolution,
                "fine QEM vertex_voxel_coords",
                require_unique=True,
            )
        with load_npz(paths["coarse_support"]) as payload:
            coarse_support = validate_coords(
                payload["coords"],
                coarse_resolution,
                "coarse triangle-field coords",
                require_unique=True,
            )
    except Exception as exc:
        return {
            "status": "error",
            "sha256": instance,
            "error": repr(exc),
        }

    parent_coords = fine_vertices // 2
    parent_keys = coord_keys(parent_coords, coarse_resolution)
    coarse_vertex_keys = coord_keys(coarse_vertices, coarse_resolution)
    coarse_support_keys = coord_keys(coarse_support, coarse_resolution)

    parent_in_support = np.isin(parent_keys, coarse_support_keys)
    parent_is_coarse_vertex = np.isin(parent_keys, coarse_vertex_keys)
    coarse_vertices_in_support = np.isin(coarse_vertex_keys, coarse_support_keys)

    required_parent_keys, children_per_required_parent = np.unique(
        parent_keys, return_counts=True
    )
    required_parent_in_support = np.isin(required_parent_keys, coarse_support_keys)
    required_parent_is_vertex = np.isin(required_parent_keys, coarse_vertex_keys)

    # Count fine GT children belonging to every coarse GT vertex. A valid 2x
    # voxel hierarchy permits 0 through 8 unique children per parent.
    child_count_by_coarse_vertex = np.zeros(len(coarse_vertex_keys), dtype=np.int64)
    positions = np.searchsorted(required_parent_keys, coarse_vertex_keys)
    matches = positions < len(required_parent_keys)
    matches[matches] &= required_parent_keys[positions[matches]] == coarse_vertex_keys[matches]
    child_count_by_coarse_vertex[matches] = children_per_required_parent[positions[matches]]
    if len(child_count_by_coarse_vertex) and child_count_by_coarse_vertex.max() > 8:
        raise ValueError(
            f"{instance}: a coarse voxel has more than eight unique fine children"
        )
    child_histogram = np.bincount(child_count_by_coarse_vertex, minlength=9)[:9]

    unreachable = ~parent_is_coarse_vertex
    unreachable_indices = np.flatnonzero(unreachable)[:examples_per_mesh]
    missing_examples = []
    for index in unreachable_indices:
        missing_examples.append(
            {
                "sha256": instance,
                "fine_x": int(fine_vertices[index, 0]),
                "fine_y": int(fine_vertices[index, 1]),
                "fine_z": int(fine_vertices[index, 2]),
                "parent_x": int(parent_coords[index, 0]),
                "parent_y": int(parent_coords[index, 1]),
                "parent_z": int(parent_coords[index, 2]),
                "parent_in_coarse_support": bool(parent_in_support[index]),
                "parent_is_coarse_qem_vertex": bool(parent_is_coarse_vertex[index]),
            }
        )

    fine_count = len(fine_vertices)
    coarse_count = len(coarse_vertices)
    fine_parent_in_support = int(parent_in_support.sum())
    fine_parent_is_vertex = int(parent_is_coarse_vertex.sum())
    coarse_with_children = int((child_count_by_coarse_vertex > 0).sum())
    row = {
        "sha256": instance,
        "coarse_resolution": coarse_resolution,
        "fine_resolution": fine_resolution,
        "coarse_active_voxels": len(coarse_support),
        "coarse_qem_vertices": coarse_count,
        "fine_qem_vertices": fine_count,
        "fine_parent_in_support": fine_parent_in_support,
        "fine_parent_is_coarse_vertex": fine_parent_is_vertex,
        "fine_parent_outside_support": int((~parent_in_support).sum()),
        "fine_parent_active_but_not_vertex": int(
            (parent_in_support & ~parent_is_coarse_vertex).sum()
        ),
        "support_coverage": safe_ratio(fine_parent_in_support, fine_count),
        "fine_vertex_coverage": safe_ratio(fine_parent_is_vertex, fine_count),
        "unique_required_parents": len(required_parent_keys),
        "unique_required_parent_in_support": int(required_parent_in_support.sum()),
        "unique_required_parent_is_vertex": int(required_parent_is_vertex.sum()),
        "unique_parent_support_coverage": safe_ratio(
            int(required_parent_in_support.sum()), len(required_parent_keys)
        ),
        "unique_parent_vertex_coverage": safe_ratio(
            int(required_parent_is_vertex.sum()), len(required_parent_keys)
        ),
        "coarse_vertices_in_support": int(coarse_vertices_in_support.sum()),
        "coarse_vertices_with_fine_children": coarse_with_children,
        "dead_end_coarse_vertices": coarse_count - coarse_with_children,
        "coarse_continuation_rate": safe_ratio(coarse_with_children, coarse_count),
        "perfect_support_mesh": bool(parent_in_support.all()),
        "perfect_vertex_parent_mesh": bool(parent_is_coarse_vertex.all()),
    }
    return {
        "status": "ok",
        "row": row,
        "child_histogram": child_histogram.tolist(),
        "missing_examples": missing_examples,
    }


def percentile_summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {}
    return {
        "min": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pair_worklist(
    root: Path,
    coarse_resolution: int,
    fine_resolution: int,
    allowed: set[str] | None,
) -> list[str]:
    metadata_sources = [
        (
            root / f"qem_edge_collapsed_meshes_{coarse_resolution}" / "metadata.csv",
            "qem_edge_collapsed",
        ),
        (
            root / f"qem_edge_collapsed_meshes_{fine_resolution}" / "metadata.csv",
            "qem_edge_collapsed",
        ),
        (
            root / f"triangle_field_voxels_{coarse_resolution}" / "metadata.csv",
            "triangle_field_voxelized",
        ),
    ]
    available = None
    for path, success_column in metadata_sources:
        current = metadata_instances(path, success_column)
        available = current if available is None else available & current
    if allowed is not None:
        available &= allowed
    return sorted(available)


def summarize_pair(
    coarse_resolution: int,
    fine_resolution: int,
    requested_instances: int,
    rows: list[dict],
    errors: list[dict],
    child_histogram: np.ndarray,
) -> dict:
    totals = {
        key: int(sum(int(row[key]) for row in rows))
        for key in (
            "coarse_active_voxels",
            "coarse_qem_vertices",
            "fine_qem_vertices",
            "fine_parent_in_support",
            "fine_parent_is_coarse_vertex",
            "fine_parent_outside_support",
            "fine_parent_active_but_not_vertex",
            "unique_required_parents",
            "unique_required_parent_in_support",
            "unique_required_parent_is_vertex",
            "coarse_vertices_in_support",
            "coarse_vertices_with_fine_children",
            "dead_end_coarse_vertices",
        )
    }
    fine_count = totals["fine_qem_vertices"]
    required_parent_count = totals["unique_required_parents"]
    coarse_count = totals["coarse_qem_vertices"]
    return {
        "coarse_resolution": coarse_resolution,
        "fine_resolution": fine_resolution,
        "requested_instances": requested_instances,
        "scanned_instances": len(rows),
        "error_instances": len(errors),
        "perfect_support_meshes": int(sum(row["perfect_support_mesh"] for row in rows)),
        "perfect_vertex_parent_meshes": int(
            sum(row["perfect_vertex_parent_mesh"] for row in rows)
        ),
        "micro": {
            "support_coverage": safe_ratio(
                totals["fine_parent_in_support"], fine_count
            ),
            "fine_vertex_coverage": safe_ratio(
                totals["fine_parent_is_coarse_vertex"], fine_count
            ),
            "unique_parent_support_coverage": safe_ratio(
                totals["unique_required_parent_in_support"], required_parent_count
            ),
            "unique_parent_vertex_coverage": safe_ratio(
                totals["unique_required_parent_is_vertex"], required_parent_count
            ),
            "coarse_vertex_support_coverage": safe_ratio(
                totals["coarse_vertices_in_support"], coarse_count
            ),
            "coarse_continuation_rate": safe_ratio(
                totals["coarse_vertices_with_fine_children"], coarse_count
            ),
            **totals,
        },
        "per_mesh_distribution": {
            key: percentile_summary(row[key] for row in rows)
            for key in (
                "support_coverage",
                "fine_vertex_coverage",
                "unique_parent_support_coverage",
                "unique_parent_vertex_coverage",
                "coarse_continuation_rate",
            )
        },
        "children_per_coarse_qem_vertex_histogram": {
            str(child_count): int(count)
            for child_count, count in enumerate(child_histogram.tolist())
        },
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    resolutions = [int(value) for value in args.resolutions]
    if len(resolutions) < 2:
        raise ValueError("At least two resolutions are required")
    for coarse, fine in zip(resolutions, resolutions[1:]):
        if fine != 2 * coarse:
            raise ValueError(f"Expected 2x adjacent resolutions, got {coarse} -> {fine}")
    if args.num_workers <= 0 or args.chunksize <= 0:
        raise ValueError("num_workers and chunksize must be positive")
    if args.limit_per_pair is not None and args.limit_per_pair <= 0:
        raise ValueError("limit_per_pair must be positive")
    if args.max_missing_examples < 0 or args.examples_per_mesh < 0:
        raise ValueError("missing-example limits must be non-negative")

    allowed = None
    if args.instances is not None:
        allowed = {
            line.strip()
            for line in args.instances.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if args.metadata_filter_csv:
        for value in args.metadata_filter_csv.split(","):
            value = value.strip()
            if not value:
                continue
            path = Path(value).resolve()
            current = filter_instances(path)
            allowed = current if allowed is None else allowed & current

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "outputs" / "qem_vertex_hierarchy_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_errors: list[dict] = []
    all_missing_examples: list[dict] = []
    pair_summaries = []

    for coarse_resolution, fine_resolution in zip(resolutions, resolutions[1:]):
        instances = pair_worklist(root, coarse_resolution, fine_resolution, allowed)
        if args.limit_per_pair is not None:
            instances = instances[: args.limit_per_pair]
        print(
            f"{coarse_resolution} -> {fine_resolution}: "
            f"{len(instances)} metadata-complete instances",
            flush=True,
        )
        task_suffix = (
            coarse_resolution,
            fine_resolution,
            str(root / f"qem_edge_collapsed_meshes_{coarse_resolution}"),
            str(root / f"qem_edge_collapsed_meshes_{fine_resolution}"),
            str(root / f"triangle_field_voxels_{coarse_resolution}"),
            args.examples_per_mesh,
        )
        tasks = ((instance, *task_suffix) for instance in instances)
        executor = None
        if args.num_workers == 1:
            results = map(check_one, tasks)
        else:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_workers
            )
            results = executor.map(check_one, tasks, chunksize=args.chunksize)

        pair_rows = []
        pair_errors = []
        pair_missing_examples = []
        child_histogram = np.zeros(9, dtype=np.int64)
        try:
            for result in tqdm(
                results,
                total=len(instances),
                desc=f"Checking {coarse_resolution}->{fine_resolution}",
            ):
                if result["status"] != "ok":
                    pair_errors.append(
                        {
                            "coarse_resolution": coarse_resolution,
                            "fine_resolution": fine_resolution,
                            "sha256": result["sha256"],
                            "error": result["error"],
                        }
                    )
                    continue
                pair_rows.append(result["row"])
                child_histogram += np.asarray(
                    result["child_histogram"], dtype=np.int64
                )
                remaining = args.max_missing_examples - len(pair_missing_examples)
                if remaining > 0:
                    for example in result["missing_examples"][:remaining]:
                        pair_missing_examples.append(
                            {
                                "coarse_resolution": coarse_resolution,
                                "fine_resolution": fine_resolution,
                                **example,
                            }
                        )
        finally:
            if executor is not None:
                executor.shutdown()

        pair_rows.sort(key=lambda row: (row["fine_vertex_coverage"], row["sha256"]))
        summary = summarize_pair(
            coarse_resolution,
            fine_resolution,
            len(instances),
            pair_rows,
            pair_errors,
            child_histogram,
        )
        pair_summaries.append(summary)
        all_rows.extend(pair_rows)
        all_errors.extend(pair_errors)
        all_missing_examples.extend(pair_missing_examples)
        print(json.dumps(summary, indent=2), flush=True)

    report = {
        "root": str(root),
        "resolutions": resolutions,
        "instances_filter": str(args.instances) if args.instances else None,
        "metadata_filter_csv": args.metadata_filter_csv,
        "pairs": pair_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "per_mesh.csv", all_rows)
    write_csv(output_dir / "missing_examples.csv", all_missing_examples)
    write_csv(output_dir / "errors.csv", all_errors)
    print(f"Saved hierarchy audit to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
