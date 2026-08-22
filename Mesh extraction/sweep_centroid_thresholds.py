"""
Sweep centroid-blob thresholds and count connected components.

This is for matching the number of face-centroid blobs to the GT face count
before building the centroid-nearest mesh.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from extract_barycentric_ridge_graph import connected_components, load_triangle_field


def parse_float_list(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("Expected at least one float value")
    return out


def parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("Expected at least one int value")
    return out


def count_obj_triangles(path: str | Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] != "f":
                continue
            vertex_count = len(parts) - 1
            if vertex_count >= 3:
                count += vertex_count - 2
    return count


def component_size_stats(sizes: list[int]) -> dict[str, float | int]:
    if not sizes:
        return {
            "min_size": 0,
            "mean_size": 0.0,
            "median_size": 0.0,
            "max_size": 0,
            "total_voxels": 0,
        }
    arr = np.asarray(sizes, dtype=np.float32)
    return {
        "min_size": int(arr.min()),
        "mean_size": float(arr.mean()),
        "median_size": float(np.median(arr)),
        "max_size": int(arr.max()),
        "total_voxels": int(arr.sum()),
    }


def count_centroids(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    *,
    dtri_threshold: float,
    max_dvert: float,
    min_component_size: int,
    connectivity: int,
) -> dict[str, int | float]:
    mask = (d_tri >= float(dtri_threshold)) & (d_vert <= float(max_dvert))
    components = connected_components(coords, mask, connectivity=connectivity)
    kept_sizes = []
    rejected_small = 0
    for component in components:
        size = int(component.size)
        if size < int(min_component_size):
            rejected_small += 1
            continue
        kept_sizes.append(size)

    size_stats = component_size_stats(kept_sizes)
    return {
        "centroids": int(len(kept_sizes)),
        "all_components": int(len(components)),
        "rejected_small": int(rejected_small),
        "mask_voxels": int(mask.sum()),
        **size_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep centroid thresholds and count components.")
    parser.add_argument(
        "field",
        type=str,
        help="Path to triangle-field .npz or .npz.zst",
    )
    parser.add_argument("--gt_obj", type=str, default=None)
    parser.add_argument("--target_centroids", type=int, default=None)
    parser.add_argument("--dtri_thresholds", type=str, default="0.56,0.58,0.60,0.62,0.64")
    parser.add_argument("--max_dverts", type=str, default="0.52,0.55,0.58,0.61")
    parser.add_argument("--min_component_sizes", type=str, default="1,2,3,4")
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--out_csv", type=str, default="Mesh extraction/results/centroid_threshold_sweep.csv")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    if args.target_centroids is None and args.gt_obj is None:
        raise ValueError("Pass --target_centroids or --gt_obj")

    target = int(args.target_centroids) if args.target_centroids is not None else count_obj_triangles(args.gt_obj)
    dtri_thresholds = parse_float_list(args.dtri_thresholds)
    max_dverts = parse_float_list(args.max_dverts)
    min_component_sizes = parse_int_list(args.min_component_sizes)

    coords, features = load_triangle_field(args.field)
    d_tri = np.asarray(features[:, 0], dtype=np.float32)
    d_vert = np.asarray(features[:, 1], dtype=np.float32)

    rows = []
    for dtri_threshold in dtri_thresholds:
        for max_dvert in max_dverts:
            for min_component_size in min_component_sizes:
                stats = count_centroids(
                    coords,
                    d_tri,
                    d_vert,
                    dtri_threshold=dtri_threshold,
                    max_dvert=max_dvert,
                    min_component_size=min_component_size,
                    connectivity=args.connectivity,
                )
                diff = int(stats["centroids"]) - target
                fill_needed = max(target - int(stats["centroids"]), 0)
                fillable_to_target = int(int(stats["centroids"]) <= target <= int(stats["centroids"]) + int(stats["rejected_small"]))
                rows.append({
                    "fillable_to_target": fillable_to_target,
                    "fill_needed": int(fill_needed),
                    "abs_diff": abs(diff),
                    "diff": diff,
                    "target_centroids": target,
                    "centroids": int(stats["centroids"]),
                    "dtri_threshold": float(dtri_threshold),
                    "max_dvert": float(max_dvert),
                    "min_component_size": int(min_component_size),
                    "connectivity": int(args.connectivity),
                    "all_components": int(stats["all_components"]),
                    "rejected_small": int(stats["rejected_small"]),
                    "mask_voxels": int(stats["mask_voxels"]),
                    "kept_total_voxels": int(stats["total_voxels"]),
                    "min_size": int(stats["min_size"]),
                    "mean_size": float(stats["mean_size"]),
                    "median_size": float(stats["median_size"]),
                    "max_size": int(stats["max_size"]),
                })

    rows.sort(key=lambda row: (
        -int(row["fillable_to_target"]),
        int(row["fill_needed"]) if int(row["fillable_to_target"]) else int(row["abs_diff"]),
        int(row["abs_diff"]),
        abs(float(row["dtri_threshold"]) - 0.60),
        abs(float(row["max_dvert"]) - 0.55),
        int(row["min_component_size"]),
    ))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fillable_to_target",
        "fill_needed",
        "abs_diff",
        "diff",
        "target_centroids",
        "centroids",
        "dtri_threshold",
        "max_dvert",
        "min_component_size",
        "connectivity",
        "all_components",
        "rejected_small",
        "mask_voxels",
        "kept_total_voxels",
        "min_size",
        "mean_size",
        "median_size",
        "max_size",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Centroid threshold sweep")
    print(f"Target centroids: {target}")
    print(f"Connectivity: {args.connectivity}")
    print(f"Saved {out_csv}")
    print(f"Top {min(args.top_k, len(rows))} closest settings:")
    for row in rows[: args.top_k]:
        print(
            "  "
            f"centroids={row['centroids']} diff={row['diff']} "
            f"d_tri>={row['dtri_threshold']} d_vert<={row['max_dvert']} "
            f"min_size={row['min_component_size']} "
            f"fill_needed={row['fill_needed']} fillable={row['fillable_to_target']} "
            f"mask_voxels={row['mask_voxels']} rejected_small={row['rejected_small']}"
        )


if __name__ == "__main__":
    main()
