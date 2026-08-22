"""
Visualize BRG edge extraction up to stage 3 only.

This stops before vertex-contact filtering, midpoint checks, accepted/rejected
edge decisions, and mesh construction:
  1. Select edge-looking voxels from d_tri/d_vert thresholds.
  2. Remove vertex-core clearance voxels.
  3. Connected-component the remaining edge voxels.

The output PLY colors each remaining connected edge component with a stable
component color.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from extract_barycentric_ridge_graph import (
    build_coord_to_index,
    connected_components,
    default_output_stem,
    default_results_dir,
    dilate_sparse_mask,
    extract_vertex_blobs,
    infer_resolution,
    load_triangle_field,
    threshold_to_raw_max_barycentric,
    voxel_centers,
)


def component_color(component_id: int) -> tuple[int, int, int]:
    # Stable bright-ish pseudo-random color from component id.
    x = (int(component_id) + 1) * 2654435761
    r = 70 + ((x >> 0) & 127)
    g = 70 + ((x >> 8) & 127)
    b = 70 + ((x >> 16) & 127)
    return int(r), int(g), int(b)


def build_stage3_labels(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    *,
    vertex_dvert_threshold: float = 0.84,
    vertex_connectivity: int = 18,
    vertex_min_component_size: int = 1,
    vertex_position_mode: str = "weighted",
    vertex_core_mode: str = "closest4",
    edge_dtri_threshold: float = 0.175,
    edge_min_dvert: float = 0.25,
    edge_max_dvert: float | None = None,
    edge_connectivity: int = 18,
    vertex_clearance: int = 1,
    include_vertex_context: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    d_tri = np.asarray(d_tri, dtype=np.float32).reshape(-1)
    d_vert = np.asarray(d_vert, dtype=np.float32).reshape(-1)
    if edge_max_dvert is None:
        edge_max_dvert = vertex_dvert_threshold

    coord_to_index = build_coord_to_index(coords)
    vertex_data = extract_vertex_blobs(
        coords,
        d_vert,
        resolution,
        threshold=vertex_dvert_threshold,
        connectivity=vertex_connectivity,
        min_component_size=vertex_min_component_size,
        position_mode=vertex_position_mode,
        core_mode=vertex_core_mode,
    )
    vertex_core_mask = vertex_data["vertex_core_mask"]
    vertex_clear_mask = dilate_sparse_mask(
        coords,
        vertex_core_mask,
        coord_to_index,
        radius=vertex_clearance,
    )

    raw_edge_mask = (
        (d_tri <= float(edge_dtri_threshold))
        & (d_vert >= float(edge_min_dvert))
        & (d_vert <= float(edge_max_dvert))
    )
    cleared_edge_mask = raw_edge_mask & (~vertex_clear_mask)
    components = connected_components(coords, cleared_edge_mask, connectivity=edge_connectivity)

    labels = np.full(coords.shape[0], -999, dtype=np.int32)
    component_sizes = np.zeros(coords.shape[0], dtype=np.int32)
    for component_id, component in enumerate(components):
        labels[component] = int(component_id)
        component_sizes[component] = int(component.size)

    if include_vertex_context:
        labels[(vertex_clear_mask & ~cleared_edge_mask) & (labels == -999)] = -2
        labels[vertex_core_mask] = -1

    stats = {
        "resolution": int(resolution),
        "vertex_blobs": int(vertex_data["vertices"].shape[0]),
        "vertex_blob_voxels": int(vertex_data["vertex_mask"].sum()),
        "vertex_core_mode": str(vertex_core_mode),
        "vertex_core_voxels": int(vertex_core_mask.sum()),
        "vertex_clearance": int(vertex_clearance),
        "vertex_clearance_voxels": int(vertex_clear_mask.sum()),
        "raw_edge_voxels": int(raw_edge_mask.sum()),
        "cleared_edge_voxels": int(cleared_edge_mask.sum()),
        "edge_components": int(len(components)),
        "largest_edge_component": int(max((c.size for c in components), default=0)),
        "vertex_dvert_threshold": float(vertex_dvert_threshold),
        "vertex_raw_max_barycentric": float(threshold_to_raw_max_barycentric(vertex_dvert_threshold)),
        "edge_dtri_threshold": float(edge_dtri_threshold),
        "edge_min_dvert": float(edge_min_dvert),
        "edge_max_dvert": float(edge_max_dvert),
        "include_vertex_context": str(bool(include_vertex_context)),
    }
    return labels, component_sizes, np.asarray([c.size for c in components], dtype=np.int32), stats


def write_stage3_ply(
    path: str | Path,
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    labels: np.ndarray,
    component_sizes: np.ndarray,
    resolution: int,
) -> None:
    selected = np.flatnonzero(labels != -999)
    centers = voxel_centers(coords[selected], resolution)
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment BRG stage 3 edge components before contact filtering\n")
        f.write("comment component_id >= 0 means cleared edge component\n")
        f.write("comment component_id -1 means vertex core, if context enabled\n")
        f.write("comment component_id -2 means vertex clearance removed, if context enabled\n")
        f.write(f"element vertex {selected.size}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int component_id\n")
        f.write("property int component_size\n")
        f.write("property float d_tri\n")
        f.write("property float d_vert\n")
        f.write("end_header\n")
        for center, idx in zip(centers, selected):
            label = int(labels[int(idx)])
            if label == -1:
                color = (255, 40, 40)
                size = 0
            elif label == -2:
                color = (255, 165, 40)
                size = 0
            else:
                color = component_color(label)
                size = int(component_sizes[int(idx)])
            f.write(
                f"{center[0]:.8f} {center[1]:.8f} {center[2]:.8f} "
                f"{color[0]} {color[1]} {color[2]} "
                f"{label} {size} "
                f"{float(d_tri[int(idx)]):.8f} {float(d_vert[int(idx)]):.8f}\n"
            )


def write_component_csv(path: str | Path, component_lengths: np.ndarray, stats: dict[str, int | float | str]) -> None:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        for key in sorted(stats):
            writer.writerow([key, stats[key]])
        writer.writerow([])
        writer.writerow(["component_id", "voxel_count"])
        for component_id, size in enumerate(component_lengths.tolist()):
            writer.writerow([component_id, int(size)])


def print_summary(stats: dict[str, int | float | str]) -> None:
    print("BRG stage 3 edge-component visualization")
    print(f"Resolution: {stats['resolution']}")
    print(f"Vertex blobs: {stats['vertex_blobs']}")
    print(f"Vertex core mode: {stats['vertex_core_mode']}")
    print(f"Vertex core voxels: {stats['vertex_core_voxels']}")
    print(f"Vertex clearance voxels: {stats['vertex_clearance_voxels']}")
    print(f"Raw edge voxels: {stats['raw_edge_voxels']}")
    print(f"Cleared edge voxels: {stats['cleared_edge_voxels']}")
    print(f"Edge components: {stats['edge_components']}")
    print(f"Largest edge component: {stats['largest_edge_component']}")
    print("Thresholds:")
    print(
        f"  vertex d_vert >= {stats['vertex_dvert_threshold']} "
        f"(raw max barycentric >= {stats['vertex_raw_max_barycentric']:.6f})"
    )
    print(f"  edge d_tri <= {stats['edge_dtri_threshold']}")
    print(f"  edge d_vert in [{stats['edge_min_dvert']}, {stats['edge_max_dvert']}]")
    print(f"  vertex clearance radius from core = {stats['vertex_clearance']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BRG edge voxels through connected components only.")
    parser.add_argument("path", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--vertex_dvert_threshold", type=float, default=0.84)
    parser.add_argument("--vertex_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--vertex_min_component_size", type=int, default=1)
    parser.add_argument("--vertex_position_mode", choices=["peak", "mean", "weighted"], default="weighted")
    parser.add_argument(
        "--vertex_core_mode",
        choices=["closest4", "peak", "position_nearest", "blob", "none"],
        default="closest4",
    )
    parser.add_argument("--edge_dtri_threshold", type=float, default=0.175)
    parser.add_argument("--edge_min_dvert", type=float, default=0.25)
    parser.add_argument("--edge_max_dvert", type=float, default=None)
    parser.add_argument("--edge_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--vertex_clearance", type=int, default=1)
    parser.add_argument("--include_vertex_context", action="store_true")
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)
    d_tri = features[:, 0]
    d_vert = features[:, 1]

    labels, component_sizes, component_lengths, stats = build_stage3_labels(
        coords,
        d_tri,
        d_vert,
        resolution,
        vertex_dvert_threshold=args.vertex_dvert_threshold,
        vertex_connectivity=args.vertex_connectivity,
        vertex_min_component_size=args.vertex_min_component_size,
        vertex_position_mode=args.vertex_position_mode,
        vertex_core_mode=args.vertex_core_mode,
        edge_dtri_threshold=args.edge_dtri_threshold,
        edge_min_dvert=args.edge_min_dvert,
        edge_max_dvert=args.edge_max_dvert,
        edge_connectivity=args.edge_connectivity,
        vertex_clearance=args.vertex_clearance,
        include_vertex_context=args.include_vertex_context,
    )

    stem = default_output_stem(path)
    out_dir = default_results_dir(path)
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_stage3_edge_components.ply"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"{stem}_stage3_edge_components.csv"
    write_stage3_ply(out_ply, coords, d_tri, d_vert, labels, component_sizes, resolution)
    write_component_csv(out_csv, component_lengths, stats)
    print_summary(stats)
    print(f"Saved {out_ply}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
