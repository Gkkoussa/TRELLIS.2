"""
Colored voxel visualization for the Barycentric Ridge Graph pipeline.

Writes colored PLY point clouds of the exact voxel sets used by BRG:
  - collapsed vertex voxels from high-d_vert blobs
  - edge-looking voxels after vertex clearance

Only coords, d_tri, and d_vert are used.
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


LABELS = {
    1: ("vertex_blob", (255, 40, 40)),
    2: ("edge_blob", (0, 190, 255)),
}


def classify_brg_voxels(
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
    edge_min_component_size: int = 2,
    edge_attach_mode: str = "evidence",
    vertex_clearance: int = 1,
    attach_radius: int = 3,
    ridge_midpoint_dvert: float = 0.25,
    ridge_midpoint_tolerance: float = 0.075,
    require_ridge_midpoint: bool = True,
    midpoint_dtri_threshold: float = 0.25,
    midpoint_dvert: float = 0.25,
    midpoint_tolerance: float = 0.05,
) -> tuple[np.ndarray, dict[str, int | float]]:
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
    vertex_mask = vertex_data["vertex_mask"]
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
    edge_components = connected_components(coords, cleared_edge_mask, connectivity=edge_connectivity)

    labels = np.zeros(coords.shape[0], dtype=np.uint8)
    kept_edge_components = 0

    for component in edge_components:
        if component.size < edge_min_component_size:
            continue
        labels[component] = 2
        kept_edge_components += 1
    labels[vertex_core_mask] = 1

    stats = {
        "resolution": int(resolution),
        "vertex_blobs": int(vertex_data["vertices"].shape[0]),
        "vertex_blob_voxels": int(vertex_mask.sum()),
        "vertex_core_voxels": int(vertex_core_mask.sum()),
        "vertex_clearance_voxels": int(vertex_clear_mask.sum()),
        "raw_edge_voxels": int(raw_edge_mask.sum()),
        "cleared_edge_voxels": int(cleared_edge_mask.sum()),
        "edge_components": int(len(edge_components)),
        "visualized_edge_components": int(kept_edge_components),
        "visualized_voxels": int((labels > 0).sum()),
        "vertex_dvert_threshold": float(vertex_dvert_threshold),
        "vertex_raw_max_barycentric": float(threshold_to_raw_max_barycentric(vertex_dvert_threshold)),
        "vertex_core_mode": str(vertex_core_mode),
        "edge_dtri_threshold": float(edge_dtri_threshold),
        "edge_min_dvert": float(edge_min_dvert),
        "edge_max_dvert": float(edge_max_dvert),
        "edge_attach_mode": str(edge_attach_mode),
        "vertex_clearance": int(vertex_clearance),
        "attach_radius": int(attach_radius),
        "ridge_midpoint_dvert": float(ridge_midpoint_dvert),
        "ridge_midpoint_tolerance": float(ridge_midpoint_tolerance),
        "midpoint_dtri_threshold": float(midpoint_dtri_threshold),
        "midpoint_dvert": float(midpoint_dvert),
        "midpoint_tolerance": float(midpoint_tolerance),
    }
    for label_id, (name, _) in LABELS.items():
        stats[f"label_{label_id}_{name}_voxels"] = int((labels == label_id).sum())
    return labels, stats


def write_colored_ply(
    path: str | Path,
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    labels: np.ndarray,
    resolution: int,
) -> None:
    selected = np.flatnonzero(labels > 0)
    centers = voxel_centers(coords[selected], resolution)
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment BRG colored voxel classification\n")
        f.write(f"element vertex {selected.size}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("property float d_tri\n")
        f.write("property float d_vert\n")
        f.write("end_header\n")
        for center, idx in zip(centers, selected):
            label = int(labels[int(idx)])
            _, color = LABELS[label]
            f.write(
                f"{center[0]:.8f} {center[1]:.8f} {center[2]:.8f} "
                f"{color[0]} {color[1]} {color[2]} {label} "
                f"{float(d_tri[int(idx)]):.8f} {float(d_vert[int(idx)]):.8f}\n"
            )


def write_binary_mask_ply(
    path: str | Path,
    coords: np.ndarray,
    mask: np.ndarray,
    resolution: int,
    color: tuple[int, int, int],
) -> None:
    labels = np.zeros(coords.shape[0], dtype=np.uint8)
    labels[mask] = 1
    selected = np.flatnonzero(labels > 0)
    centers = voxel_centers(coords[selected], resolution)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {selected.size}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for center in centers:
            f.write(
                f"{center[0]:.8f} {center[1]:.8f} {center[2]:.8f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def write_legend(path: str | Path, stats: dict[str, int | float]) -> None:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "name", "red", "green", "blue"])
        for label_id, (name, color) in LABELS.items():
            writer.writerow([label_id, name, color[0], color[1], color[2]])
        writer.writerow([])
        writer.writerow(["stat", "value"])
        for key in sorted(stats):
            writer.writerow([key, stats[key]])


def print_summary(stats: dict[str, int | float]) -> None:
    print("BRG voxel visualization")
    print(f"Resolution: {stats['resolution']}")
    print(f"Vertex blobs: {stats['vertex_blobs']}")
    print(f"Vertex blob voxels: {stats['vertex_blob_voxels']}")
    print(f"Vertex core mode: {stats['vertex_core_mode']}")
    print(f"Vertex core voxels: {stats['vertex_core_voxels']}")
    print(f"Vertex clearance voxels: {stats['vertex_clearance_voxels']}")
    print(f"Raw edge voxels: {stats['raw_edge_voxels']}")
    print(f"Cleared edge voxels: {stats['cleared_edge_voxels']}")
    print(f"Edge components: {stats['edge_components']}")
    print(f"Visualized edge components: {stats['visualized_edge_components']}")
    print(f"Visualized voxels: {stats['visualized_voxels']}")
    print("Colors:")
    for label_id, (name, color) in LABELS.items():
        count_key = f"label_{label_id}_{name}_voxels"
        print(f"  {label_id}: {name:30s} rgb={color} voxels={stats[count_key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BRG vertex and edge voxels as colored PLY.")
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
    parser.add_argument("--edge_min_component_size", type=int, default=2)
    parser.add_argument("--edge_attach_mode", choices=["evidence", "endpoints", "contacts"], default="evidence")
    parser.add_argument("--vertex_clearance", type=int, default=1)
    parser.add_argument("--attach_radius", type=int, default=3)
    parser.add_argument("--ridge_midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--ridge_midpoint_tolerance", type=float, default=0.075)
    parser.add_argument("--no_ridge_midpoint_check", action="store_true")
    parser.add_argument("--midpoint_dtri_threshold", type=float, default=0.25)
    parser.add_argument("--midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--midpoint_tolerance", type=float, default=0.05)
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--legend_csv", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)
    d_tri = features[:, 0]
    d_vert = features[:, 1]

    labels, stats = classify_brg_voxels(
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
        edge_min_component_size=args.edge_min_component_size,
        edge_attach_mode=args.edge_attach_mode,
        vertex_clearance=args.vertex_clearance,
        attach_radius=args.attach_radius,
        ridge_midpoint_dvert=args.ridge_midpoint_dvert,
        ridge_midpoint_tolerance=args.ridge_midpoint_tolerance,
        require_ridge_midpoint=not args.no_ridge_midpoint_check,
        midpoint_dtri_threshold=args.midpoint_dtri_threshold,
        midpoint_dvert=args.midpoint_dvert,
        midpoint_tolerance=args.midpoint_tolerance,
    )

    stem = default_output_stem(path)
    out_dir = default_results_dir(path)
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_brg_voxels_colored.ply"
    legend_csv = Path(args.legend_csv) if args.legend_csv else out_dir / f"{stem}_brg_voxels_legend.csv"
    write_colored_ply(out_ply, coords, d_tri, d_vert, labels, resolution)
    write_legend(legend_csv, stats)
    print_summary(stats)
    print(f"Saved {out_ply}")
    print(f"Saved {legend_csv}")


if __name__ == "__main__":
    main()
