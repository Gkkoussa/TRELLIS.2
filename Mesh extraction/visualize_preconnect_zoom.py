"""
Zoomed pre-connect visualization for BRG vertex and edge voxels.

This stops before edge components are bridged, attached to vertices, or used to
make faces. It crops a local region and writes:
  - a colored PLY point cloud
  - a PNG with 3D/2D projections
  - a CSV with visible component counts
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
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
from visualize_stage3_edge_components import component_color


IGNORE_LABEL = -999
VERTEX_CORE_LABEL = -1
VERTEX_BLOB_LABEL = -2


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected comma-separated integer ids")
    return [int(item) for item in items]


def parse_float3(value: str) -> np.ndarray:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("Expected exactly 3 comma-separated values")
    return np.asarray([float(v) for v in parts], dtype=np.float32)


def parse_int3(value: str) -> np.ndarray:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("Expected exactly 3 comma-separated integer values")
    return np.asarray([int(v) for v in parts], dtype=np.int32)


def build_preconnect_labels(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    *,
    vertex_dvert_threshold: float,
    vertex_connectivity: int,
    vertex_min_component_size: int,
    vertex_position_mode: str,
    vertex_core_mode: str,
    edge_dtri_threshold: float,
    edge_min_dvert: float,
    edge_max_dvert: float | None,
    edge_connectivity: int,
    edge_min_component_size: int,
    vertex_clearance: int,
    include_vertex_blob: bool,
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
    edge_components = connected_components(coords, cleared_edge_mask, connectivity=edge_connectivity)

    labels = np.full(coords.shape[0], IGNORE_LABEL, dtype=np.int32)
    component_sizes = np.zeros(coords.shape[0], dtype=np.int32)
    kept_components = 0
    for component_id, component in enumerate(edge_components):
        if component.size < edge_min_component_size:
            continue
        labels[component] = int(component_id)
        component_sizes[component] = int(component.size)
        kept_components += 1

    if include_vertex_blob:
        blob_mask = vertex_data["vertex_mask"] & (labels == IGNORE_LABEL)
        labels[blob_mask] = VERTEX_BLOB_LABEL
    labels[vertex_core_mask] = VERTEX_CORE_LABEL

    stats = {
        "resolution": int(resolution),
        "vertex_blobs": int(vertex_data["vertices"].shape[0]),
        "vertex_blob_voxels": int(vertex_data["vertex_mask"].sum()),
        "vertex_core_voxels": int(vertex_core_mask.sum()),
        "vertex_core_mode": str(vertex_core_mode),
        "vertex_clearance": int(vertex_clearance),
        "vertex_clearance_voxels": int(vertex_clear_mask.sum()),
        "raw_edge_voxels": int(raw_edge_mask.sum()),
        "cleared_edge_voxels": int(cleared_edge_mask.sum()),
        "edge_components": int(len(edge_components)),
        "kept_edge_components": int(kept_components),
        "edge_min_component_size": int(edge_min_component_size),
        "vertex_dvert_threshold": float(vertex_dvert_threshold),
        "vertex_raw_max_barycentric": float(threshold_to_raw_max_barycentric(vertex_dvert_threshold)),
        "edge_dtri_threshold": float(edge_dtri_threshold),
        "edge_min_dvert": float(edge_min_dvert),
        "edge_max_dvert": float(edge_max_dvert),
        "include_vertex_blob": str(bool(include_vertex_blob)),
    }
    return labels, component_sizes, vertex_data["vertices"], stats


def crop_indices(
    centers: np.ndarray,
    labels: np.ndarray,
    reference_points: np.ndarray,
    radius_world: float,
    margin_world: float,
) -> np.ndarray:
    if reference_points.shape[0] == 1:
        lo = reference_points[0] - radius_world
        hi = reference_points[0] + radius_world
    else:
        lo = reference_points.min(axis=0) - margin_world
        hi = reference_points.max(axis=0) + margin_world
    return np.flatnonzero(
        (labels != IGNORE_LABEL)
        & np.all(centers >= lo[None, :], axis=1)
        & np.all(centers <= hi[None, :], axis=1)
    )


def color_for_label(label: int) -> tuple[int, int, int]:
    if label == VERTEX_CORE_LABEL:
        return 255, 35, 35
    if label == VERTEX_BLOB_LABEL:
        return 255, 135, 135
    return component_color(label)


def colors_for_labels(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for i, label in enumerate(labels.tolist()):
        colors[i] = np.asarray(color_for_label(int(label)), dtype=np.float32) / 255.0
    return colors


def write_zoom_ply(
    path: str | Path,
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    labels: np.ndarray,
    component_sizes: np.ndarray,
    selected: np.ndarray,
    resolution: int,
) -> None:
    centers = voxel_centers(coords[selected], resolution)
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment BRG pre-connect zoom: vertex cores/blobs and edge components only\n")
        f.write("comment component_id >= 0 means edge component before final connection\n")
        f.write("comment component_id -1 means collapsed vertex core\n")
        f.write("comment component_id -2 means high-d_vert vertex blob, if enabled\n")
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
            color = color_for_label(label)
            size = int(component_sizes[int(idx)]) if label >= 0 else 0
            f.write(
                f"{center[0]:.8f} {center[1]:.8f} {center[2]:.8f} "
                f"{color[0]} {color[1]} {color[2]} "
                f"{label} {size} "
                f"{float(d_tri[int(idx)]):.8f} {float(d_vert[int(idx)]):.8f}\n"
            )


def equalize_3d_axes(ax, points: np.ndarray) -> None:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(float((hi - lo).max() * 0.5), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_reference_3d(ax, reference_points: np.ndarray) -> None:
    ax.scatter(
        reference_points[:, 0],
        reference_points[:, 1],
        reference_points[:, 2],
        c="white",
        s=85,
        edgecolors="black",
        linewidths=1.0,
        depthshade=False,
    )
    if reference_points.shape[0] == 3:
        closed = np.vstack([reference_points, reference_points[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="black", linewidth=1.2)


def draw_reference_2d(ax, reference_points: np.ndarray, axes: tuple[int, int]) -> None:
    a, b = axes
    ax.scatter(
        reference_points[:, a],
        reference_points[:, b],
        c="white",
        s=45,
        edgecolors="black",
        linewidths=0.9,
    )
    if reference_points.shape[0] == 3:
        closed = np.vstack([reference_points, reference_points[0:1]])
        ax.plot(closed[:, a], closed[:, b], color="black", linewidth=1.0)


def plot_zoom_png(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
    reference_points: np.ndarray,
    title: str,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(13.5, 11.0), constrained_layout=False)
    fig.suptitle(title, fontsize=12)

    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    ax0.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=8, linewidths=0, alpha=0.92)
    draw_reference_3d(ax0, reference_points)
    equalize_3d_axes(ax0, np.vstack([points, reference_points]))
    ax0.view_init(elev=24, azim=38)
    ax0.set_title("Perspective", fontsize=10)

    for subplot, axes, name in [
        (222, (0, 1), "XY projection"),
        (223, (0, 2), "XZ projection"),
        (224, (1, 2), "YZ projection"),
    ]:
        ax = fig.add_subplot(subplot)
        a, b = axes
        ax.scatter(points[:, a], points[:, b], c=colors, s=8, linewidths=0, alpha=0.92)
        draw_reference_2d(ax, reference_points, axes)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("xyz"[a])
        ax.set_ylabel("xyz"[b])

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=(1.0, 0.14, 0.14), label="vertex_core"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=(1.0, 0.53, 0.53), label="vertex_blob"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=(0.0, 0.75, 1.0), label="edge_components"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.12, wspace=0.18, hspace=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_zoom_csv(
    path: str | Path,
    stats: dict[str, int | float | str],
    labels: np.ndarray,
    selected: np.ndarray,
    component_sizes: np.ndarray,
) -> None:
    selected_labels = labels[selected]
    visible_components = sorted(int(v) for v in np.unique(selected_labels) if int(v) >= 0)
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        for key in sorted(stats):
            writer.writerow([key, stats[key]])
        writer.writerow(["zoom_selected_voxels", int(selected.size)])
        writer.writerow(["zoom_vertex_core_voxels", int((selected_labels == VERTEX_CORE_LABEL).sum())])
        writer.writerow(["zoom_vertex_blob_voxels", int((selected_labels == VERTEX_BLOB_LABEL).sum())])
        writer.writerow(["zoom_visible_edge_components", len(visible_components)])
        writer.writerow([])
        writer.writerow(["visible_edge_component_id", "full_component_size", "visible_voxels"])
        for component_id in visible_components:
            visible = int((selected_labels == component_id).sum())
            component_indices = selected[selected_labels == component_id]
            full_size = int(component_sizes[int(component_indices[0])]) if component_indices.size else 0
            writer.writerow([component_id, full_size, visible])


def main() -> None:
    parser = argparse.ArgumentParser(description="Zoom BRG vertex/edge voxels before final graph connection.")
    parser.add_argument("path", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--vertex_ids", type=str, default=None, help="Comma-separated extracted vertex ids, e.g. 10,42,77")
    parser.add_argument("--center_world", type=str, default=None, help="World crop center x,y,z")
    parser.add_argument("--center_voxel", type=str, default=None, help="Voxel crop center i,j,k")
    parser.add_argument("--radius_voxels", type=float, default=24.0)
    parser.add_argument("--margin_voxels", type=float, default=12.0)
    parser.add_argument("--include_vertex_blob", action="store_true")
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--out_png", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=220)

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
    parser.add_argument("--vertex_clearance", type=int, default=1)
    args = parser.parse_args()

    selectors = [args.vertex_ids is not None, args.center_world is not None, args.center_voxel is not None]
    if sum(selectors) != 1:
        raise ValueError("Set exactly one of --vertex_ids, --center_world, or --center_voxel")

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)
    d_tri = features[:, 0]
    d_vert = features[:, 1]

    labels, component_sizes, vertices, stats = build_preconnect_labels(
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
        vertex_clearance=args.vertex_clearance,
        include_vertex_blob=args.include_vertex_blob,
    )

    if args.vertex_ids is not None:
        vertex_ids = parse_int_list(args.vertex_ids)
        bad_ids = [v for v in vertex_ids if v < 0 or v >= vertices.shape[0]]
        if bad_ids:
            raise ValueError(f"Vertex id(s) out of range: {bad_ids}; valid range is 0..{vertices.shape[0] - 1}")
        reference_points = vertices[np.asarray(vertex_ids, dtype=np.int32)]
        target_name = "vertices_" + "_".join(str(v) for v in vertex_ids)
    elif args.center_world is not None:
        reference_points = parse_float3(args.center_world).reshape(1, 3)
        target_name = "world_" + "_".join(f"{v:.4f}" for v in reference_points[0])
    else:
        center_voxel = parse_int3(args.center_voxel).reshape(1, 3)
        reference_points = voxel_centers(center_voxel, resolution)
        target_name = "voxel_" + "_".join(str(int(v)) for v in center_voxel[0])

    centers = voxel_centers(coords, resolution)
    selected = crop_indices(
        centers,
        labels,
        reference_points,
        radius_world=float(args.radius_voxels) / float(resolution),
        margin_world=float(args.margin_voxels) / float(resolution),
    )
    if selected.size == 0:
        raise RuntimeError("Crop selected zero voxels; increase radius/margin or choose another target")

    points = centers[selected]
    selected_labels = labels[selected]
    colors = colors_for_labels(selected_labels)

    out_dir = default_results_dir(path)
    stem = default_output_stem(path)
    default_prefix = f"{stem}_preconnect_zoom_{target_name}"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{default_prefix}.ply"
    out_png = Path(args.out_png) if args.out_png else out_dir / f"{default_prefix}.png"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"{default_prefix}.csv"

    write_zoom_ply(out_ply, coords, d_tri, d_vert, labels, component_sizes, selected, resolution)
    plot_zoom_png(
        out_png,
        points,
        colors,
        reference_points,
        title=f"Pre-connect BRG zoom | {target_name} | voxels {selected.size}",
        dpi=args.dpi,
    )
    write_zoom_csv(out_csv, stats, labels, selected, component_sizes)

    print(f"Saved {out_ply}")
    print(f"Saved {out_png}")
    print(f"Saved {out_csv}")
    print(f"Selected voxels: {selected.size}")
    print(f"Visible edge components: {len([v for v in np.unique(selected_labels) if int(v) >= 0])}")
    print(f"Vertex core voxels: {int((selected_labels == VERTEX_CORE_LABEL).sum())}")
    print(f"Vertex blob voxels: {int((selected_labels == VERTEX_BLOB_LABEL).sum())}")


if __name__ == "__main__":
    main()
