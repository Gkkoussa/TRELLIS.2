"""
Rasterize a zoomed-in BRG voxel neighborhood around three vertex blobs.

The script picks one triangle from the reconstructed BRG mesh, crops all
classified BRG voxels around its three vertices, and saves a colored PNG.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from extract_barycentric_ridge_graph import (
    default_output_stem,
    default_results_dir,
    extract_vertex_blobs,
    infer_resolution,
    load_triangle_field,
    voxel_centers,
)
from visualize_brg_voxels import LABELS, classify_brg_voxels


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def choose_face(vertices: np.ndarray, faces: np.ndarray, mode: str) -> int:
    areas = face_areas(vertices, faces)
    valid = np.flatnonzero(areas > 1e-12)
    if valid.size == 0:
        raise ValueError("No non-degenerate faces found in mesh")
    if mode == "largest":
        return int(valid[np.argmax(areas[valid])])
    if mode == "median":
        target = np.quantile(areas[valid], 0.5)
    elif mode == "p75":
        target = np.quantile(areas[valid], 0.75)
    else:
        raise ValueError(f"Unsupported face selection mode: {mode}")
    return int(valid[np.argmin(np.abs(areas[valid] - target))])


def crop_voxels(
    centers: np.ndarray,
    labels: np.ndarray,
    face_vertices: np.ndarray,
    margin: float,
) -> np.ndarray:
    lo = face_vertices.min(axis=0) - margin
    hi = face_vertices.max(axis=0) + margin
    return np.flatnonzero(
        (labels > 0)
        & np.all(centers >= lo[None, :], axis=1)
        & np.all(centers <= hi[None, :], axis=1)
    )


def colors_for_labels(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for label_id, (_, rgb) in LABELS.items():
        mask = labels == label_id
        colors[mask] = np.asarray(rgb, dtype=np.float32) / 255.0
    return colors


def equalize_3d_axes(ax, points: np.ndarray, pad: float = 0.0) -> None:
    lo = points.min(axis=0) - pad
    hi = points.max(axis=0) + pad
    center = (lo + hi) * 0.5
    radius = float((hi - lo).max() * 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_3d_panel(ax, points, colors, face_vertices, title, elev=25, azim=45):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=8, linewidths=0, alpha=0.9)
    ax.scatter(
        face_vertices[:, 0],
        face_vertices[:, 1],
        face_vertices[:, 2],
        c="white",
        s=95,
        edgecolors="black",
        linewidths=1.2,
        depthshade=False,
    )
    closed = np.vstack([face_vertices, face_vertices[0:1]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="black", linewidth=1.4)
    equalize_3d_axes(ax, np.vstack([points, face_vertices]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_2d_panel(ax, points, colors, face_vertices, axes, title):
    a, b = axes
    ax.scatter(points[:, a], points[:, b], c=colors, s=8, linewidths=0, alpha=0.9)
    closed = np.vstack([face_vertices, face_vertices[0:1]])
    ax.plot(closed[:, a], closed[:, b], color="black", linewidth=1.2)
    ax.scatter(
        face_vertices[:, a],
        face_vertices[:, b],
        c="white",
        s=55,
        edgecolors="black",
        linewidths=1.0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("xyz"[a])
    ax.set_ylabel("xyz"[b])


def add_legend(fig) -> None:
    handles = []
    for _, (name, rgb) in LABELS.items():
        color = np.asarray(rgb, dtype=np.float32) / 255.0
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                label=name,
                markersize=7,
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize a zoomed BRG voxel neighborhood around 3 vertex blobs.")
    parser.add_argument("triangle_field", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--mesh_npz", type=str, default=None, help="BRG triangle mesh .npz")
    parser.add_argument("--face_index", type=int, default=None)
    parser.add_argument("--face_mode", choices=["largest", "median", "p75"], default="p75")
    parser.add_argument("--margin_voxels", type=float, default=14.0)
    parser.add_argument("--out_png", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=220)

    parser.add_argument("--vertex_dvert_threshold", type=float, default=0.84)
    parser.add_argument("--vertex_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument(
        "--vertex_core_mode",
        choices=["closest4", "peak", "position_nearest", "blob", "none"],
        default="closest4",
    )
    parser.add_argument("--edge_dtri_threshold", type=float, default=0.175)
    parser.add_argument("--edge_min_dvert", type=float, default=0.25)
    parser.add_argument("--edge_max_dvert", type=float, default=None)
    parser.add_argument("--edge_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--edge_attach_mode", choices=["evidence", "endpoints", "contacts"], default="evidence")
    parser.add_argument("--vertex_clearance", type=int, default=1)
    parser.add_argument("--attach_radius", type=int, default=3)
    parser.add_argument("--ridge_midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--ridge_midpoint_tolerance", type=float, default=0.075)
    parser.add_argument("--midpoint_dtri_threshold", type=float, default=0.25)
    parser.add_argument("--midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--midpoint_tolerance", type=float, default=0.05)
    args = parser.parse_args()

    field_path = Path(args.triangle_field)
    coords, features = load_triangle_field(field_path)
    resolution = infer_resolution(coords)

    out_dir = default_results_dir(field_path)
    stem = default_output_stem(field_path)
    mesh_npz = Path(args.mesh_npz) if args.mesh_npz else out_dir / f"{stem}_brg_triangle_mesh.npz"
    mesh = np.load(mesh_npz, allow_pickle=False)
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    face_index = args.face_index if args.face_index is not None else choose_face(vertices, faces, args.face_mode)
    face = faces[face_index]
    face_vertices = vertices[face]

    labels, stats = classify_brg_voxels(
        coords,
        features[:, 0],
        features[:, 1],
        resolution,
        vertex_dvert_threshold=args.vertex_dvert_threshold,
        vertex_connectivity=args.vertex_connectivity,
        vertex_core_mode=args.vertex_core_mode,
        edge_dtri_threshold=args.edge_dtri_threshold,
        edge_min_dvert=args.edge_min_dvert,
        edge_max_dvert=args.edge_max_dvert,
        edge_connectivity=args.edge_connectivity,
        edge_attach_mode=args.edge_attach_mode,
        vertex_clearance=args.vertex_clearance,
        attach_radius=args.attach_radius,
        ridge_midpoint_dvert=args.ridge_midpoint_dvert,
        ridge_midpoint_tolerance=args.ridge_midpoint_tolerance,
        midpoint_dtri_threshold=args.midpoint_dtri_threshold,
        midpoint_dvert=args.midpoint_dvert,
        midpoint_tolerance=args.midpoint_tolerance,
    )

    centers = voxel_centers(coords, resolution)
    margin = float(args.margin_voxels) / float(resolution)
    selected = crop_voxels(centers, labels, face_vertices, margin)
    if selected.size == 0:
        raise RuntimeError("Crop selected zero voxels; increase --margin_voxels")

    points = centers[selected]
    point_labels = labels[selected]
    colors = colors_for_labels(point_labels)

    fig = plt.figure(figsize=(13.5, 11.0), constrained_layout=False)
    fig.suptitle(
        f"BRG zoom around 3 vertex blobs | face {face_index} | "
        f"vertices {face.tolist()} | voxels {selected.size}",
        fontsize=12,
    )
    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_3d_panel(ax0, points, colors, face_vertices, "Perspective", elev=24, azim=38)
    ax1 = fig.add_subplot(2, 2, 2)
    plot_2d_panel(ax1, points, colors, face_vertices, (0, 1), "XY projection")
    ax2 = fig.add_subplot(2, 2, 3)
    plot_2d_panel(ax2, points, colors, face_vertices, (0, 2), "XZ projection")
    ax3 = fig.add_subplot(2, 2, 4)
    plot_2d_panel(ax3, points, colors, face_vertices, (1, 2), "YZ projection")
    add_legend(fig)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.12, wspace=0.18, hspace=0.25)

    out_png = Path(args.out_png) if args.out_png else out_dir / f"brg_zoom_face_{face_index}.png"
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Face index: {face_index}")
    print(f"Vertex ids: {face.tolist()}")
    print(f"Crop margin: {args.margin_voxels} voxels ({margin:.6f} world units)")
    print(f"Selected voxels: {selected.size}")
    for label_id, (name, _) in LABELS.items():
        print(f"{name}: {int((point_labels == label_id).sum())}")
    print(f"Full visualization stats accepted_edge_components={stats['accepted_edge_components']}")


if __name__ == "__main__":
    main()
