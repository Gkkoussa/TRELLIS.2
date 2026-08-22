"""
Label Skeleton Mesh (LSM).

This experimental mesh builder follows the label-driven pipeline:

  Red label-1 blobs
      -> connected components
      -> recovered mesh vertices

  Red + orange voxel mask
      -> binary closing
      -> 3-D skeletonization
      -> graph edges

  Cyan label-3 regions
      -> planar connected components
      -> associate three graph vertices
      -> triangular faces

  Topology cleanup
      -> orient
      -> remove non-manifold/duplicate faces
      -> export OBJ/PLY/NPZ

The labels are computed from coords, d_tri, and d_vert by visualize_brg_voxels.py.
No offset channels or source mesh faces are used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize_3d

from build_brg_triangle_mesh import write_obj, write_ply_mesh
from extract_barycentric_ridge_graph import (
    connected_components,
    default_output_stem,
    default_results_dir,
    extract_vertex_blobs,
    infer_resolution,
    load_triangle_field,
    threshold_to_raw_max_barycentric,
    voxel_centers,
)
from visualize_brg_voxels import classify_brg_voxels


ALGO_NAME = "Label Skeleton Mesh"
ALGO_SHORT_NAME = "LSM"


def make_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    if connectivity not in (6, 18, 26):
        raise ValueError(f"connectivity must be 6, 18, or 26, got {connectivity}")
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def edge_tuple(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    if a == b:
        raise ValueError("self-edge")
    return (a, b) if a < b else (b, a)


def face_tuple(face: tuple[int, int, int] | list[int] | np.ndarray) -> tuple[int, int, int]:
    face = tuple(int(v) for v in face)
    if len(set(face)) != 3:
        raise ValueError("degenerate face")
    return tuple(sorted(face))


def dense_mask_from_sparse(
    coords: np.ndarray,
    mask: np.ndarray,
    padding: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    selected = coords[np.flatnonzero(mask)]
    if selected.size == 0:
        raise ValueError("Cannot densify an empty sparse mask")
    lo = np.maximum(selected.min(axis=0) - int(padding), 0)
    hi = selected.max(axis=0) + int(padding) + 1
    shape = tuple((hi - lo).tolist())
    dense = np.zeros(shape, dtype=bool)
    local = selected - lo[None, :]
    dense[local[:, 0], local[:, 1], local[:, 2]] = True
    return dense, lo.astype(np.int32)


def sparse_coords_from_dense(dense: np.ndarray, origin: np.ndarray) -> np.ndarray:
    local = np.argwhere(dense)
    if local.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return (local + origin[None, :]).astype(np.int32, copy=False)


def close_and_skeletonize(
    coords: np.ndarray,
    mask: np.ndarray,
    *,
    closing_radius: int = 1,
    closing_iterations: int = 1,
    dense_padding: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    dense, origin = dense_mask_from_sparse(coords, mask, padding=dense_padding + closing_radius)
    if closing_radius > 0 and closing_iterations > 0:
        structure = np.ones((2 * closing_radius + 1,) * 3, dtype=bool)
        closed = ndimage.binary_closing(dense, structure=structure, iterations=closing_iterations)
    else:
        closed = dense
    skeleton = skeletonize_3d(closed) > 0
    stats = {
        "dense_voxels_before_closing": int(dense.sum()),
        "dense_voxels_after_closing": int(closed.sum()),
        "skeleton_voxels": int(skeleton.sum()),
        "dense_crop_x": int(dense.shape[0]),
        "dense_crop_y": int(dense.shape[1]),
        "dense_crop_z": int(dense.shape[2]),
    }
    return sparse_coords_from_dense(skeleton, origin), stats


def trace_skeleton_edges(
    skeleton_coords: np.ndarray,
    vertex_positions: np.ndarray,
    resolution: int,
    *,
    skeleton_connectivity: int = 26,
    endpoint_attach_voxels: float = 6.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Convert a 1-voxel skeleton into graph edges between recovered vertices.

    Branch/end voxels and voxels near recovered vertices are key nodes. Chains
    between key nodes are traced, and chain endpoints are attached to nearest
    recovered vertices.
    """
    if skeleton_coords.shape[0] == 0 or vertex_positions.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32), {
            "skeleton_branches": 0,
            "skeleton_edges": 0,
            "skeleton_branch_rejected_unattached": 0,
            "skeleton_branch_rejected_self": 0,
        }

    offsets = make_offsets(skeleton_connectivity)
    coord_to_local = {tuple(c.tolist()): int(i) for i, c in enumerate(skeleton_coords)}
    neighbors: list[list[int]] = [[] for _ in range(skeleton_coords.shape[0])]
    for i, coord in enumerate(skeleton_coords):
        x, y, z = coord.tolist()
        for dx, dy, dz in offsets:
            j = coord_to_local.get((x + dx, y + dy, z + dz))
            if j is not None:
                neighbors[i].append(j)

    degrees = np.asarray([len(n) for n in neighbors], dtype=np.int32)
    centers = voxel_centers(skeleton_coords, resolution)
    tree = cKDTree(vertex_positions)
    attach_radius_world = float(endpoint_attach_voxels) / float(resolution)
    nearest_dist, nearest_vertex = tree.query(centers, k=1)
    near_vertex = nearest_dist <= attach_radius_world
    is_key = (degrees != 2) | near_vertex
    key_indices = np.flatnonzero(is_key)

    if key_indices.size == 0:
        return np.zeros((0, 2), dtype=np.int32), {
            "skeleton_branches": 0,
            "skeleton_edges": 0,
            "skeleton_branch_rejected_unattached": 0,
            "skeleton_branch_rejected_self": 0,
        }

    visited_directed: set[tuple[int, int]] = set()
    edges: set[tuple[int, int]] = set()
    branches = 0
    rejected_unattached = 0
    rejected_self = 0

    for start in key_indices:
        start = int(start)
        for nxt in neighbors[start]:
            directed = (start, int(nxt))
            if directed in visited_directed:
                continue
            path = [start, int(nxt)]
            prev = start
            cur = int(nxt)
            visited_directed.add((start, cur))

            while not is_key[cur]:
                choices = [n for n in neighbors[cur] if n != prev]
                if not choices:
                    break
                nxt2 = int(choices[0])
                visited_directed.add((cur, nxt2))
                prev, cur = cur, nxt2
                path.append(cur)
                if len(path) > skeleton_coords.shape[0]:
                    break

            end = cur
            visited_directed.add((end, path[-2] if len(path) > 1 else start))
            branches += 1

            if not near_vertex[start] or not near_vertex[end]:
                rejected_unattached += 1
                continue
            a = int(nearest_vertex[start])
            b = int(nearest_vertex[end])
            if a == b:
                rejected_self += 1
                continue
            edges.add(edge_tuple(a, b))

    return np.asarray(sorted(edges), dtype=np.int32), {
        "skeleton_branches": int(branches),
        "skeleton_edges": int(len(edges)),
        "skeleton_branch_rejected_unattached": int(rejected_unattached),
        "skeleton_branch_rejected_self": int(rejected_self),
    }


def adjacency_from_edges(num_vertices: int, edges: np.ndarray) -> list[set[int]]:
    adjacency = [set() for _ in range(num_vertices)]
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def graph_cycle_faces(num_vertices: int, edges: np.ndarray) -> np.ndarray:
    adjacency = adjacency_from_edges(num_vertices, edges)
    faces = []
    for i, neighbors_i in enumerate(adjacency):
        for j in neighbors_i:
            if j <= i:
                continue
            for k in neighbors_i.intersection(adjacency[j]):
                if k > j:
                    faces.append((i, int(j), int(k)))
    if not faces:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(faces, dtype=np.int32)


def component_is_planar(points: np.ndarray, max_thickness_ratio: float, min_spread_ratio: float) -> bool:
    if points.shape[0] < 3:
        return False
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.sort(np.maximum(eig, 0.0))
    total = float(eig.sum())
    if total <= 1e-16:
        return False
    thickness_ratio = float(eig[0] / total)
    spread_ratio = float(eig[1] / max(eig[2], 1e-16))
    return thickness_ratio <= max_thickness_ratio and spread_ratio >= min_spread_ratio


def cyan_planar_faces(
    coords: np.ndarray,
    labels: np.ndarray,
    vertex_positions: np.ndarray,
    graph_edges: np.ndarray,
    resolution: int,
    *,
    cyan_connectivity: int = 18,
    cyan_min_component_size: int = 8,
    max_planar_thickness_ratio: float = 0.025,
    min_planar_spread_ratio: float = 0.05,
    nearest_vertices: int = 8,
    require_graph_triangle: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    cyan_mask = labels == 3
    cyan_components = connected_components(coords, cyan_mask, connectivity=cyan_connectivity)
    vertex_tree = cKDTree(vertex_positions)
    edge_set = {edge_tuple(int(e[0]), int(e[1])) for e in graph_edges}

    faces: set[tuple[int, int, int]] = set()
    rejected_small = 0
    rejected_nonplanar = 0
    rejected_no_graph_triangle = 0

    for component in cyan_components:
        if component.size < cyan_min_component_size:
            rejected_small += 1
            continue
        points = voxel_centers(coords[component], resolution)
        if not component_is_planar(points, max_planar_thickness_ratio, min_planar_spread_ratio):
            rejected_nonplanar += 1
            continue

        center = points.mean(axis=0)
        k = min(nearest_vertices, vertex_positions.shape[0])
        _, candidate_ids = vertex_tree.query(center, k=k)
        candidate_ids = np.atleast_1d(candidate_ids).astype(np.int32)

        best_face = None
        best_area = -1.0
        for a_i in range(candidate_ids.size):
            for b_i in range(a_i + 1, candidate_ids.size):
                for c_i in range(b_i + 1, candidate_ids.size):
                    face = [int(candidate_ids[a_i]), int(candidate_ids[b_i]), int(candidate_ids[c_i])]
                    if require_graph_triangle:
                        e01 = edge_tuple(face[0], face[1])
                        e12 = edge_tuple(face[1], face[2])
                        e20 = edge_tuple(face[2], face[0])
                        if e01 not in edge_set or e12 not in edge_set or e20 not in edge_set:
                            continue
                    p0, p1, p2 = vertex_positions[face]
                    area = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)) * 0.5)
                    if area > best_area:
                        best_area = area
                        best_face = face

        if best_face is None or best_area <= 1e-12:
            rejected_no_graph_triangle += 1
            continue
        faces.add(face_tuple(best_face))

    return (
        np.asarray(sorted(faces), dtype=np.int32).reshape(-1, 3)
        if faces
        else np.zeros((0, 3), dtype=np.int32),
        {
            "cyan_components": int(len(cyan_components)),
            "cyan_faces": int(len(faces)),
            "cyan_rejected_small": int(rejected_small),
            "cyan_rejected_nonplanar": int(rejected_nonplanar),
            "cyan_rejected_no_graph_triangle": int(rejected_no_graph_triangle),
        },
    )


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    oriented = faces.copy()
    center = vertices.mean(axis=0)
    for idx, face in enumerate(oriented):
        p0, p1, p2 = vertices[face]
        normal = np.cross(p1 - p0, p2 - p0)
        face_center = (p0 + p1 + p2) / 3.0
        if float(np.dot(normal, face_center - center)) < 0.0:
            oriented[idx, 1], oriented[idx, 2] = oriented[idx, 2], oriented[idx, 1]
    return oriented


def remove_duplicate_and_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_face_area: float,
) -> tuple[np.ndarray, dict[str, int]]:
    unique: dict[tuple[int, int, int], list[int]] = {}
    degenerate = 0
    duplicate = 0
    for face in faces:
        if len(set(map(int, face))) != 3:
            degenerate += 1
            continue
        p0, p1, p2 = vertices[face]
        area = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)) * 0.5)
        if area < min_face_area:
            degenerate += 1
            continue
        key = face_tuple(face)
        if key in unique:
            duplicate += 1
            continue
        unique[key] = [int(face[0]), int(face[1]), int(face[2])]
    cleaned = np.asarray(list(unique.values()), dtype=np.int32) if unique else np.zeros((0, 3), dtype=np.int32)
    return cleaned, {
        "cleanup_removed_degenerate_faces": int(degenerate),
        "cleanup_removed_duplicate_faces": int(duplicate),
    }


def remove_nonmanifold_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces_per_edge: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    if faces.size == 0:
        return faces, {"cleanup_removed_nonmanifold_faces": 0, "cleanup_nonmanifold_edges": 0}

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    areas = np.zeros(faces.shape[0], dtype=np.float32)
    for fi, face in enumerate(faces):
        p0, p1, p2 = vertices[face]
        areas[fi] = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)) * 0.5)
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_to_faces[edge_tuple(int(a), int(b))].append(fi)

    remove: set[int] = set()
    nonmanifold_edges = 0
    for incident in edge_to_faces.values():
        if len(incident) <= max_faces_per_edge:
            continue
        nonmanifold_edges += 1
        keep = sorted(incident, key=lambda idx: areas[idx], reverse=True)[:max_faces_per_edge]
        for idx in incident:
            if idx not in keep:
                remove.add(idx)

    if not remove:
        return faces, {"cleanup_removed_nonmanifold_faces": 0, "cleanup_nonmanifold_edges": int(nonmanifold_edges)}
    keep_mask = np.ones(faces.shape[0], dtype=bool)
    keep_mask[list(remove)] = False
    return faces[keep_mask], {
        "cleanup_removed_nonmanifold_faces": int(len(remove)),
        "cleanup_nonmanifold_edges": int(nonmanifold_edges),
    }


def build_lsm_mesh(coords: np.ndarray, features: np.ndarray, resolution: int, args):
    labels, label_stats = classify_brg_voxels(
        coords,
        features[:, 0],
        features[:, 1],
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
        midpoint_dtri_threshold=args.midpoint_dtri_threshold,
        midpoint_dvert=args.midpoint_dvert,
        midpoint_tolerance=args.midpoint_tolerance,
    )

    vertex_data = extract_vertex_blobs(
        coords,
        features[:, 1],
        resolution,
        threshold=args.vertex_dvert_threshold,
        connectivity=args.vertex_connectivity,
        min_component_size=args.vertex_min_component_size,
        position_mode=args.vertex_position_mode,
    )
    vertices = vertex_data["vertices"]

    skeleton_labels = parse_label_list(args.skeleton_labels)
    skeleton_mask = np.isin(labels, skeleton_labels)
    skeleton_coords, skeleton_stats = close_and_skeletonize(
        coords,
        skeleton_mask,
        closing_radius=args.closing_radius,
        closing_iterations=args.closing_iterations,
        dense_padding=args.dense_padding,
    )

    skeleton_edges, skeleton_edge_stats = trace_skeleton_edges(
        skeleton_coords,
        vertices,
        resolution,
        skeleton_connectivity=args.skeleton_connectivity,
        endpoint_attach_voxels=args.endpoint_attach_voxels,
    )

    cycle_faces = graph_cycle_faces(vertices.shape[0], skeleton_edges)
    cyan_faces, cyan_stats = cyan_planar_faces(
        coords,
        labels,
        vertices,
        skeleton_edges,
        resolution,
        cyan_connectivity=args.cyan_connectivity,
        cyan_min_component_size=args.cyan_min_component_size,
        max_planar_thickness_ratio=args.max_planar_thickness_ratio,
        min_planar_spread_ratio=args.min_planar_spread_ratio,
        nearest_vertices=args.cyan_nearest_vertices,
        require_graph_triangle=args.require_cyan_graph_triangle,
    )

    face_sets = []
    if args.face_source in ("cycles", "both"):
        face_sets.append(cycle_faces)
    if args.face_source in ("cyan", "both"):
        face_sets.append(cyan_faces)
    faces = np.concatenate(face_sets, axis=0) if face_sets else np.zeros((0, 3), dtype=np.int32)

    faces, duplicate_stats = remove_duplicate_and_degenerate_faces(vertices, faces, args.min_face_area)
    faces, nonmanifold_stats = remove_nonmanifold_faces(vertices, faces, max_faces_per_edge=2)
    faces = orient_faces_outward(vertices, faces)

    stats = {}
    stats.update(label_stats)
    stats.update(skeleton_stats)
    stats.update(skeleton_edge_stats)
    stats.update(cyan_stats)
    stats.update(duplicate_stats)
    stats.update(nonmanifold_stats)
    stats.update({
        "algorithm": ALGO_NAME,
        "vertices": int(vertices.shape[0]),
        "edges": int(skeleton_edges.shape[0]),
        "cycle_faces_before_cleanup": int(cycle_faces.shape[0]),
        "cyan_faces_before_cleanup": int(cyan_faces.shape[0]),
        "faces_after_cleanup": int(faces.shape[0]),
        "vertex_dvert_threshold": float(args.vertex_dvert_threshold),
        "vertex_raw_max_barycentric": float(threshold_to_raw_max_barycentric(args.vertex_dvert_threshold)),
        "closing_radius": int(args.closing_radius),
        "closing_iterations": int(args.closing_iterations),
        "endpoint_attach_voxels": float(args.endpoint_attach_voxels),
        "skeleton_labels": ",".join(str(v) for v in skeleton_labels),
        "skeleton_label_mask_voxels": int(skeleton_mask.sum()),
    })
    return vertices, skeleton_edges, faces, skeleton_coords, labels, stats


def write_npz(
    path: str | Path,
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    skeleton_coords: np.ndarray,
    stats: dict,
    resolution: int,
) -> None:
    np.savez_compressed(
        path,
        vertices=vertices.astype(np.float32, copy=False),
        edges=edges.astype(np.int32, copy=False),
        faces=faces.astype(np.int32, copy=False),
        skeleton_coords=skeleton_coords.astype(np.int32, copy=False),
        resolution=np.asarray(resolution, dtype=np.int32),
        algorithm=np.asarray(ALGO_NAME),
        stats_keys=np.asarray(list(stats.keys())),
        stats_values=np.asarray([str(v) for v in stats.values()]),
    )


def parse_label_list(value: str) -> list[int]:
    labels = []
    for item in value.split(","):
        item = item.strip()
        if item:
            labels.append(int(item))
    if not labels:
        raise ValueError("--skeleton_labels must contain at least one label")
    return labels


def print_summary(stats: dict) -> None:
    print(f"Algorithm: {ALGO_NAME} ({ALGO_SHORT_NAME})")
    for key in (
        "vertices",
        "skeleton_labels",
        "skeleton_label_mask_voxels",
        "dense_voxels_before_closing",
        "dense_voxels_after_closing",
        "skeleton_voxels",
        "skeleton_branches",
        "skeleton_edges",
        "skeleton_branch_rejected_unattached",
        "skeleton_branch_rejected_self",
        "cyan_components",
        "cyan_faces",
        "cycle_faces_before_cleanup",
        "cyan_faces_before_cleanup",
        "cleanup_removed_degenerate_faces",
        "cleanup_removed_duplicate_faces",
        "cleanup_nonmanifold_edges",
        "cleanup_removed_nonmanifold_faces",
        "faces_after_cleanup",
    ):
        print(f"{key}: {stats.get(key)}")
    print("Thresholds/settings:")
    print(
        f"  vertex d_vert >= {stats['vertex_dvert_threshold']} "
        f"(raw max barycentric >= {stats['vertex_raw_max_barycentric']:.6f})"
    )
    print(f"  skeleton labels = {stats['skeleton_labels']}")
    print(f"  skeleton closing radius = {stats['closing_radius']}")
    print(f"  skeleton closing iterations = {stats['closing_iterations']}")
    print(f"  skeleton endpoint attach radius = {stats['endpoint_attach_voxels']} voxels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an LSM mesh from BRG labels.")
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
    parser.add_argument("--vertex_clearance", type=int, default=2)
    parser.add_argument("--attach_radius", type=int, default=3)
    parser.add_argument("--ridge_midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--ridge_midpoint_tolerance", type=float, default=0.075)
    parser.add_argument("--midpoint_dtri_threshold", type=float, default=0.25)
    parser.add_argument("--midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--midpoint_tolerance", type=float, default=0.05)

    parser.add_argument(
        "--skeleton_labels",
        type=str,
        default="1,2",
        help=(
            "Comma-separated labels used for closing/skeletonization. "
            "Default 1,2 = red vertex cores + orange core clearance. "
            "Try 1,2,3,7 for core + edges + midpoint evidence."
        ),
    )
    parser.add_argument("--closing_radius", type=int, default=1)
    parser.add_argument("--closing_iterations", type=int, default=1)
    parser.add_argument("--dense_padding", type=int, default=2)
    parser.add_argument("--skeleton_connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument("--endpoint_attach_voxels", type=float, default=6.0)

    parser.add_argument("--face_source", choices=["cycles", "cyan", "both"], default="both")
    parser.add_argument("--cyan_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--cyan_min_component_size", type=int, default=8)
    parser.add_argument("--max_planar_thickness_ratio", type=float, default=0.025)
    parser.add_argument("--min_planar_spread_ratio", type=float, default=0.05)
    parser.add_argument("--cyan_nearest_vertices", type=int, default=8)
    parser.add_argument("--require_cyan_graph_triangle", action="store_true")
    parser.add_argument("--min_face_area", type=float, default=1e-12)

    parser.add_argument("--out_obj", type=str, default=None)
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--out_npz", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)
    stem = default_output_stem(path)

    vertices, edges, faces, skeleton_coords, labels, stats = build_lsm_mesh(coords, features, resolution, args)

    out_dir = default_results_dir(path)
    out_obj = Path(args.out_obj) if args.out_obj else out_dir / f"{stem}_label_skeleton_mesh.obj"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_label_skeleton_mesh.ply"
    out_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{stem}_label_skeleton_mesh.npz"

    write_obj(out_obj, vertices, faces)
    write_ply_mesh(out_ply, vertices, faces, edges)
    write_npz(out_npz, vertices, edges, faces, skeleton_coords, stats, resolution)

    print_summary(stats)
    print(f"Saved {out_obj}")
    print(f"Saved {out_ply}")
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
