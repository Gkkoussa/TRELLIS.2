"""
Build a triangle mesh with primal/dual watershed clustering on sparse voxels.

Inputs:
  F = features[:, 0] = 3 * min(a, b, c)
  D = features[:, 1] = (3 * max(a, b, c) - 1) / 2

Algorithm:
  1. Build a 6-connected graph over active voxels.
  2. Extract high-F seed blobs. These are face seeds.
  3. Watershed/flood F from those seeds. Each voxel gets a primal face label.
  4. Extract high-D seed blobs. These are vertex seeds.
  5. Watershed/flood D from those seeds. Each voxel gets a dual vertex label.
  6. Build one mesh vertex per dual seed.
  7. Build one mesh face per primal basin by taking the three dominant dual
     labels inside that basin.

This is intentionally separate from the BRG and centroid-nearest builders.
Only coords, F, and D are used. No offsets are used.
"""

from __future__ import annotations

import argparse
import csv
import heapq
from collections import Counter
from pathlib import Path

import numpy as np

from extract_barycentric_ridge_graph import (
    component_core_indices,
    component_position,
    connected_components,
    default_output_stem,
    default_results_dir,
    infer_resolution,
    load_triangle_field,
    threshold_to_raw_max_barycentric,
    voxel_centers,
)


ALGO_NAME = "Primal Dual Watershed Mesh"


SIX_OFFSETS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def raw_min_barycentric_from_f(value: float) -> float:
    return float(value) / 3.0


def build_coord_to_index(coords: np.ndarray) -> dict[tuple[int, int, int], int]:
    return {tuple(c.tolist()): int(i) for i, c in enumerate(coords)}


def build_six_neighbors(coords: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    coord_to_index = build_coord_to_index(coords)
    neighbors: list[list[int]] = [[] for _ in range(coords.shape[0])]
    pairs = []
    positive_offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    for i, coord in enumerate(coords):
        x, y, z = coord.tolist()
        for dx, dy, dz in positive_offsets:
            j = coord_to_index.get((x + dx, y + dy, z + dz))
            if j is None:
                continue
            j = int(j)
            neighbors[i].append(j)
            neighbors[j].append(i)
            pairs.append((i, j))

    if pairs:
        pair_array = np.asarray(pairs, dtype=np.int64)
    else:
        pair_array = np.zeros((0, 2), dtype=np.int64)
    return neighbors, pair_array


def seed_sort_key(record: dict[str, object], mode: str) -> tuple[float, float, int]:
    if mode == "peak":
        return (-float(record["peak_value"]), -float(record["size"]), int(record["component_id"]))
    if mode == "size_then_peak":
        return (-float(record["size"]), -float(record["peak_value"]), int(record["component_id"]))
    raise ValueError(f"Unsupported seed sort mode: {mode}")


def extract_seed_blobs(
    coords: np.ndarray,
    values: np.ndarray,
    resolution: int,
    *,
    threshold: float,
    connectivity: int,
    min_component_size: int,
    target_count: int,
    fill_small_by: str,
    position_mode: str,
    core_mode: str,
) -> dict[str, np.ndarray | int | str]:
    mask = np.asarray(values, dtype=np.float32) >= float(threshold)
    components = connected_components(coords, mask, connectivity=connectivity)

    main_records = []
    small_records = []
    for component_id, component in enumerate(components):
        position = component_position(coords, values, component, resolution, threshold, position_mode)
        core_idx = component_core_indices(coords, values, component, resolution, position, core_mode)
        record = {
            "component_id": int(component_id),
            "component": component,
            "position": position,
            "core_indices": core_idx,
            "size": int(component.size),
            "peak_value": float(values[component].max()),
        }
        if component.size >= int(min_component_size):
            main_records.append(record)
        else:
            small_records.append(record)

    target_fill_count = 0
    target_trim_count = 0
    target = int(target_count)
    records = list(main_records)

    if target > 0 and len(records) > target:
        records.sort(key=lambda item: seed_sort_key(item, "size_then_peak"))
        target_trim_count = len(records) - target
        records = records[:target]
    elif target > 0 and len(records) < target and fill_small_by != "none":
        needed = target - len(records)
        small_records.sort(key=lambda item: seed_sort_key(item, fill_small_by))
        fill_records = small_records[:needed]
        records.extend(fill_records)
        target_fill_count = len(fill_records)

    records.sort(key=lambda item: int(item["component_id"]))

    seed_id_for_index = np.full(coords.shape[0], -1, dtype=np.int32)
    seed_mask = np.zeros(coords.shape[0], dtype=bool)
    seed_core_mask = np.zeros(coords.shape[0], dtype=bool)
    positions = []
    peak_values = []
    sizes = []
    core_counts = []

    for seed_id, record in enumerate(records):
        component = np.asarray(record["component"], dtype=np.int64)
        core_idx = np.asarray(record["core_indices"], dtype=np.int64)
        seed_id_for_index[component] = int(seed_id)
        seed_mask[component] = True
        if core_idx.size:
            seed_core_mask[core_idx] = True
        positions.append(np.asarray(record["position"], dtype=np.float32))
        peak_values.append(float(record["peak_value"]))
        sizes.append(int(record["size"]))
        core_counts.append(int(core_idx.size))

    if positions:
        position_array = np.stack(positions).astype(np.float32, copy=False)
    else:
        position_array = np.zeros((0, 3), dtype=np.float32)

    return {
        "positions": position_array,
        "peak_value": np.asarray(peak_values, dtype=np.float32),
        "component_size": np.asarray(sizes, dtype=np.int32),
        "core_count": np.asarray(core_counts, dtype=np.int32),
        "seed_id_for_index": seed_id_for_index,
        "seed_mask": seed_mask,
        "seed_core_mask": seed_core_mask,
        "raw_mask_voxels": int(mask.sum()),
        "raw_components": int(len(components)),
        "main_components_before_target": int(len(main_records)),
        "small_components_available": int(len(small_records)),
        "target_fill_count": int(target_fill_count),
        "target_trim_count": int(target_trim_count),
        "target_count": int(target),
        "fill_small_by": str(fill_small_by),
    }


def watershed_labels(
    values: np.ndarray,
    neighbors: list[list[int]],
    seed_id_for_index: np.ndarray,
) -> tuple[np.ndarray, int]:
    labels = np.asarray(seed_id_for_index, dtype=np.int32).copy()
    heap: list[tuple[float, int, int]] = []
    for index in np.flatnonzero(labels >= 0):
        label = int(labels[int(index)])
        heapq.heappush(heap, (-float(values[int(index)]), int(index), label))

    while heap:
        _, index, label = heapq.heappop(heap)
        if int(labels[index]) != int(label):
            continue
        for neighbor in neighbors[index]:
            neighbor = int(neighbor)
            if labels[neighbor] >= 0:
                continue
            labels[neighbor] = int(label)
            heapq.heappush(heap, (-float(values[neighbor]), neighbor, int(label)))

    return labels, int(np.sum(labels < 0))


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            parent = self.find(parent)
            self.parent[value] = parent
        return parent

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def compact_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    unique = np.asarray(sorted(int(v) for v in np.unique(labels) if int(v) >= 0), dtype=np.int32)
    mapping = {int(old): int(new) for new, old in enumerate(unique.tolist())}
    compact = np.full(labels.shape, -1, dtype=np.int32)
    for old, new in mapping.items():
        compact[labels == old] = int(new)
    return compact, mapping


def merge_fake_boundaries(
    face_labels: np.ndarray,
    neighbor_pairs: np.ndarray,
    f_values: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if threshold < 0.0 or face_labels.size == 0:
        return face_labels, {
            "fake_boundary_merge_enabled": 0,
            "fake_boundary_threshold": float(threshold),
            "fake_boundary_pairs": 0,
            "fake_boundary_unions": 0,
            "face_labels_after_merge": int(np.max(face_labels) + 1) if np.any(face_labels >= 0) else 0,
        }

    num_labels = int(np.max(face_labels) + 1) if np.any(face_labels >= 0) else 0
    uf = UnionFind(num_labels)
    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}

    for i, j in neighbor_pairs:
        a = int(face_labels[int(i)])
        b = int(face_labels[int(j)])
        if a < 0 or b < 0 or a == b:
            continue
        key = (a, b) if a < b else (b, a)
        value = 0.5 * (float(f_values[int(i)]) + float(f_values[int(j)]))
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1

    unions = 0
    for key, total in sums.items():
        mean_boundary_f = total / float(counts[key])
        if mean_boundary_f >= float(threshold):
            if uf.union(key[0], key[1]):
                unions += 1

    merged = face_labels.copy()
    for label in range(num_labels):
        root = uf.find(label)
        merged[face_labels == label] = root
    merged, _ = compact_labels(merged)
    return merged, {
        "fake_boundary_merge_enabled": 1,
        "fake_boundary_threshold": float(threshold),
        "fake_boundary_pairs": int(len(sums)),
        "fake_boundary_unions": int(unions),
        "face_labels_after_merge": int(np.max(merged) + 1) if np.any(merged >= 0) else 0,
    }


def boundary_and_junction_masks(
    coords: np.ndarray,
    face_labels: np.ndarray,
    neighbor_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    boundary_mask = np.zeros(face_labels.shape, dtype=bool)
    boundary_links = 0
    boundary_pairs: set[tuple[int, int]] = set()

    for i, j in neighbor_pairs:
        a = int(face_labels[int(i)])
        b = int(face_labels[int(j)])
        if a < 0 or b < 0 or a == b:
            continue
        boundary_mask[int(i)] = True
        boundary_mask[int(j)] = True
        boundary_links += 1
        boundary_pairs.add((a, b) if a < b else (b, a))

    coord_to_index = build_coord_to_index(coords)
    junction_mask = np.zeros(face_labels.shape, dtype=bool)
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                offsets.append((dx, dy, dz))

    for index in np.flatnonzero(boundary_mask):
        x, y, z = coords[int(index)].tolist()
        labels = set()
        for dx, dy, dz in offsets:
            neighbor = coord_to_index.get((x + dx, y + dy, z + dz))
            if neighbor is None:
                continue
            label = int(face_labels[int(neighbor)])
            if label >= 0:
                labels.add(label)
        if len(labels) >= 3:
            junction_mask[int(index)] = True

    stats = {
        "boundary_links": int(boundary_links),
        "boundary_voxels": int(boundary_mask.sum()),
        "boundary_face_pairs": int(len(boundary_pairs)),
        "junction_voxels": int(junction_mask.sum()),
    }
    return boundary_mask, junction_mask, stats


def sorted_edge(edge: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    if a == b:
        raise ValueError("self-edge is not allowed")
    return (a, b) if a < b else (b, a)


def edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for face in faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edge_set.add(sorted_edge((a, b)))
        edge_set.add(sorted_edge((b, c)))
        edge_set.add(sorted_edge((c, a)))
    if not edge_set:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(edge_set), dtype=np.int32)


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    center = vertices.mean(axis=0) if vertices.size else np.zeros(3, dtype=np.float32)
    oriented = faces.copy()
    for i, face in enumerate(oriented):
        p0, p1, p2 = vertices[face]
        normal = np.cross(p1 - p0, p2 - p0)
        face_center = (p0 + p1 + p2) / 3.0
        if float(np.dot(normal, face_center - center)) < 0.0:
            oriented[i] = np.asarray([face[0], face[2], face[1]], dtype=np.int32)
    return oriented


def transform_to_output_axes(points: np.ndarray, output_axes: str) -> np.ndarray:
    points = np.asarray(points)
    if output_axes == "voxel":
        return points.copy()
    if output_axes != "input":
        raise ValueError(f"Unsupported output_axes: {output_axes}")
    transformed = np.empty_like(points)
    transformed[..., 0] = points[..., 0]
    transformed[..., 1] = points[..., 2]
    transformed[..., 2] = -points[..., 1]
    return transformed


def stats_for_output_axes(
    stats: dict[str, np.ndarray | int | float | str],
    output_axes: str,
) -> dict[str, np.ndarray | int | float | str]:
    out: dict[str, np.ndarray | int | float | str] = dict(stats)
    out["output_axes"] = str(output_axes)
    if output_axes == "voxel":
        return out
    position_keys = {
        "face_seed_positions",
        "centroid_positions",
    }
    for key, value in stats.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 2
            and value.shape[-1] == 3
            and np.issubdtype(value.dtype, np.floating)
            and (key in position_keys or key.endswith("_positions"))
        ):
            out[key] = transform_to_output_axes(value, output_axes)
    return out


def triangle_area_from_ids(vertex_positions: np.ndarray, corner_ids: list[int] | tuple[int, int, int]) -> float:
    p0, p1, p2 = vertex_positions[np.asarray(corner_ids, dtype=np.int32)]
    return 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))


def basin_plane_axes(points: np.ndarray, fallback_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] >= 3:
        center = points.mean(axis=0)
        centered = points - center[None, :]
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_u = vh[0]
        normal = vh[-1]
    else:
        center = fallback_points.mean(axis=0)
        axis_u = fallback_points[1] - fallback_points[0]
        normal = np.cross(fallback_points[1] - fallback_points[0], fallback_points[2] - fallback_points[0])

    axis_u_norm = float(np.linalg.norm(axis_u))
    normal_norm = float(np.linalg.norm(normal))
    if axis_u_norm < 1e-12:
        axis_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis_u = axis_u / axis_u_norm
    if normal_norm < 1e-12:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        normal = normal / normal_norm
    axis_v = np.cross(normal, axis_u)
    axis_v_norm = float(np.linalg.norm(axis_v))
    if axis_v_norm < 1e-12:
        axis_v = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        axis_v = axis_v / axis_v_norm
    return center.astype(np.float32, copy=False), axis_u.astype(np.float32, copy=False), axis_v.astype(np.float32, copy=False)


def split_top4_corners(
    coords: np.ndarray,
    face_labels: np.ndarray,
    vertex_positions: np.ndarray,
    resolution: int,
    basin_id: int,
    corner_ids: list[int],
) -> list[tuple[int, int, int]]:
    corner_points = vertex_positions[np.asarray(corner_ids, dtype=np.int32)]
    basin_indices = np.flatnonzero(face_labels == int(basin_id))
    basin_points = voxel_centers(coords[basin_indices], resolution) if basin_indices.size else np.zeros((0, 3), dtype=np.float32)
    center, axis_u, axis_v = basin_plane_axes(basin_points, corner_points)

    rel = corner_points - center[None, :]
    projected = np.stack([rel @ axis_u, rel @ axis_v], axis=1)
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    order = np.argsort(angles, kind="stable")
    ordered = [int(corner_ids[int(i)]) for i in order.tolist()]

    p = vertex_positions[np.asarray(ordered, dtype=np.int32)]
    diagonal_ac = float(np.linalg.norm(p[2] - p[0]))
    diagonal_bd = float(np.linalg.norm(p[3] - p[1]))
    if diagonal_ac <= diagonal_bd:
        return [(ordered[0], ordered[1], ordered[2]), (ordered[0], ordered[2], ordered[3])]
    return [(ordered[0], ordered[1], ordered[3]), (ordered[1], ordered[2], ordered[3])]


def assemble_faces_from_primal_dual(
    coords: np.ndarray,
    face_labels: np.ndarray,
    vertex_labels: np.ndarray,
    vertex_positions: np.ndarray,
    resolution: int,
    *,
    corner_policy: str,
    min_corner_voxels: int,
    top4_split_ratio: float,
    min_face_area: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    valid_face_labels = face_labels[face_labels >= 0]
    num_face_labels = int(valid_face_labels.max() + 1) if valid_face_labels.size else 0
    counters = [Counter() for _ in range(num_face_labels)]
    basin_sizes = np.zeros(num_face_labels, dtype=np.int32)

    for face_label, vertex_label in zip(face_labels, vertex_labels):
        face_label = int(face_label)
        vertex_label = int(vertex_label)
        if face_label < 0:
            continue
        basin_sizes[face_label] += 1
        if vertex_label >= 0:
            counters[face_label][vertex_label] += 1

    face_by_key: dict[tuple[int, int, int], dict[str, object]] = {}
    rejected_too_few_corners = 0
    rejected_nontri = 0
    rejected_weak_corner = 0
    rejected_degenerate = 0
    duplicate_faces = 0
    top4_split_basins = 0
    top4_split_faces = 0
    top4_not_close = 0

    for basin_id, counter in enumerate(counters):
        if not counter:
            rejected_too_few_corners += 1
            continue

        labels_and_counts = sorted(counter.items(), key=lambda item: (-int(item[1]), int(item[0])))
        strong = [(int(label), int(count)) for label, count in labels_and_counts if int(count) >= int(min_corner_voxels)]
        if len(strong) < 3:
            rejected_too_few_corners += 1
            continue
        if corner_policy == "exact3" and len(strong) != 3:
            rejected_nontri += 1
            continue
        if corner_policy != "top3" and corner_policy != "exact3" and corner_policy != "top4_split":
            raise ValueError(f"Unsupported corner_policy: {corner_policy}")

        selected_faces: list[tuple[int, int, int]]
        selected = strong[:3]
        corner_ids = [item[0] for item in selected]
        corner_counts = [item[1] for item in selected]

        if corner_policy == "top4_split" and len(strong) >= 4:
            fourth_count = int(strong[3][1])
            third_count = max(int(strong[2][1]), 1)
            if float(fourth_count) >= float(top4_split_ratio) * float(third_count):
                top4_ids = [int(item[0]) for item in strong[:4]]
                selected_faces = split_top4_corners(
                    coords,
                    face_labels,
                    vertex_positions,
                    resolution,
                    basin_id,
                    top4_ids,
                )
                selected = strong[:4]
                corner_ids = top4_ids
                corner_counts = [int(item[1]) for item in selected]
                top4_split_basins += 1
            else:
                selected_faces = [tuple(corner_ids)]
                top4_not_close += 1
        else:
            selected_faces = [tuple(corner_ids)]

        if min(corner_counts) < int(min_corner_voxels):
            rejected_weak_corner += 1
            continue

        for face_tuple in selected_faces:
            area = triangle_area_from_ids(vertex_positions, face_tuple)
            if area < float(min_face_area):
                rejected_degenerate += 1
                continue

            face_counts = [int(counter[int(label)]) for label in face_tuple]
            key = tuple(sorted(int(v) for v in face_tuple))
            support = int(sum(face_counts))
            record = {
                "face": tuple(int(v) for v in face_tuple),
                "basin_id": int(basin_id),
                "basin_size": int(basin_sizes[basin_id]),
                "corner_label_count": int(len(strong)),
                "corner0_voxels": int(face_counts[0]),
                "corner1_voxels": int(face_counts[1]),
                "corner2_voxels": int(face_counts[2]),
                "support": int(support),
                "area": float(area),
            }
            old = face_by_key.get(key)
            if old is None:
                face_by_key[key] = record
                if len(selected_faces) == 2:
                    top4_split_faces += 1
            else:
                duplicate_faces += 1
                if support > int(old["support"]):
                    face_by_key[key] = record

    records = [face_by_key[key] for key in sorted(face_by_key)]
    if records:
        faces = np.asarray([record["face"] for record in records], dtype=np.int32)
    else:
        faces = np.zeros((0, 3), dtype=np.int32)

    stats = {
        "accepted_basin_id": np.asarray([record["basin_id"] for record in records], dtype=np.int32),
        "accepted_basin_size": np.asarray([record["basin_size"] for record in records], dtype=np.int32),
        "accepted_corner_label_count": np.asarray(
            [record["corner_label_count"] for record in records],
            dtype=np.int32,
        ),
        "accepted_corner0_voxels": np.asarray([record["corner0_voxels"] for record in records], dtype=np.int32),
        "accepted_corner1_voxels": np.asarray([record["corner1_voxels"] for record in records], dtype=np.int32),
        "accepted_corner2_voxels": np.asarray([record["corner2_voxels"] for record in records], dtype=np.int32),
        "accepted_face_area": np.asarray([record["area"] for record in records], dtype=np.float32),
        "rejected_too_few_corners": int(rejected_too_few_corners),
        "rejected_nontri": int(rejected_nontri),
        "rejected_weak_corner": int(rejected_weak_corner),
        "rejected_degenerate": int(rejected_degenerate),
        "duplicate_faces": int(duplicate_faces),
        "top4_split_basins": int(top4_split_basins),
        "top4_split_faces": int(top4_split_faces),
        "top4_not_close": int(top4_not_close),
        "primal_basins": int(num_face_labels),
    }
    return faces, stats


def extract_primal_dual_mesh(
    coords: np.ndarray,
    features: np.ndarray,
    resolution: int,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | int | float | str]]:
    f_values = np.asarray(features[:, 0], dtype=np.float32)
    d_values = np.asarray(features[:, 1], dtype=np.float32)

    neighbors, neighbor_pairs = build_six_neighbors(coords)

    face_seed_data = extract_seed_blobs(
        coords,
        f_values,
        resolution,
        threshold=args.face_seed_threshold,
        connectivity=args.face_seed_connectivity,
        min_component_size=args.face_seed_min_component_size,
        target_count=args.face_seed_target_count,
        fill_small_by=args.face_seed_fill_small_by,
        position_mode=args.face_seed_position_mode,
        core_mode=args.face_seed_core_mode,
    )
    face_labels, unassigned_face_voxels = watershed_labels(
        f_values,
        neighbors,
        np.asarray(face_seed_data["seed_id_for_index"], dtype=np.int32),
    )

    face_labels, merge_stats = merge_fake_boundaries(
        face_labels,
        neighbor_pairs,
        f_values,
        threshold=args.merge_fake_boundaries_f_threshold,
    )

    vertex_seed_data = extract_seed_blobs(
        coords,
        d_values,
        resolution,
        threshold=args.vertex_seed_threshold,
        connectivity=args.vertex_seed_connectivity,
        min_component_size=args.vertex_seed_min_component_size,
        target_count=args.vertex_seed_target_count,
        fill_small_by=args.vertex_seed_fill_small_by,
        position_mode=args.vertex_seed_position_mode,
        core_mode=args.vertex_seed_core_mode,
    )
    vertex_labels, unassigned_vertex_voxels = watershed_labels(
        d_values,
        neighbors,
        np.asarray(vertex_seed_data["seed_id_for_index"], dtype=np.int32),
    )
    vertices = np.asarray(vertex_seed_data["positions"], dtype=np.float32)

    boundary_mask, junction_mask, boundary_stats = boundary_and_junction_masks(coords, face_labels, neighbor_pairs)

    faces, face_stats = assemble_faces_from_primal_dual(
        coords,
        face_labels,
        vertex_labels,
        vertices,
        resolution,
        corner_policy=args.face_corner_policy,
        min_corner_voxels=args.min_corner_voxels,
        top4_split_ratio=args.top4_split_ratio,
        min_face_area=args.min_face_area,
    )
    if not args.no_orient_outward:
        faces = orient_faces_outward(vertices, faces)
    edges = edges_from_faces(faces)

    stats: dict[str, np.ndarray | int | float | str] = {
        "algorithm": ALGO_NAME,
        "resolution": int(resolution),
        "active_voxels": int(coords.shape[0]),
        "graph_connectivity": 6,
        "graph_edges": int(neighbor_pairs.shape[0]),
        "normal_pruning": "not_available",
        "vertices": int(vertices.shape[0]),
        "edges": int(edges.shape[0]),
        "faces": int(faces.shape[0]),
        "face_seed_threshold": float(args.face_seed_threshold),
        "face_seed_raw_min_barycentric": raw_min_barycentric_from_f(args.face_seed_threshold),
        "face_seed_connectivity": int(args.face_seed_connectivity),
        "face_seed_min_component_size": int(args.face_seed_min_component_size),
        "face_seed_target_count": int(args.face_seed_target_count),
        "face_seed_fill_small_by": str(args.face_seed_fill_small_by),
        "face_seed_position_mode": str(args.face_seed_position_mode),
        "face_seed_core_mode": str(args.face_seed_core_mode),
        "face_seed_raw_mask_voxels": int(face_seed_data["raw_mask_voxels"]),
        "face_seed_raw_components": int(face_seed_data["raw_components"]),
        "face_seed_main_components_before_target": int(face_seed_data["main_components_before_target"]),
        "face_seed_small_components_available": int(face_seed_data["small_components_available"]),
        "face_seed_target_fill_count": int(face_seed_data["target_fill_count"]),
        "face_seed_target_trim_count": int(face_seed_data["target_trim_count"]),
        "face_seed_count": int(np.asarray(face_seed_data["positions"]).shape[0]),
        "unassigned_face_voxels": int(unassigned_face_voxels),
        "vertex_seed_threshold": float(args.vertex_seed_threshold),
        "vertex_seed_raw_max_barycentric": threshold_to_raw_max_barycentric(args.vertex_seed_threshold),
        "vertex_seed_connectivity": int(args.vertex_seed_connectivity),
        "vertex_seed_min_component_size": int(args.vertex_seed_min_component_size),
        "vertex_seed_target_count": int(args.vertex_seed_target_count),
        "vertex_seed_fill_small_by": str(args.vertex_seed_fill_small_by),
        "vertex_seed_position_mode": str(args.vertex_seed_position_mode),
        "vertex_seed_core_mode": str(args.vertex_seed_core_mode),
        "vertex_seed_raw_mask_voxels": int(vertex_seed_data["raw_mask_voxels"]),
        "vertex_seed_raw_components": int(vertex_seed_data["raw_components"]),
        "vertex_seed_main_components_before_target": int(vertex_seed_data["main_components_before_target"]),
        "vertex_seed_small_components_available": int(vertex_seed_data["small_components_available"]),
        "vertex_seed_target_fill_count": int(vertex_seed_data["target_fill_count"]),
        "vertex_seed_target_trim_count": int(vertex_seed_data["target_trim_count"]),
        "vertex_seed_count": int(vertices.shape[0]),
        "unassigned_vertex_voxels": int(unassigned_vertex_voxels),
        "face_corner_policy": str(args.face_corner_policy),
        "min_corner_voxels": int(args.min_corner_voxels),
        "top4_split_ratio": float(args.top4_split_ratio),
        "min_face_area": float(args.min_face_area),
        "primal_basins": int(face_stats["primal_basins"]),
        "rejected_too_few_corners": int(face_stats["rejected_too_few_corners"]),
        "rejected_nontri": int(face_stats["rejected_nontri"]),
        "rejected_weak_corner": int(face_stats["rejected_weak_corner"]),
        "rejected_degenerate": int(face_stats["rejected_degenerate"]),
        "duplicate_faces": int(face_stats["duplicate_faces"]),
        "top4_split_basins": int(face_stats["top4_split_basins"]),
        "top4_split_faces": int(face_stats["top4_split_faces"]),
        "top4_not_close": int(face_stats["top4_not_close"]),
        "face_labels": face_labels.astype(np.int32, copy=False),
        "vertex_labels": vertex_labels.astype(np.int32, copy=False),
        "face_seed_positions": np.asarray(face_seed_data["positions"], dtype=np.float32),
        "face_seed_peak_value": np.asarray(face_seed_data["peak_value"], dtype=np.float32),
        "face_seed_component_size": np.asarray(face_seed_data["component_size"], dtype=np.int32),
        "vertex_seed_peak_value": np.asarray(vertex_seed_data["peak_value"], dtype=np.float32),
        "vertex_seed_component_size": np.asarray(vertex_seed_data["component_size"], dtype=np.int32),
        "face_seed_core_mask": np.asarray(face_seed_data["seed_core_mask"], dtype=bool),
        "vertex_seed_core_mask": np.asarray(vertex_seed_data["seed_core_mask"], dtype=bool),
        "boundary_mask": boundary_mask,
        "junction_mask": junction_mask,
        **merge_stats,
        **boundary_stats,
        "accepted_basin_id": face_stats["accepted_basin_id"],
        "accepted_basin_size": face_stats["accepted_basin_size"],
        "accepted_corner_label_count": face_stats["accepted_corner_label_count"],
        "accepted_corner0_voxels": face_stats["accepted_corner0_voxels"],
        "accepted_corner1_voxels": face_stats["accepted_corner1_voxels"],
        "accepted_corner2_voxels": face_stats["accepted_corner2_voxels"],
        "accepted_face_area": face_stats["accepted_face_area"],
    }
    return vertices, edges, faces, stats


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {ALGO_NAME}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            f.write(f"f {a} {b} {c}\n")


def write_mesh_ply(path: str | Path, vertices: np.ndarray, edges: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment algorithm {ALGO_NAME}\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for edge in edges:
            f.write(f"{int(edge[0])} {int(edge[1])}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_debug_ply(
    path: str | Path,
    coords: np.ndarray,
    resolution: int,
    stats: dict[str, object],
    output_axes: str,
) -> None:
    labels = np.zeros(coords.shape[0], dtype=np.uint8)
    labels[np.asarray(stats["face_seed_core_mask"], dtype=bool)] = 1
    labels[np.asarray(stats["vertex_seed_core_mask"], dtype=bool)] = 2
    labels[np.asarray(stats["boundary_mask"], dtype=bool)] = 3
    labels[np.asarray(stats["junction_mask"], dtype=bool)] = 4

    selected = np.flatnonzero(labels > 0)
    points = voxel_centers(coords[selected], resolution) if selected.size else np.zeros((0, 3), dtype=np.float32)
    points = transform_to_output_axes(points, output_axes)
    colors = {
        1: (40, 230, 120),
        2: (255, 40, 40),
        3: (0, 190, 255),
        4: (255, 220, 40),
    }
    names = {
        1: "face_seed_core",
        2: "vertex_seed_core",
        3: "primal_boundary_voxel",
        4: "junction_voxel",
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment algorithm {ALGO_NAME}\n")
        for label, name in names.items():
            r, g, b = colors[label]
            f.write(f"comment color {label} {name} rgb=({r},{g},{b})\n")
        f.write(f"element vertex {selected.size}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")
        for point, label in zip(points, labels[selected]):
            r, g, b = colors[int(label)]
            f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {r} {g} {b} {int(label)}\n")


def write_npz(
    path: str | Path,
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    stats: dict[str, np.ndarray | int | float | str],
) -> None:
    payload = {
        "vertices": vertices.astype(np.float32, copy=False),
        "edges": edges.astype(np.int32, copy=False),
        "faces": faces.astype(np.int32, copy=False),
    }
    for key, value in stats.items():
        if key.endswith("_mask"):
            continue
        if isinstance(value, np.ndarray):
            payload[key] = value
        elif isinstance(value, str):
            payload[key] = np.asarray(value)
        elif isinstance(value, int):
            payload[key] = np.asarray(value, dtype=np.int32)
        else:
            payload[key] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(path, **payload)


def write_csv(path: str | Path, faces: np.ndarray, stats: dict[str, np.ndarray | int | float | str]) -> None:
    scalar_keys = [
        key
        for key, value in stats.items()
        if not isinstance(value, np.ndarray) and not key.endswith("_mask")
    ]
    array_keys = [
        "accepted_basin_id",
        "accepted_basin_size",
        "accepted_corner_label_count",
        "accepted_corner0_voxels",
        "accepted_corner1_voxels",
        "accepted_corner2_voxels",
        "accepted_face_area",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        for key in sorted(scalar_keys):
            writer.writerow([key, stats[key]])
        writer.writerow([])
        writer.writerow(["face_id", "v0", "v1", "v2", *array_keys])
        for face_id, face in enumerate(faces):
            row = [face_id, int(face[0]), int(face[1]), int(face[2])]
            for key in array_keys:
                row.append(stats[key][face_id])
            writer.writerow(row)


def print_summary(stats: dict[str, np.ndarray | int | float | str]) -> None:
    print(f"Algorithm: {ALGO_NAME}")
    print(f"Resolution: {stats['resolution']}")
    print(f"Active voxels: {stats['active_voxels']}")
    print(f"Output axes: {stats.get('output_axes', 'voxel')}")
    print(f"Graph: 6-connectivity, links={stats['graph_edges']}, normal pruning={stats['normal_pruning']}")
    print("Primal F seeds:")
    print(
        f"  F >= {stats['face_seed_threshold']} "
        f"(raw min barycentric >= {float(stats['face_seed_raw_min_barycentric']):.6f})"
    )
    print(f"  seed connectivity = {stats['face_seed_connectivity']}")
    print(f"  min component size = {stats['face_seed_min_component_size']}")
    print(f"  target count = {stats['face_seed_target_count']}")
    print(f"  fill small by = {stats['face_seed_fill_small_by']}")
    print(f"  raw components = {stats['face_seed_raw_components']}")
    print(f"  main components before target = {stats['face_seed_main_components_before_target']}")
    print(f"  target-filled small components = {stats['face_seed_target_fill_count']}")
    print(f"  target-trimmed components = {stats['face_seed_target_trim_count']}")
    print(f"  final face seed count = {stats['face_seed_count']}")
    print(f"  unassigned F-watershed voxels = {stats['unassigned_face_voxels']}")
    print("Dual D seeds:")
    print(
        f"  D >= {stats['vertex_seed_threshold']} "
        f"(raw max barycentric >= {float(stats['vertex_seed_raw_max_barycentric']):.6f})"
    )
    print(f"  seed connectivity = {stats['vertex_seed_connectivity']}")
    print(f"  min component size = {stats['vertex_seed_min_component_size']}")
    print(f"  final vertex seed count = {stats['vertex_seed_count']}")
    print(f"  unassigned D-watershed voxels = {stats['unassigned_vertex_voxels']}")
    print("Primal map:")
    print(f"  basins = {stats['primal_basins']}")
    print(f"  boundary links = {stats['boundary_links']}")
    print(f"  boundary voxels = {stats['boundary_voxels']}")
    print(f"  boundary face pairs = {stats['boundary_face_pairs']}")
    print(f"  junction voxels = {stats['junction_voxels']}")
    print("Face assembly:")
    print(f"  corner policy = {stats['face_corner_policy']}")
    print(f"  top4 split ratio = {stats['top4_split_ratio']}")
    print(f"  accepted faces = {stats['faces']}")
    print(f"  edges = {stats['edges']}")
    print(f"  vertices = {stats['vertices']}")
    print(f"  rejected too few corners = {stats['rejected_too_few_corners']}")
    print(f"  rejected non-tri = {stats['rejected_nontri']}")
    print(f"  rejected weak corner = {stats['rejected_weak_corner']}")
    print(f"  rejected degenerate = {stats['rejected_degenerate']}")
    print(f"  duplicate faces collapsed = {stats['duplicate_faces']}")
    print(f"  top4 split basins = {stats['top4_split_basins']}")
    print(f"  top4 split faces accepted = {stats['top4_split_faces']}")
    print(f"  top4 candidates not close = {stats['top4_not_close']}")
    print("Fake-boundary merge:")
    print(f"  enabled = {stats['fake_boundary_merge_enabled']}")
    print(f"  threshold = {stats['fake_boundary_threshold']}")
    print(f"  unions = {stats['fake_boundary_unions']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO_NAME}: primal/dual watershed reconstruction.")
    parser.add_argument("path", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--resolution", type=int, default=None)

    parser.add_argument("--face_seed_threshold", type=float, default=0.56)
    parser.add_argument("--face_seed_connectivity", type=int, choices=[6, 18, 26], default=6)
    parser.add_argument("--face_seed_min_component_size", type=int, default=3)
    parser.add_argument("--face_seed_target_count", type=int, default=0)
    parser.add_argument("--face_seed_fill_small_by", choices=["none", "peak", "size_then_peak"], default="none")
    parser.add_argument("--face_seed_position_mode", choices=["peak", "mean", "weighted"], default="weighted")
    parser.add_argument(
        "--face_seed_core_mode",
        choices=["closest4", "peak", "position_nearest", "blob", "none"],
        default="none",
    )

    parser.add_argument("--vertex_seed_threshold", type=float, default=0.84)
    parser.add_argument("--vertex_seed_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--vertex_seed_min_component_size", type=int, default=1)
    parser.add_argument("--vertex_seed_target_count", type=int, default=0)
    parser.add_argument("--vertex_seed_fill_small_by", choices=["none", "peak", "size_then_peak"], default="none")
    parser.add_argument("--vertex_seed_position_mode", choices=["peak", "mean", "weighted"], default="weighted")
    parser.add_argument(
        "--vertex_seed_core_mode",
        choices=["closest4", "peak", "position_nearest", "blob", "none"],
        default="none",
    )

    parser.add_argument("--merge_fake_boundaries_f_threshold", type=float, default=-1.0)
    parser.add_argument("--face_corner_policy", choices=["top3", "exact3", "top4_split"], default="top3")
    parser.add_argument("--min_corner_voxels", type=int, default=1)
    parser.add_argument("--top4_split_ratio", type=float, default=0.85)
    parser.add_argument("--min_face_area", type=float, default=1e-12)
    parser.add_argument("--no_orient_outward", action="store_true")
    parser.add_argument(
        "--output_axes",
        choices=["voxel", "input"],
        default="voxel",
        help="Write coordinates in voxel field axes or rotate to input.obj axes with (x, z, -y).",
    )

    parser.add_argument("--out_obj", type=str, default=None)
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--out_npz", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_debug_ply", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)

    vertices, edges, faces, stats = extract_primal_dual_mesh(coords, features, resolution, args)
    output_vertices = transform_to_output_axes(vertices, args.output_axes)
    output_stats = stats_for_output_axes(stats, args.output_axes)

    stem = default_output_stem(path)
    out_dir = default_results_dir(path)
    out_obj = Path(args.out_obj) if args.out_obj else out_dir / f"{stem}_primal_dual_watershed.obj"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_primal_dual_watershed.ply"
    out_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{stem}_primal_dual_watershed.npz"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"{stem}_primal_dual_watershed_faces.csv"
    out_debug_ply = (
        Path(args.out_debug_ply)
        if args.out_debug_ply
        else out_dir / f"{stem}_primal_dual_watershed_debug_voxels.ply"
    )

    write_obj(out_obj, output_vertices, faces)
    write_mesh_ply(out_ply, output_vertices, edges, faces)
    write_npz(out_npz, output_vertices, edges, faces, output_stats)
    write_csv(out_csv, faces, output_stats)
    write_debug_ply(out_debug_ply, coords, resolution, output_stats, args.output_axes)

    print_summary(output_stats)
    print(f"Saved {out_obj}")
    print(f"Saved {out_ply}")
    print(f"Saved {out_npz}")
    print(f"Saved {out_csv}")
    print(f"Saved {out_debug_ply}")


if __name__ == "__main__":
    main()
