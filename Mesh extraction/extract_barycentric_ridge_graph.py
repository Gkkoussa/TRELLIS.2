"""
Barycentric Ridge Graph (BRG) extraction.

BRG builds a graph from only the two triangle-field scalar channels:
  d_tri  = 3 * min(a, b, c)
  d_vert = (max(a, b, c) - 1/3) / (2/3)

The algorithm:
  1. Extract vertex blobs from high d_vert.
  2. Extract edge-ridge voxels from low d_tri.
  3. Collapse each vertex blob to a tiny core and clear only that core.
  4. Connected-component the remaining edge ridges.
  5. Merge split ridge components only when a midpoint blob touches exactly two components.
  6. Attach each ridge component to nearby vertex blobs.

No offset channels or source mesh faces are used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from extract_vertices_threshold import (
    connected_components,
    default_output_stem,
    default_results_dir,
    infer_resolution,
    load_triangle_field,
    threshold_to_raw_max_barycentric,
    voxel_centers,
)


ALGO_NAME = "Barycentric Ridge Graph"
ALGO_SHORT_NAME = "BRG"


def make_chebyshev_offsets(radius: int, include_origin: bool = False) -> list[tuple[int, int, int]]:
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if not include_origin and dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def build_coord_to_index(coords: np.ndarray) -> dict[tuple[int, int, int], int]:
    return {tuple(c.tolist()): int(i) for i, c in enumerate(coords)}


def dilate_sparse_mask(
    coords: np.ndarray,
    source_mask: np.ndarray,
    coord_to_index: dict[tuple[int, int, int], int],
    radius: int,
) -> np.ndarray:
    if radius == 0:
        return source_mask.copy()

    dilated = source_mask.copy()
    offsets = make_chebyshev_offsets(radius, include_origin=False)
    for idx in np.flatnonzero(source_mask):
        x, y, z = coords[int(idx)].tolist()
        for dx, dy, dz in offsets:
            neighbor = coord_to_index.get((x + dx, y + dy, z + dz))
            if neighbor is not None:
                dilated[neighbor] = True
    return dilated


def component_position(
    coords: np.ndarray,
    values: np.ndarray,
    component: np.ndarray,
    resolution: int,
    threshold: float,
    mode: str,
) -> np.ndarray:
    centers = voxel_centers(coords[component], resolution)
    if mode == "peak":
        peak = int(component[np.argmax(values[component])])
        return voxel_centers(coords[peak : peak + 1], resolution)[0]
    if mode == "mean":
        return centers.mean(axis=0).astype(np.float32, copy=False)
    if mode != "weighted":
        raise ValueError(f"Unsupported vertex position mode: {mode}")

    weights = values[component] - float(threshold)
    weights = np.maximum(weights, 0.0) + 1e-6
    return np.average(centers, axis=0, weights=weights).astype(np.float32, copy=False)


def component_core_indices(
    coords: np.ndarray,
    values: np.ndarray,
    component: np.ndarray,
    resolution: int,
    position: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return np.zeros((0,), dtype=np.int64)
    if mode == "peak":
        return np.asarray([int(component[np.argmax(values[component])])], dtype=np.int64)
    if mode == "position_nearest":
        centers = voxel_centers(coords[component], resolution)
        dist2 = np.sum((centers - position[None, :]) ** 2, axis=1)
        return np.asarray([int(component[int(np.argmin(dist2))])], dtype=np.int64)
    if mode == "closest4":
        centers = voxel_centers(coords[component], resolution)
        dist2 = np.sum((centers - position[None, :]) ** 2, axis=1)
        count = min(4, int(component.size))
        nearest = np.argsort(dist2, kind="stable")[:count]
        return component[nearest].astype(np.int64, copy=False)
    if mode == "blob":
        return component.astype(np.int64, copy=False)
    raise ValueError(f"Unsupported vertex core mode: {mode}")


def extract_vertex_blobs(
    coords: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    threshold: float,
    connectivity: int,
    min_component_size: int,
    position_mode: str,
    core_mode: str = "closest4",
) -> dict[str, np.ndarray]:
    selected = np.asarray(d_vert, dtype=np.float32).reshape(-1) >= float(threshold)
    components = connected_components(coords, selected, connectivity=connectivity)

    vertex_id_for_index = np.full(coords.shape[0], -1, dtype=np.int32)
    vertex_core_id_for_index = np.full(coords.shape[0], -1, dtype=np.int32)
    vertex_mask = np.zeros(coords.shape[0], dtype=bool)
    vertex_core_mask = np.zeros(coords.shape[0], dtype=bool)
    vertices = []
    core_indices = []
    core_counts = []
    peak_values = []
    sizes = []

    for component in components:
        if component.size < min_component_size:
            continue
        vertex_id = len(vertices)
        vertex_id_for_index[component] = vertex_id
        vertex_mask[component] = True
        position = component_position(coords, d_vert, component, resolution, threshold, position_mode)
        vertices.append(position)
        core_idx = component_core_indices(coords, d_vert, component, resolution, position, core_mode)
        if core_idx.size:
            vertex_core_mask[core_idx] = True
            vertex_core_id_for_index[core_idx] = vertex_id
            core_indices.append(int(core_idx[0]))
        else:
            core_indices.append(-1)
        core_counts.append(int(core_idx.size))
        peak_values.append(float(d_vert[component].max()))
        sizes.append(int(component.size))

    if vertices:
        vertex_array = np.stack(vertices).astype(np.float32, copy=False)
    else:
        vertex_array = np.zeros((0, 3), dtype=np.float32)

    return {
        "vertices": vertex_array,
        "vertex_peak_d_vert": np.asarray(peak_values, dtype=np.float32),
        "vertex_component_size": np.asarray(sizes, dtype=np.int32),
        "vertex_core_index": np.asarray(core_indices, dtype=np.int32),
        "vertex_core_count": np.asarray(core_counts, dtype=np.int32),
        "vertex_id_for_index": vertex_id_for_index,
        "vertex_core_id_for_index": vertex_core_id_for_index,
        "vertex_mask": vertex_mask,
        "vertex_core_mask": vertex_core_mask,
    }


def build_vertex_coord_to_id(
    coords: np.ndarray,
    vertex_id_for_index: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    out = {}
    for idx in np.flatnonzero(vertex_id_for_index >= 0):
        out[tuple(coords[int(idx)].tolist())] = int(vertex_id_for_index[int(idx)])
    return out


def component_vertex_contacts(
    coords: np.ndarray,
    component: np.ndarray,
    vertex_coord_to_id: dict[tuple[int, int, int], int],
    attach_radius: int,
) -> set[int]:
    contacts: set[int] = set()
    offsets = make_chebyshev_offsets(attach_radius, include_origin=True)
    for idx in component:
        x, y, z = coords[int(idx)].tolist()
        for dx, dy, dz in offsets:
            vertex_id = vertex_coord_to_id.get((x + dx, y + dy, z + dz))
            if vertex_id is not None:
                contacts.add(vertex_id)
    return contacts


def component_endpoint_indices(coords: np.ndarray, component: np.ndarray) -> tuple[int, int]:
    if component.size == 0:
        raise ValueError("Cannot find endpoints of an empty component")
    if component.size == 1:
        only = int(component[0])
        return only, only

    points = coords[component].astype(np.float32, copy=False)
    seed = points[0]
    first_local = int(np.argmax(np.sum((points - seed[None, :]) ** 2, axis=1)))
    first = points[first_local]
    second_local = int(np.argmax(np.sum((points - first[None, :]) ** 2, axis=1)))
    return int(component[first_local]), int(component[second_local])


def nearest_vertex_contact(
    coords: np.ndarray,
    endpoint_index: int,
    vertex_coord_to_id: dict[tuple[int, int, int], int],
    attach_radius: int,
) -> int | None:
    x, y, z = coords[int(endpoint_index)].tolist()
    best_id = None
    best_dist2 = None
    offsets = make_chebyshev_offsets(attach_radius, include_origin=True)
    for dx, dy, dz in offsets:
        vertex_id = vertex_coord_to_id.get((x + dx, y + dy, z + dz))
        if vertex_id is None:
            continue
        dist2 = dx * dx + dy * dy + dz * dz
        if best_dist2 is None or dist2 < best_dist2:
            best_id = int(vertex_id)
            best_dist2 = int(dist2)
    return best_id


def component_endpoint_contacts(
    coords: np.ndarray,
    component: np.ndarray,
    vertex_coord_to_id: dict[tuple[int, int, int], int],
    attach_radius: int,
) -> set[int]:
    start_idx, end_idx = component_endpoint_indices(coords, component)
    contacts = set()
    for endpoint_index in (start_idx, end_idx):
        vertex_id = nearest_vertex_contact(
            coords,
            endpoint_index,
            vertex_coord_to_id,
            attach_radius,
        )
        if vertex_id is not None:
            contacts.add(vertex_id)
    return contacts


def vertex_core_positions_voxel(
    coords: np.ndarray,
    vertex_core_id_for_index: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    positions = np.zeros((num_vertices, 3), dtype=np.float32)
    counts = np.zeros(num_vertices, dtype=np.int32)
    for idx in np.flatnonzero(vertex_core_id_for_index >= 0):
        vertex_id = int(vertex_core_id_for_index[int(idx)])
        positions[vertex_id] += coords[int(idx)].astype(np.float32, copy=False)
        counts[vertex_id] += 1
    valid = counts > 0
    positions[valid] /= counts[valid, None].astype(np.float32)
    return positions


def nearest_candidate_vertices(
    vertex_positions: np.ndarray,
    query_points: np.ndarray,
    k: int,
) -> np.ndarray:
    if vertex_positions.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(2, min(int(k), int(vertex_positions.shape[0])))
    candidates: set[int] = set()
    for query in query_points:
        dist2 = np.sum((vertex_positions - query[None, :]) ** 2, axis=1)
        nearest = np.argpartition(dist2, k - 1)[:k]
        candidates.update(int(i) for i in nearest.tolist())
    return np.asarray(sorted(candidates), dtype=np.int32)


def segment_fit_distance(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    projection_margin: float,
) -> float | None:
    ab = b - a
    length2 = float(np.dot(ab, ab))
    if length2 <= 1e-6:
        return None
    length = float(np.sqrt(length2))
    t = np.sum((points - a[None, :]) * ab[None, :], axis=1) / length2
    margin_t = float(projection_margin) / max(length, 1e-6)
    if float(t.min()) < -margin_t or float(t.max()) > 1.0 + margin_t:
        return None
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    dist = np.linalg.norm(points - closest, axis=1)
    return float(np.quantile(dist, 0.90))


def component_evidence_pair(
    coords: np.ndarray,
    component: np.ndarray,
    vertex_positions: np.ndarray,
    *,
    candidate_k: int,
    max_segment_distance: float,
    max_pair_distance: float,
    projection_margin: float,
) -> tuple[tuple[int, int], float] | None:
    if component.size == 0 or vertex_positions.shape[0] < 2:
        return None

    start_idx, end_idx = component_endpoint_indices(coords, component)
    points = coords[component].astype(np.float32, copy=False)
    query_points = np.stack(
        [
            points.mean(axis=0),
            coords[start_idx].astype(np.float32, copy=False),
            coords[end_idx].astype(np.float32, copy=False),
        ],
        axis=0,
    )
    candidates = nearest_candidate_vertices(vertex_positions, query_points, candidate_k)
    if candidates.size < 2:
        return None

    best_pair = None
    best_score = None
    for i, a_id in enumerate(candidates):
        a = vertex_positions[int(a_id)]
        for b_id in candidates[i + 1 :]:
            b = vertex_positions[int(b_id)]
            pair_distance = float(np.linalg.norm(a - b))
            if pair_distance > float(max_pair_distance):
                continue
            score = segment_fit_distance(points, a, b, projection_margin)
            if score is None or score > float(max_segment_distance):
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_pair = tuple(sorted((int(a_id), int(b_id))))

    if best_pair is None or best_score is None:
        return None
    return best_pair, float(best_score)


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


def nearby_edge_components_for_voxels(
    coords: np.ndarray,
    component: np.ndarray,
    edge_coord_to_component: dict[tuple[int, int, int], list[int]],
    radius: float,
) -> dict[int, int]:
    radius_i = int(np.ceil(float(radius)))
    max_dist2 = float(radius) * float(radius)
    nearby: dict[int, int] = {}
    offsets = make_chebyshev_offsets(radius_i, include_origin=True)

    for index in component:
        x, y, z = coords[int(index)].tolist()
        for dx, dy, dz in offsets:
            dist2 = dx * dx + dy * dy + dz * dz
            if float(dist2) > max_dist2:
                continue
            for component_id in edge_coord_to_component.get((x + dx, y + dy, z + dz), []):
                component_id = int(component_id)
                old_dist2 = nearby.get(component_id)
                if old_dist2 is None or dist2 < old_dist2:
                    nearby[component_id] = int(dist2)
    return nearby


def merge_edge_components(
    edge_components: list[np.ndarray],
    uf: UnionFind,
) -> list[np.ndarray]:
    root_to_components: dict[int, list[np.ndarray]] = {}
    for component_id, component in enumerate(edge_components):
        root = uf.find(component_id)
        root_to_components.setdefault(root, []).append(component)

    merged_components = []
    for parts in root_to_components.values():
        if len(parts) == 1:
            merged_components.append(parts[0])
        else:
            merged_components.append(np.concatenate(parts).astype(np.int64, copy=False))
    return merged_components


def bridge_edge_components_by_midpoint_blobs(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    edge_components: list[np.ndarray],
    vertex_clear_mask: np.ndarray,
    midpoint_dtri_threshold: float,
    midpoint_dvert: float,
    midpoint_tolerance: float,
    search_radius: float,
    connectivity: int,
    min_component_size: int,
    max_component_size: int,
) -> tuple[list[np.ndarray], dict[str, int | float | str]]:
    enabled = int(float(search_radius) > 0.0)
    empty_stats = {
        "edge_bridge_mode": "midpoint_blob",
        "edge_bridge_enabled": enabled,
        "edge_bridge_max_distance": float(search_radius),
        "edge_bridge_midpoint_dtri_threshold": float(midpoint_dtri_threshold),
        "edge_bridge_midpoint_dvert": float(midpoint_dvert),
        "edge_bridge_midpoint_tolerance": float(midpoint_tolerance),
        "edge_bridge_midpoint_min_component_size": int(min_component_size),
        "edge_bridge_midpoint_max_component_size": int(max_component_size),
        "edge_components_before_bridge": int(len(edge_components)),
        "edge_components_after_bridge": int(len(edge_components)),
        "edge_bridge_midpoint_voxels": 0,
        "edge_bridge_midpoint_components": 0,
        "edge_bridge_midpoint_components_used": 0,
        "edge_bridge_midpoint_components_no_edge": 0,
        "edge_bridge_midpoint_components_single_edge": 0,
        "edge_bridge_midpoint_components_ambiguous": 0,
        "edge_bridge_midpoint_components_too_small": 0,
        "edge_bridge_midpoint_components_too_large": 0,
        "edge_bridge_unions": 0,
    }
    if not edge_components or not enabled:
        return edge_components, empty_stats

    midpoint_mask = (
        (d_tri <= float(midpoint_dtri_threshold))
        & (np.abs(d_vert - float(midpoint_dvert)) <= float(midpoint_tolerance))
        & (~vertex_clear_mask)
    )
    midpoint_components = connected_components(coords, midpoint_mask, connectivity=connectivity)
    edge_coord_to_component: dict[tuple[int, int, int], list[int]] = {}
    for component_id, component in enumerate(edge_components):
        for index in component:
            key = tuple(coords[int(index)].tolist())
            edge_coord_to_component.setdefault(key, []).append(int(component_id))

    uf = UnionFind(len(edge_components))
    used = 0
    no_edge = 0
    single_edge = 0
    ambiguous = 0
    too_small = 0
    too_large = 0
    union_count = 0

    for midpoint_component in midpoint_components:
        size = int(midpoint_component.size)
        if size < int(min_component_size):
            too_small += 1
            continue
        if int(max_component_size) > 0 and size > int(max_component_size):
            too_large += 1
            continue

        nearby = nearby_edge_components_for_voxels(
            coords,
            midpoint_component,
            edge_coord_to_component,
            search_radius,
        )
        if len(nearby) == 0:
            no_edge += 1
            continue
        if len(nearby) == 1:
            single_edge += 1
            continue
        if len(nearby) > 2:
            ambiguous += 1
            continue

        left, right = sorted(nearby, key=lambda component_id: (nearby[component_id], component_id))
        used += 1
        if uf.union(left, right):
            union_count += 1

    merged_components = merge_edge_components(edge_components, uf)
    stats = dict(empty_stats)
    stats.update({
        "edge_components_after_bridge": int(len(merged_components)),
        "edge_bridge_midpoint_voxels": int(midpoint_mask.sum()),
        "edge_bridge_midpoint_components": int(len(midpoint_components)),
        "edge_bridge_midpoint_components_used": int(used),
        "edge_bridge_midpoint_components_no_edge": int(no_edge),
        "edge_bridge_midpoint_components_single_edge": int(single_edge),
        "edge_bridge_midpoint_components_ambiguous": int(ambiguous),
        "edge_bridge_midpoint_components_too_small": int(too_small),
        "edge_bridge_midpoint_components_too_large": int(too_large),
        "edge_bridge_unions": int(union_count),
    })
    return merged_components, stats


def edge_component_has_midpoint(
    d_vert: np.ndarray,
    component: np.ndarray,
    midpoint_d_vert: float,
    midpoint_tolerance: float,
) -> tuple[bool, int]:
    values = d_vert[component]
    midpoint_mask = np.abs(values - float(midpoint_d_vert)) <= float(midpoint_tolerance)
    return bool(midpoint_mask.any()), int(midpoint_mask.sum())


def extract_brg(
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
    edge_evidence_candidate_vertices: int = 8,
    edge_evidence_max_segment_distance: float = 3.0,
    edge_evidence_max_pair_distance: float = 64.0,
    edge_evidence_projection_margin: float = 4.0,
    edge_bridge_max_distance: float = 4.0,
    edge_bridge_midpoint_dtri_threshold: float | None = None,
    edge_bridge_midpoint_tolerance: float | None = None,
    edge_bridge_midpoint_connectivity: int = 18,
    edge_bridge_midpoint_min_component_size: int = 1,
    edge_bridge_midpoint_max_component_size: int = 128,
    vertex_clearance: int = 1,
    attach_radius: int = 3,
    midpoint_dvert: float = 0.25,
    midpoint_tolerance: float = 0.075,
    require_midpoint: bool = True,
) -> dict[str, np.ndarray | int | float]:
    d_tri = np.asarray(d_tri, dtype=np.float32).reshape(-1)
    d_vert = np.asarray(d_vert, dtype=np.float32).reshape(-1)
    if edge_max_dvert is None:
        edge_max_dvert = vertex_dvert_threshold
    if edge_bridge_midpoint_dtri_threshold is None:
        edge_bridge_midpoint_dtri_threshold = edge_dtri_threshold
    if edge_bridge_midpoint_tolerance is None:
        edge_bridge_midpoint_tolerance = midpoint_tolerance

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

    vertex_clear_mask = dilate_sparse_mask(
        coords,
        vertex_data["vertex_core_mask"],
        coord_to_index,
        radius=vertex_clearance,
    )
    edge_mask = (
        (d_tri <= float(edge_dtri_threshold))
        & (d_vert >= float(edge_min_dvert))
        & (d_vert <= float(edge_max_dvert))
        & (~vertex_clear_mask)
    )

    edge_components = connected_components(coords, edge_mask, connectivity=edge_connectivity)
    edge_components, edge_bridge_stats = bridge_edge_components_by_midpoint_blobs(
        coords,
        d_tri,
        d_vert,
        edge_components,
        vertex_clear_mask,
        midpoint_dtri_threshold=edge_bridge_midpoint_dtri_threshold,
        midpoint_dvert=midpoint_dvert,
        midpoint_tolerance=edge_bridge_midpoint_tolerance,
        search_radius=edge_bridge_max_distance,
        connectivity=edge_bridge_midpoint_connectivity,
        min_component_size=edge_bridge_midpoint_min_component_size,
        max_component_size=edge_bridge_midpoint_max_component_size,
    )
    vertex_coord_to_id = build_vertex_coord_to_id(coords, vertex_data["vertex_core_id_for_index"])
    vertex_positions_voxel = vertex_core_positions_voxel(
        coords,
        vertex_data["vertex_core_id_for_index"],
        int(vertex_data["vertices"].shape[0]),
    )

    edge_by_pair: dict[tuple[int, int], dict[str, int]] = {}
    rejected_small = 0
    rejected_no_midpoint = 0
    rejected_bad_contacts = 0
    rejected_bad_evidence_fit = 0
    duplicate_pairs = 0
    evidence_voted_components = 0

    for component_id, component in enumerate(edge_components):
        if component.size < edge_min_component_size:
            rejected_small += 1
            continue

        has_midpoint, midpoint_count = edge_component_has_midpoint(
            d_vert,
            component,
            midpoint_dvert,
            midpoint_tolerance,
        )
        if edge_attach_mode != "evidence" and require_midpoint and not has_midpoint:
            rejected_no_midpoint += 1
            continue

        if edge_attach_mode == "contacts":
            contacts = component_vertex_contacts(coords, component, vertex_coord_to_id, attach_radius)
            if len(contacts) != 2:
                rejected_bad_contacts += 1
                continue
            pair = tuple(sorted(contacts))
        elif edge_attach_mode == "endpoints":
            contacts = component_endpoint_contacts(coords, component, vertex_coord_to_id, attach_radius)
            if len(contacts) != 2:
                rejected_bad_contacts += 1
                continue
            pair = tuple(sorted(contacts))
        elif edge_attach_mode == "evidence":
            evidence = component_evidence_pair(
                coords,
                component,
                vertex_positions_voxel,
                candidate_k=edge_evidence_candidate_vertices,
                max_segment_distance=edge_evidence_max_segment_distance,
                max_pair_distance=edge_evidence_max_pair_distance,
                projection_margin=edge_evidence_projection_margin,
            )
            if evidence is None:
                rejected_bad_evidence_fit += 1
                continue
            pair, _ = evidence
            evidence_voted_components += 1
        else:
            raise ValueError(f"Unsupported edge attach mode: {edge_attach_mode}")

        record = {
            "component_id": int(component_id),
            "size": int(component.size),
            "midpoint_voxels": int(midpoint_count),
            "evidence_components": 1,
        }
        old = edge_by_pair.get(pair)
        if old is None:
            edge_by_pair[pair] = record
        else:
            duplicate_pairs += 1
            old["size"] += record["size"]
            old["midpoint_voxels"] += record["midpoint_voxels"]
            old["evidence_components"] += 1

    pairs = sorted(edge_by_pair)
    edges = np.asarray(pairs, dtype=np.int32).reshape(-1, 2) if pairs else np.zeros((0, 2), dtype=np.int32)
    edge_component_ids = np.asarray([edge_by_pair[p]["component_id"] for p in pairs], dtype=np.int32)
    edge_component_sizes = np.asarray([edge_by_pair[p]["size"] for p in pairs], dtype=np.int32)
    edge_midpoint_voxels = np.asarray([edge_by_pair[p]["midpoint_voxels"] for p in pairs], dtype=np.int32)
    edge_evidence_components = np.asarray(
        [edge_by_pair[p]["evidence_components"] for p in pairs],
        dtype=np.int32,
    )

    return {
        "vertices": vertex_data["vertices"],
        "edges": edges,
        "vertex_peak_d_vert": vertex_data["vertex_peak_d_vert"],
        "vertex_component_size": vertex_data["vertex_component_size"],
        "vertex_core_index": vertex_data["vertex_core_index"],
        "vertex_core_count": vertex_data["vertex_core_count"],
        "edge_component_id": edge_component_ids,
        "edge_component_size": edge_component_sizes,
        "edge_midpoint_voxels": edge_midpoint_voxels,
        "edge_evidence_components": edge_evidence_components,
        "num_vertex_blob_voxels": int(vertex_data["vertex_mask"].sum()),
        "num_vertex_core_voxels": int(vertex_data["vertex_core_mask"].sum()),
        "num_vertex_clearance_voxels": int(vertex_clear_mask.sum()),
        "num_edge_mask_voxels": int(edge_mask.sum()),
        "num_edge_components": int(len(edge_components)),
        "num_rejected_small_components": int(rejected_small),
        "num_rejected_no_midpoint_components": int(rejected_no_midpoint),
        "num_rejected_bad_contact_components": int(rejected_bad_contacts),
        "num_rejected_bad_evidence_fit_components": int(rejected_bad_evidence_fit),
        "num_duplicate_edge_pairs": int(duplicate_pairs),
        "num_evidence_voted_components": int(evidence_voted_components),
        "vertex_dvert_threshold": float(vertex_dvert_threshold),
        "vertex_raw_max_barycentric": float(threshold_to_raw_max_barycentric(vertex_dvert_threshold)),
        "vertex_core_mode": str(vertex_core_mode),
        "edge_dtri_threshold": float(edge_dtri_threshold),
        "edge_min_dvert": float(edge_min_dvert),
        "edge_max_dvert": float(edge_max_dvert),
        "edge_attach_mode": str(edge_attach_mode),
        "edge_evidence_candidate_vertices": int(edge_evidence_candidate_vertices),
        "edge_evidence_max_segment_distance": float(edge_evidence_max_segment_distance),
        "edge_evidence_max_pair_distance": float(edge_evidence_max_pair_distance),
        "edge_evidence_projection_margin": float(edge_evidence_projection_margin),
        "edge_bridge_mode": str(edge_bridge_stats["edge_bridge_mode"]),
        "edge_bridge_max_distance": float(edge_bridge_max_distance),
        "edge_bridge_enabled": int(edge_bridge_stats["edge_bridge_enabled"]),
        "edge_bridge_midpoint_dtri_threshold": float(edge_bridge_stats["edge_bridge_midpoint_dtri_threshold"]),
        "edge_bridge_midpoint_dvert": float(edge_bridge_stats["edge_bridge_midpoint_dvert"]),
        "edge_bridge_midpoint_tolerance": float(edge_bridge_stats["edge_bridge_midpoint_tolerance"]),
        "edge_bridge_midpoint_min_component_size": int(edge_bridge_stats["edge_bridge_midpoint_min_component_size"]),
        "edge_bridge_midpoint_max_component_size": int(edge_bridge_stats["edge_bridge_midpoint_max_component_size"]),
        "edge_components_before_bridge": int(edge_bridge_stats["edge_components_before_bridge"]),
        "edge_components_after_bridge": int(edge_bridge_stats["edge_components_after_bridge"]),
        "edge_bridge_midpoint_voxels": int(edge_bridge_stats["edge_bridge_midpoint_voxels"]),
        "edge_bridge_midpoint_components": int(edge_bridge_stats["edge_bridge_midpoint_components"]),
        "edge_bridge_midpoint_components_used": int(edge_bridge_stats["edge_bridge_midpoint_components_used"]),
        "edge_bridge_midpoint_components_no_edge": int(edge_bridge_stats["edge_bridge_midpoint_components_no_edge"]),
        "edge_bridge_midpoint_components_single_edge": int(edge_bridge_stats["edge_bridge_midpoint_components_single_edge"]),
        "edge_bridge_midpoint_components_ambiguous": int(edge_bridge_stats["edge_bridge_midpoint_components_ambiguous"]),
        "edge_bridge_midpoint_components_too_small": int(edge_bridge_stats["edge_bridge_midpoint_components_too_small"]),
        "edge_bridge_midpoint_components_too_large": int(edge_bridge_stats["edge_bridge_midpoint_components_too_large"]),
        "edge_bridge_unions": int(edge_bridge_stats["edge_bridge_unions"]),
        "edge_raw_min_max_barycentric": float(threshold_to_raw_max_barycentric(edge_min_dvert)),
        "edge_raw_max_max_barycentric": float(threshold_to_raw_max_barycentric(edge_max_dvert)),
        "vertex_clearance": int(vertex_clearance),
        "attach_radius": int(attach_radius),
        "midpoint_dvert": float(midpoint_dvert),
        "midpoint_tolerance": float(midpoint_tolerance),
    }


def sweep_edge_dtri_thresholds(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    thresholds: list[float],
    **kwargs,
) -> list[dict[str, int | float]]:
    rows = []
    for threshold in thresholds:
        graph = extract_brg(
            coords,
            d_tri,
            d_vert,
            resolution,
            edge_dtri_threshold=threshold,
            **kwargs,
        )
        rows.append({
            "edge_dtri_threshold": float(threshold),
            "vertices": int(graph["vertices"].shape[0]),
            "edge_mask_voxels": int(graph["num_edge_mask_voxels"]),
            "edge_components": int(graph["num_edge_components"]),
            "accepted_edges": int(graph["edges"].shape[0]),
            "rejected_small": int(graph["num_rejected_small_components"]),
            "rejected_no_midpoint": int(graph["num_rejected_no_midpoint_components"]),
            "rejected_bad_contacts": int(graph["num_rejected_bad_contact_components"]),
            "rejected_bad_evidence_fit": int(graph["num_rejected_bad_evidence_fit_components"]),
            "duplicate_edge_pairs": int(graph["num_duplicate_edge_pairs"]),
        })
    return rows


def write_sweep_csv(path: str | Path, rows: list[dict[str, int | float]]) -> None:
    path = Path(path)
    fieldnames = [
        "edge_dtri_threshold",
        "vertices",
        "edge_mask_voxels",
        "edge_components",
        "accepted_edges",
        "rejected_small",
        "rejected_no_midpoint",
        "rejected_bad_contacts",
        "rejected_bad_evidence_fit",
        "duplicate_edge_pairs",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_graph_npz(path: str | Path, graph: dict[str, np.ndarray | int | float], resolution: int) -> None:
    np.savez_compressed(
        path,
        vertices=graph["vertices"],
        edges=graph["edges"],
        vertex_peak_d_vert=graph["vertex_peak_d_vert"],
        vertex_component_size=graph["vertex_component_size"],
        vertex_core_index=graph["vertex_core_index"],
        vertex_core_count=graph["vertex_core_count"],
        edge_component_id=graph["edge_component_id"],
        edge_component_size=graph["edge_component_size"],
        edge_midpoint_voxels=graph["edge_midpoint_voxels"],
        edge_evidence_components=graph["edge_evidence_components"],
        resolution=np.asarray(resolution, dtype=np.int32),
        algorithm=np.asarray(ALGO_NAME),
        vertex_dvert_threshold=np.asarray(graph["vertex_dvert_threshold"], dtype=np.float32),
        vertex_raw_max_barycentric=np.asarray(graph["vertex_raw_max_barycentric"], dtype=np.float32),
        vertex_core_mode=np.asarray(graph["vertex_core_mode"]),
        vertex_blob_voxels=np.asarray(graph["num_vertex_blob_voxels"], dtype=np.int32),
        vertex_core_voxels=np.asarray(graph["num_vertex_core_voxels"], dtype=np.int32),
        vertex_clearance_voxels=np.asarray(graph["num_vertex_clearance_voxels"], dtype=np.int32),
        edge_dtri_threshold=np.asarray(graph["edge_dtri_threshold"], dtype=np.float32),
        edge_min_dvert=np.asarray(graph["edge_min_dvert"], dtype=np.float32),
        edge_max_dvert=np.asarray(graph["edge_max_dvert"], dtype=np.float32),
        edge_attach_mode=np.asarray(graph["edge_attach_mode"]),
        edge_evidence_candidate_vertices=np.asarray(graph["edge_evidence_candidate_vertices"], dtype=np.int32),
        edge_evidence_max_segment_distance=np.asarray(graph["edge_evidence_max_segment_distance"], dtype=np.float32),
        edge_evidence_max_pair_distance=np.asarray(graph["edge_evidence_max_pair_distance"], dtype=np.float32),
        edge_evidence_projection_margin=np.asarray(graph["edge_evidence_projection_margin"], dtype=np.float32),
        edge_bridge_mode=np.asarray(graph["edge_bridge_mode"]),
        edge_bridge_max_distance=np.asarray(graph["edge_bridge_max_distance"], dtype=np.float32),
        edge_bridge_enabled=np.asarray(graph["edge_bridge_enabled"], dtype=np.int32),
        edge_bridge_midpoint_dtri_threshold=np.asarray(
            graph["edge_bridge_midpoint_dtri_threshold"],
            dtype=np.float32,
        ),
        edge_bridge_midpoint_dvert=np.asarray(graph["edge_bridge_midpoint_dvert"], dtype=np.float32),
        edge_bridge_midpoint_tolerance=np.asarray(graph["edge_bridge_midpoint_tolerance"], dtype=np.float32),
        edge_bridge_midpoint_min_component_size=np.asarray(
            graph["edge_bridge_midpoint_min_component_size"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_max_component_size=np.asarray(
            graph["edge_bridge_midpoint_max_component_size"],
            dtype=np.int32,
        ),
        edge_components_before_bridge=np.asarray(graph["edge_components_before_bridge"], dtype=np.int32),
        edge_components_after_bridge=np.asarray(graph["edge_components_after_bridge"], dtype=np.int32),
        edge_bridge_midpoint_voxels=np.asarray(graph["edge_bridge_midpoint_voxels"], dtype=np.int32),
        edge_bridge_midpoint_components=np.asarray(graph["edge_bridge_midpoint_components"], dtype=np.int32),
        edge_bridge_midpoint_components_used=np.asarray(
            graph["edge_bridge_midpoint_components_used"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_no_edge=np.asarray(
            graph["edge_bridge_midpoint_components_no_edge"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_single_edge=np.asarray(
            graph["edge_bridge_midpoint_components_single_edge"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_ambiguous=np.asarray(
            graph["edge_bridge_midpoint_components_ambiguous"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_too_small=np.asarray(
            graph["edge_bridge_midpoint_components_too_small"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_too_large=np.asarray(
            graph["edge_bridge_midpoint_components_too_large"],
            dtype=np.int32,
        ),
        edge_bridge_unions=np.asarray(graph["edge_bridge_unions"], dtype=np.int32),
        evidence_voted_components=np.asarray(graph["num_evidence_voted_components"], dtype=np.int32),
        rejected_bad_evidence_fit_components=np.asarray(
            graph["num_rejected_bad_evidence_fit_components"],
            dtype=np.int32,
        ),
        edge_raw_min_max_barycentric=np.asarray(graph["edge_raw_min_max_barycentric"], dtype=np.float32),
        edge_raw_max_max_barycentric=np.asarray(graph["edge_raw_max_max_barycentric"], dtype=np.float32),
        vertex_clearance=np.asarray(graph["vertex_clearance"], dtype=np.int32),
        attach_radius=np.asarray(graph["attach_radius"], dtype=np.int32),
        midpoint_dvert=np.asarray(graph["midpoint_dvert"], dtype=np.float32),
        midpoint_tolerance=np.asarray(graph["midpoint_tolerance"], dtype=np.float32),
    )


def write_graph_ply(path: str | Path, graph: dict[str, np.ndarray | int | float]) -> None:
    vertices = graph["vertices"]
    edges = graph["edges"]
    vertex_peak = graph["vertex_peak_d_vert"]
    vertex_sizes = graph["vertex_component_size"]
    vertex_core_counts = graph["vertex_core_count"]
    edge_sizes = graph["edge_component_size"]
    edge_midpoints = graph["edge_midpoint_voxels"]
    edge_evidence_counts = graph["edge_evidence_components"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment algorithm {ALGO_NAME}\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float peak_d_vert\n")
        f.write("property int component_size\n")
        f.write("property int core_voxels\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property int component_size\n")
        f.write("property int midpoint_voxels\n")
        f.write("property int evidence_components\n")
        f.write("end_header\n")
        for vertex, peak, size, core_count in zip(vertices, vertex_peak, vertex_sizes, vertex_core_counts):
            f.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{float(peak):.8f} {int(size)} {int(core_count)}\n"
            )
        for edge, size, midpoint_count, evidence_count in zip(
            edges,
            edge_sizes,
            edge_midpoints,
            edge_evidence_counts,
        ):
            f.write(
                f"{int(edge[0])} {int(edge[1])} {int(size)} "
                f"{int(midpoint_count)} {int(evidence_count)}\n"
            )


def parse_float_list(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("Expected at least one float")
    return out


def print_graph_summary(graph: dict[str, np.ndarray | int | float], resolution: int) -> None:
    print(f"Algorithm: {ALGO_NAME} ({ALGO_SHORT_NAME})")
    print(f"Resolution: {resolution}")
    print(f"Vertices: {graph['vertices'].shape[0]}")
    print(f"Edges: {graph['edges'].shape[0]}")
    print(f"Vertex blob voxels: {graph['num_vertex_blob_voxels']}")
    print(f"Vertex core voxels: {graph['num_vertex_core_voxels']}")
    print(f"Vertex clearance voxels: {graph['num_vertex_clearance_voxels']}")
    print(f"Edge mask voxels: {graph['num_edge_mask_voxels']}")
    print(f"Edge components before filtering: {graph['num_edge_components']}")
    print(f"Rejected small components: {graph['num_rejected_small_components']}")
    print(f"Rejected no-midpoint components: {graph['num_rejected_no_midpoint_components']}")
    print(f"Rejected bad-contact components: {graph['num_rejected_bad_contact_components']}")
    print(f"Rejected bad-evidence-fit components: {graph['num_rejected_bad_evidence_fit_components']}")
    print(f"Evidence voted components: {graph['num_evidence_voted_components']}")
    print(f"Duplicate edge pairs collapsed: {graph['num_duplicate_edge_pairs']}")
    print("Thresholds:")
    print(
        f"  vertex d_vert >= {graph['vertex_dvert_threshold']} "
        f"(raw max barycentric >= {graph['vertex_raw_max_barycentric']:.6f})"
    )
    print(f"  vertex core mode: {graph['vertex_core_mode']}")
    print(f"  edge d_tri <= {graph['edge_dtri_threshold']}")
    print(
        f"  edge d_vert in [{graph['edge_min_dvert']}, {graph['edge_max_dvert']}] "
        f"(raw max barycentric in "
        f"[{graph['edge_raw_min_max_barycentric']:.6f}, "
        f"{graph['edge_raw_max_max_barycentric']:.6f}])"
    )
    print(f"  edge attach mode: {graph['edge_attach_mode']}")
    if graph["edge_attach_mode"] == "evidence":
        print(f"  evidence candidate vertices: {graph['edge_evidence_candidate_vertices']}")
        print(f"  evidence max segment distance: {graph['edge_evidence_max_segment_distance']} voxel(s)")
        print(f"  evidence max pair distance: {graph['edge_evidence_max_pair_distance']} voxel(s)")
        print(f"  evidence projection margin: {graph['edge_evidence_projection_margin']} voxel(s)")
    print(f"  edge bridge mode: {graph['edge_bridge_mode']}")
    print(f"  edge bridge search radius: {graph['edge_bridge_max_distance']} voxel(s)")
    print(f"  edge bridge midpoint d_tri <= {graph['edge_bridge_midpoint_dtri_threshold']}")
    print(
        f"  edge bridge midpoint abs(d_vert - {graph['edge_bridge_midpoint_dvert']}) "
        f"<= {graph['edge_bridge_midpoint_tolerance']}"
    )
    print(f"  edge bridge midpoint voxels: {graph['edge_bridge_midpoint_voxels']}")
    print(f"  edge bridge midpoint components: {graph['edge_bridge_midpoint_components']}")
    print(f"  edge bridge midpoint components used: {graph['edge_bridge_midpoint_components_used']}")
    print(f"  edge bridge midpoint components single-edge: {graph['edge_bridge_midpoint_components_single_edge']}")
    print(f"  edge bridge midpoint components ambiguous: {graph['edge_bridge_midpoint_components_ambiguous']}")
    print(f"  edge components before bridge: {graph['edge_components_before_bridge']}")
    print(f"  edge components after bridge: {graph['edge_components_after_bridge']}")
    print(f"  edge bridge unions: {graph['edge_bridge_unions']}")
    print(
        f"  midpoint check abs(d_vert - {graph['midpoint_dvert']}) "
        f"<= {graph['midpoint_tolerance']}"
    )
    print(f"  vertex clearance radius from core: {graph['vertex_clearance']} sparse voxel step(s)")
    print(f"  attach radius: {graph['attach_radius']} sparse voxel step(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{ALGO_NAME}: extract vertices and graph edges from d_tri/d_vert only."
    )
    parser.add_argument("path", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--vertex_dvert_threshold", type=float, default=0.84)
    parser.add_argument("--vertex_connectivity", type=int, default=18, choices=[6, 18, 26])
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
    parser.add_argument("--edge_connectivity", type=int, default=18, choices=[6, 18, 26])
    parser.add_argument("--edge_min_component_size", type=int, default=2)
    parser.add_argument("--edge_attach_mode", choices=["evidence", "endpoints", "contacts"], default="evidence")
    parser.add_argument("--edge_evidence_candidate_vertices", type=int, default=8)
    parser.add_argument("--edge_evidence_max_segment_distance", type=float, default=3.0)
    parser.add_argument("--edge_evidence_max_pair_distance", type=float, default=64.0)
    parser.add_argument("--edge_evidence_projection_margin", type=float, default=4.0)
    parser.add_argument("--edge_bridge_max_distance", type=float, default=4.0)
    parser.add_argument("--edge_bridge_midpoint_dtri_threshold", type=float, default=None)
    parser.add_argument("--edge_bridge_midpoint_tolerance", type=float, default=None)
    parser.add_argument("--edge_bridge_midpoint_connectivity", type=int, default=18, choices=[6, 18, 26])
    parser.add_argument("--edge_bridge_midpoint_min_component_size", type=int, default=1)
    parser.add_argument("--edge_bridge_midpoint_max_component_size", type=int, default=128)
    parser.add_argument("--vertex_clearance", type=int, default=1)
    parser.add_argument("--attach_radius", type=int, default=3)
    parser.add_argument("--midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--midpoint_tolerance", type=float, default=0.075)
    parser.add_argument("--no_midpoint_check", action="store_true")
    parser.add_argument(
        "--sweep_edge_dtri",
        type=str,
        default=None,
        help="Comma-separated d_tri thresholds to sweep instead of writing a graph",
    )
    parser.add_argument("--sweep_out_csv", type=str, default=None)
    parser.add_argument("--out_npz", type=str, default=None)
    parser.add_argument("--out_ply", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)
    d_tri = features[:, 0]
    d_vert = features[:, 1]

    common_kwargs = {
        "vertex_dvert_threshold": args.vertex_dvert_threshold,
        "vertex_connectivity": args.vertex_connectivity,
        "vertex_min_component_size": args.vertex_min_component_size,
        "vertex_position_mode": args.vertex_position_mode,
        "vertex_core_mode": args.vertex_core_mode,
        "edge_min_dvert": args.edge_min_dvert,
        "edge_max_dvert": args.edge_max_dvert,
        "edge_connectivity": args.edge_connectivity,
        "edge_min_component_size": args.edge_min_component_size,
        "edge_attach_mode": args.edge_attach_mode,
        "edge_evidence_candidate_vertices": args.edge_evidence_candidate_vertices,
        "edge_evidence_max_segment_distance": args.edge_evidence_max_segment_distance,
        "edge_evidence_max_pair_distance": args.edge_evidence_max_pair_distance,
        "edge_evidence_projection_margin": args.edge_evidence_projection_margin,
        "edge_bridge_max_distance": args.edge_bridge_max_distance,
        "edge_bridge_midpoint_dtri_threshold": args.edge_bridge_midpoint_dtri_threshold,
        "edge_bridge_midpoint_tolerance": args.edge_bridge_midpoint_tolerance,
        "edge_bridge_midpoint_connectivity": args.edge_bridge_midpoint_connectivity,
        "edge_bridge_midpoint_min_component_size": args.edge_bridge_midpoint_min_component_size,
        "edge_bridge_midpoint_max_component_size": args.edge_bridge_midpoint_max_component_size,
        "vertex_clearance": args.vertex_clearance,
        "attach_radius": args.attach_radius,
        "midpoint_dvert": args.midpoint_dvert,
        "midpoint_tolerance": args.midpoint_tolerance,
        "require_midpoint": not args.no_midpoint_check,
    }

    stem = default_output_stem(path)
    if args.sweep_edge_dtri is not None:
        rows = sweep_edge_dtri_thresholds(
            coords,
            d_tri,
            d_vert,
            resolution,
            parse_float_list(args.sweep_edge_dtri),
            **common_kwargs,
        )
        out_dir = default_results_dir(path)
        out_csv = (
            Path(args.sweep_out_csv)
            if args.sweep_out_csv
            else out_dir / f"{stem}_brg_edge_dtri_sweep.csv"
        )
        write_sweep_csv(out_csv, rows)
        print(f"Saved sweep CSV {out_csv}")
        print("edge_dtri_threshold, vertices, accepted_edges, edge_mask_voxels, edge_components")
        for row in rows:
            print(
                f"{row['edge_dtri_threshold']:.6f}, "
                f"{row['vertices']}, "
                f"{row['accepted_edges']}, "
                f"{row['edge_mask_voxels']}, "
                f"{row['edge_components']}"
            )
        return

    graph = extract_brg(
        coords,
        d_tri,
        d_vert,
        resolution,
        edge_dtri_threshold=args.edge_dtri_threshold,
        **common_kwargs,
    )
    out_dir = default_results_dir(path)
    out_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{stem}_brg_graph.npz"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_brg_graph.ply"
    write_graph_npz(out_npz, graph, resolution)
    write_graph_ply(out_ply, graph)
    print_graph_summary(graph, resolution)
    print(f"Saved {out_npz}")
    print(f"Saved {out_ply}")


if __name__ == "__main__":
    main()
