"""
Build a mesh by assigning high-d_tri face-interior voxel blobs to vertex triples.

This is separate from the BRG edge-cycle mesh. It uses the fact that active
triangle-field voxels can carry direct face evidence:
  - high d_vert finds vertices
  - high d_tri and low/mid d_vert finds triangle interiors
  - each face blob votes for the best nearby vertex triple

Accepted face triples add their three mesh edges automatically, so a broken
edge ridge does not by itself delete the face.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

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


ALGO_NAME = "Barycentric Face Support Mesh"


def sorted_edge(edge: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    if a == b:
        raise ValueError("self-edge is not allowed")
    return (a, b) if a < b else (b, a)


def edge_array_from_faces(faces: np.ndarray) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        edge_set.add(sorted_edge((a, b)))
        edge_set.add(sorted_edge((b, c)))
        edge_set.add(sorted_edge((c, a)))
    if not edge_set:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(edge_set), dtype=np.int32)


def face_blob_mask(
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    *,
    dtri_grow_threshold: float,
    max_dvert: float,
    vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = (d_tri >= float(dtri_grow_threshold)) & (d_vert <= float(max_dvert))
    if vertex_mask is not None:
        mask = mask & (~vertex_mask)
    return mask


def seed_mask(
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    *,
    dtri_seed_threshold: float,
    max_dvert: float,
    vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = (d_tri >= float(dtri_seed_threshold)) & (d_vert <= float(max_dvert))
    if vertex_mask is not None:
        mask = mask & (~vertex_mask)
    return mask


def sample_component_indices(
    component: np.ndarray,
    d_tri: np.ndarray,
    max_points: int,
) -> np.ndarray:
    if max_points <= 0 or component.size <= max_points:
        return component.astype(np.int64, copy=False)

    # Mix strong face-center voxels with spatially spread voxels. Pure top-k
    # points over-focuses on the center and makes large faces harder to assign.
    strong_count = max(1, max_points // 2)
    spread_count = max_points - strong_count

    strong_order = np.argsort(d_tri[component], kind="stable")[::-1][:strong_count]
    strong = component[strong_order]

    spread_order = np.argsort(component, kind="stable")
    spread_positions = np.linspace(0, component.size - 1, spread_count, dtype=np.int64)
    spread = component[spread_order[spread_positions]]

    sampled = np.unique(np.concatenate([strong, spread])).astype(np.int64, copy=False)
    if sampled.size > max_points:
        sampled = sampled[:max_points]
    return sampled


def barycentric_coordinates(points: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, float] | None:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = points - a[None, :]
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = np.einsum("ij,j->i", v2, v0)
    d21 = np.einsum("ij,j->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1)
    area = 0.5 * float(np.linalg.norm(np.cross(v0, v1)))
    return bary.astype(np.float32, copy=False), area


def score_vertex_triple(
    points: np.ndarray,
    weights: np.ndarray,
    triangle: np.ndarray,
    resolution: int,
    *,
    barycentric_margin: float,
    min_area: float,
) -> dict[str, float] | None:
    a, b, c = triangle
    normal = np.cross(b - a, c - a)
    normal_norm = float(np.linalg.norm(normal))
    area = 0.5 * normal_norm
    if area < float(min_area) or normal_norm < 1e-12:
        return None

    edge_lengths_vox = np.asarray([
        np.linalg.norm(b - a),
        np.linalg.norm(c - b),
        np.linalg.norm(a - c),
    ], dtype=np.float32) * float(resolution)
    vertex_diffs = triangle[:, None, :] - points[None, :, :]
    vertex_dist_vox = np.sqrt(np.sum(vertex_diffs * vertex_diffs, axis=2).min(axis=1)) * float(resolution)

    n = normal / normal_norm
    signed_dist = (points - a[None, :]) @ n
    plane_dist_vox = np.abs(signed_dist) * float(resolution)
    projected = points - signed_dist[:, None] * n[None, :]

    bary_result = barycentric_coordinates(projected, triangle)
    if bary_result is None:
        return None
    bary, _ = bary_result

    min_bary = bary.min(axis=1)
    max_bary = bary.max(axis=1)
    inside = (min_bary >= -float(barycentric_margin)) & (max_bary <= 1.0 + float(barycentric_margin))

    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-6)
    weight_sum = float(weights.sum())
    inside_fraction = float(weights[inside].sum() / weight_sum)
    mean_plane_distance = float((weights * plane_dist_vox).sum() / weight_sum)
    outside_deficit = np.maximum(-min_bary - float(barycentric_margin), 0.0)
    outside_deficit += np.maximum(max_bary - 1.0 - float(barycentric_margin), 0.0)
    mean_outside_deficit = float((weights * outside_deficit).sum() / weight_sum)

    # Lower score is better. The acceptance checks below still gate bad fits.
    fit_cost = (
        mean_plane_distance
        + 8.0 * mean_outside_deficit
        + 0.03 * float(vertex_dist_vox.max())
        + 0.01 * float(edge_lengths_vox.max())
        - 3.0 * inside_fraction
    )
    return {
        "fit_cost": float(fit_cost),
        "inside_fraction": inside_fraction,
        "mean_plane_distance_voxels": mean_plane_distance,
        "mean_outside_deficit": mean_outside_deficit,
        "max_edge_length_voxels": float(edge_lengths_vox.max()),
        "max_vertex_distance_voxels": float(vertex_dist_vox.max()),
        "area": float(area),
    }


def segment_support_count(
    edge_points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    resolution: int,
    max_distance_voxels: float,
    projection_margin_voxels: float,
) -> int:
    if edge_points.size == 0:
        return 0
    ab = b - a
    length2 = float(np.dot(ab, ab))
    if length2 < 1e-20:
        return 0
    ap = edge_points - a[None, :]
    t = (ap @ ab) / length2
    segment_length_voxels = np.sqrt(length2) * float(resolution)
    margin_t = float(projection_margin_voxels) / max(segment_length_voxels, 1e-6)
    projected = a[None, :] + t[:, None] * ab[None, :]
    dist_voxels = np.linalg.norm(edge_points - projected, axis=1) * float(resolution)
    supported = (
        (t >= -margin_t)
        & (t <= 1.0 + margin_t)
        & (dist_voxels <= float(max_distance_voxels))
    )
    return int(supported.sum())


def triangle_edge_support(
    edge_points: np.ndarray,
    triangle: np.ndarray,
    resolution: int,
    max_distance_voxels: float,
    projection_margin_voxels: float,
) -> tuple[int, int, int]:
    a, b, c = triangle
    return (
        segment_support_count(edge_points, a, b, resolution, max_distance_voxels, projection_margin_voxels),
        segment_support_count(edge_points, b, c, resolution, max_distance_voxels, projection_margin_voxels),
        segment_support_count(edge_points, c, a, resolution, max_distance_voxels, projection_margin_voxels),
    )


def choose_candidate_vertices(
    vertices: np.ndarray,
    points: np.ndarray,
    candidate_count: int,
    search_radius_voxels: float | None,
    resolution: int,
) -> np.ndarray:
    if vertices.shape[0] < 3 or int(candidate_count) < 3:
        return np.zeros((0,), dtype=np.int32)
    if search_radius_voxels is not None and float(search_radius_voxels) > 0.0:
        radius = float(search_radius_voxels) / float(resolution)
        lo = points.min(axis=0) - radius
        hi = points.max(axis=0) + radius
        candidate_pool = np.flatnonzero(
            np.all(vertices >= lo[None, :], axis=1)
            & np.all(vertices <= hi[None, :], axis=1)
        )
    else:
        candidate_pool = np.arange(vertices.shape[0], dtype=np.int32)

    if candidate_pool.size < 3:
        return np.zeros((0,), dtype=np.int32)

    diffs = vertices[candidate_pool][:, None, :] - points[None, :, :]
    min_dist2 = np.sum(diffs * diffs, axis=2).min(axis=1)
    count = min(int(candidate_count), int(candidate_pool.size))
    nearest_local = np.argpartition(min_dist2, count - 1)[:count]
    nearest_local = nearest_local[np.argsort(min_dist2[nearest_local], kind="stable")]
    return candidate_pool[nearest_local].astype(np.int32, copy=False)


def best_face_for_component(
    coords: np.ndarray,
    d_tri: np.ndarray,
    vertices: np.ndarray,
    component: np.ndarray,
    resolution: int,
    *,
    edge_points: np.ndarray,
    candidate_vertices: int,
    candidate_search_radius_voxels: float | None,
    max_points: int,
    barycentric_margin: float,
    min_area: float,
    edge_support_max_distance_voxels: float,
    edge_support_projection_margin_voxels: float,
) -> tuple[tuple[int, int, int] | None, dict[str, float | int]]:
    sampled = sample_component_indices(component, d_tri, max_points)
    points = voxel_centers(coords[sampled], resolution)
    weights = np.asarray(d_tri[sampled], dtype=np.float32)
    candidates = choose_candidate_vertices(
        vertices,
        points,
        candidate_vertices,
        candidate_search_radius_voxels,
        resolution,
    )

    base_stats: dict[str, float | int] = {
        "candidate_vertices": int(candidates.size),
        "sampled_voxels": int(sampled.size),
        "inside_fraction": 0.0,
        "mean_plane_distance_voxels": 0.0,
        "mean_outside_deficit": 0.0,
        "max_edge_length_voxels": 0.0,
        "max_vertex_distance_voxels": 0.0,
        "min_edge_support_voxels": 0,
        "edge_support_ab": 0,
        "edge_support_bc": 0,
        "edge_support_ca": 0,
        "fit_cost": 0.0,
        "area": 0.0,
    }
    if candidates.size < 3:
        return None, base_stats

    best_face = None
    best_stats = None
    for triple in itertools.combinations(candidates.tolist(), 3):
        triangle = vertices[np.asarray(triple, dtype=np.int32)]
        stats = score_vertex_triple(
            points,
            weights,
            triangle,
            resolution,
            barycentric_margin=barycentric_margin,
            min_area=min_area,
        )
        if stats is None:
            continue
        edge_counts = triangle_edge_support(
            edge_points,
            triangle,
            resolution,
            edge_support_max_distance_voxels,
            edge_support_projection_margin_voxels,
        )
        stats["min_edge_support_voxels"] = int(min(edge_counts))
        stats["edge_support_ab"] = int(edge_counts[0])
        stats["edge_support_bc"] = int(edge_counts[1])
        stats["edge_support_ca"] = int(edge_counts[2])
        stats["fit_cost"] -= 0.02 * float(min(edge_counts))
        if best_stats is None or stats["fit_cost"] < best_stats["fit_cost"]:
            best_face = tuple(int(v) for v in triple)
            best_stats = stats

    if best_face is None or best_stats is None:
        return None, base_stats
    base_stats.update(best_stats)
    return best_face, base_stats


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


def local_edge_support_points(
    coords: np.ndarray,
    edge_support_indices: np.ndarray,
    edge_support_coords: np.ndarray,
    component: np.ndarray,
    resolution: int,
    search_radius_voxels: float,
) -> np.ndarray:
    if edge_support_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    radius = int(np.ceil(float(search_radius_voxels)))
    component_coords = coords[component]
    lo = component_coords.min(axis=0) - radius
    hi = component_coords.max(axis=0) + radius
    local_mask = (
        np.all(edge_support_coords >= lo[None, :], axis=1)
        & np.all(edge_support_coords <= hi[None, :], axis=1)
    )
    local_indices = edge_support_indices[local_mask]
    if local_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return voxel_centers(coords[local_indices], resolution)


def extract_face_support_mesh(
    coords: np.ndarray,
    features: np.ndarray,
    resolution: int,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | int | float | str]]:
    d_tri = np.asarray(features[:, 0], dtype=np.float32)
    d_vert = np.asarray(features[:, 1], dtype=np.float32)

    vertex_data = extract_vertex_blobs(
        coords,
        d_vert,
        resolution,
        threshold=args.vertex_dvert_threshold,
        connectivity=args.vertex_connectivity,
        min_component_size=args.vertex_min_component_size,
        position_mode=args.vertex_position_mode,
        core_mode=args.vertex_core_mode,
    )
    vertices = vertex_data["vertices"]

    excluded_vertex_mask = vertex_data["vertex_core_mask"]
    if args.exclude_full_vertex_blobs:
        excluded_vertex_mask = vertex_data["vertex_mask"]

    grow_mask = face_blob_mask(
        d_tri,
        d_vert,
        dtri_grow_threshold=args.face_dtri_grow_threshold,
        max_dvert=args.face_max_dvert,
        vertex_mask=excluded_vertex_mask,
    )
    seeds = seed_mask(
        d_tri,
        d_vert,
        dtri_seed_threshold=args.face_dtri_seed_threshold,
        max_dvert=args.face_max_dvert,
        vertex_mask=excluded_vertex_mask,
    )
    if args.face_component_mask == "seed":
        component_mask = seeds
    elif args.face_component_mask == "grow":
        component_mask = grow_mask
    else:
        raise ValueError(f"Unsupported face_component_mask: {args.face_component_mask}")
    components = connected_components(coords, component_mask, connectivity=args.face_connectivity)

    edge_support_mask = (
        (d_tri <= float(args.edge_support_dtri_threshold))
        & (d_vert >= float(args.edge_support_min_dvert))
        & (d_vert <= float(args.edge_support_max_dvert))
    )
    edge_support_indices = np.flatnonzero(edge_support_mask)
    edge_support_coords = coords[edge_support_indices]

    face_by_key: dict[tuple[int, int, int], dict[str, float | int | tuple[int, int, int]]] = {}
    rejected_no_seed = 0
    rejected_small = 0
    rejected_large = 0
    rejected_candidates = 0
    rejected_fit = 0
    duplicate_faces = 0

    for component_id, component in enumerate(components):
        size = int(component.size)
        if size < int(args.face_min_component_size):
            rejected_small += 1
            continue
        if int(args.face_max_component_size) > 0 and size > int(args.face_max_component_size):
            rejected_large += 1
            continue
        seed_count = int(seeds[component].sum())
        if seed_count == 0:
            rejected_no_seed += 1
            continue
        edge_points = local_edge_support_points(
            coords,
            edge_support_indices,
            edge_support_coords,
            component,
            resolution,
            args.edge_support_search_radius_voxels,
        )

        face, fit_stats = best_face_for_component(
            coords,
            d_tri,
            vertices,
            component,
            resolution,
            edge_points=edge_points,
            candidate_vertices=args.face_candidate_vertices,
            candidate_search_radius_voxels=args.face_candidate_search_radius_voxels,
            max_points=args.face_max_points_per_component,
            barycentric_margin=args.face_barycentric_margin,
            min_area=args.min_face_area,
            edge_support_max_distance_voxels=args.edge_support_max_distance_voxels,
            edge_support_projection_margin_voxels=args.edge_support_projection_margin_voxels,
        )
        if face is None:
            rejected_candidates += 1
            continue

        if (
            float(fit_stats["inside_fraction"]) < float(args.face_min_inside_fraction)
            or float(fit_stats["mean_plane_distance_voxels"]) > float(args.face_max_mean_plane_distance_voxels)
            or float(fit_stats["mean_outside_deficit"]) > float(args.face_max_outside_deficit)
            or float(fit_stats["max_edge_length_voxels"]) > float(args.face_max_edge_length_voxels)
            or float(fit_stats["max_vertex_distance_voxels"]) > float(args.face_max_vertex_distance_voxels)
            or int(fit_stats["min_edge_support_voxels"]) < int(args.edge_support_min_voxels_per_edge)
        ):
            rejected_fit += 1
            continue

        key = tuple(sorted(face))
        record = {
            "oriented_face": face,
            "component_id": int(component_id),
            "component_size": size,
            "seed_voxels": seed_count,
            "sampled_voxels": int(fit_stats["sampled_voxels"]),
            "candidate_vertices": int(fit_stats["candidate_vertices"]),
            "inside_fraction": float(fit_stats["inside_fraction"]),
            "mean_plane_distance_voxels": float(fit_stats["mean_plane_distance_voxels"]),
            "mean_outside_deficit": float(fit_stats["mean_outside_deficit"]),
            "max_edge_length_voxels": float(fit_stats["max_edge_length_voxels"]),
            "max_vertex_distance_voxels": float(fit_stats["max_vertex_distance_voxels"]),
            "min_edge_support_voxels": int(fit_stats["min_edge_support_voxels"]),
            "edge_support_ab": int(fit_stats["edge_support_ab"]),
            "edge_support_bc": int(fit_stats["edge_support_bc"]),
            "edge_support_ca": int(fit_stats["edge_support_ca"]),
            "fit_cost": float(fit_stats["fit_cost"]),
            "area": float(fit_stats["area"]),
        }
        old = face_by_key.get(key)
        if old is None:
            face_by_key[key] = record
        else:
            duplicate_faces += 1
            if float(record["fit_cost"]) < float(old["fit_cost"]):
                face_by_key[key] = record

    records = [face_by_key[key] for key in sorted(face_by_key)]
    if records:
        faces = np.asarray([record["oriented_face"] for record in records], dtype=np.int32)
    else:
        faces = np.zeros((0, 3), dtype=np.int32)
    if not args.no_orient_outward:
        faces = orient_faces_outward(vertices, faces)
    edges = edge_array_from_faces(faces)

    stats: dict[str, np.ndarray | int | float | str] = {
        "algorithm": ALGO_NAME,
        "resolution": int(resolution),
        "vertices": int(vertices.shape[0]),
        "face_grow_voxels": int(grow_mask.sum()),
        "face_seed_voxels": int(seeds.sum()),
        "edge_support_voxels": int(edge_support_mask.sum()),
        "face_components": int(len(components)),
        "accepted_faces": int(faces.shape[0]),
        "face_edges": int(edges.shape[0]),
        "rejected_no_seed": int(rejected_no_seed),
        "rejected_small": int(rejected_small),
        "rejected_large": int(rejected_large),
        "rejected_candidates": int(rejected_candidates),
        "rejected_fit": int(rejected_fit),
        "duplicate_faces": int(duplicate_faces),
        "vertex_dvert_threshold": float(args.vertex_dvert_threshold),
        "vertex_raw_max_barycentric": float(threshold_to_raw_max_barycentric(args.vertex_dvert_threshold)),
        "vertex_core_mode": str(args.vertex_core_mode),
        "vertex_blob_voxels": int(vertex_data["vertex_mask"].sum()),
        "vertex_core_voxels": int(vertex_data["vertex_core_mask"].sum()),
        "face_dtri_seed_threshold": float(args.face_dtri_seed_threshold),
        "face_dtri_grow_threshold": float(args.face_dtri_grow_threshold),
        "face_max_dvert": float(args.face_max_dvert),
        "face_component_mask": str(args.face_component_mask),
        "face_connectivity": int(args.face_connectivity),
        "face_min_component_size": int(args.face_min_component_size),
        "face_max_component_size": int(args.face_max_component_size),
        "face_candidate_vertices": int(args.face_candidate_vertices),
        "face_candidate_search_radius_voxels": float(args.face_candidate_search_radius_voxels),
        "face_max_points_per_component": int(args.face_max_points_per_component),
        "face_barycentric_margin": float(args.face_barycentric_margin),
        "face_min_inside_fraction": float(args.face_min_inside_fraction),
        "face_max_mean_plane_distance_voxels": float(args.face_max_mean_plane_distance_voxels),
        "face_max_outside_deficit": float(args.face_max_outside_deficit),
        "face_max_edge_length_voxels": float(args.face_max_edge_length_voxels),
        "face_max_vertex_distance_voxels": float(args.face_max_vertex_distance_voxels),
        "edge_support_dtri_threshold": float(args.edge_support_dtri_threshold),
        "edge_support_min_dvert": float(args.edge_support_min_dvert),
        "edge_support_max_dvert": float(args.edge_support_max_dvert),
        "edge_support_search_radius_voxels": float(args.edge_support_search_radius_voxels),
        "edge_support_max_distance_voxels": float(args.edge_support_max_distance_voxels),
        "edge_support_projection_margin_voxels": float(args.edge_support_projection_margin_voxels),
        "edge_support_min_voxels_per_edge": int(args.edge_support_min_voxels_per_edge),
        "component_id": np.asarray([record["component_id"] for record in records], dtype=np.int32),
        "component_size": np.asarray([record["component_size"] for record in records], dtype=np.int32),
        "seed_voxels": np.asarray([record["seed_voxels"] for record in records], dtype=np.int32),
        "sampled_voxels": np.asarray([record["sampled_voxels"] for record in records], dtype=np.int32),
        "candidate_vertices": np.asarray([record["candidate_vertices"] for record in records], dtype=np.int32),
        "inside_fraction": np.asarray([record["inside_fraction"] for record in records], dtype=np.float32),
        "mean_plane_distance_voxels": np.asarray(
            [record["mean_plane_distance_voxels"] for record in records],
            dtype=np.float32,
        ),
        "mean_outside_deficit": np.asarray([record["mean_outside_deficit"] for record in records], dtype=np.float32),
        "max_edge_length_voxels": np.asarray([record["max_edge_length_voxels"] for record in records], dtype=np.float32),
        "max_vertex_distance_voxels": np.asarray(
            [record["max_vertex_distance_voxels"] for record in records],
            dtype=np.float32,
        ),
        "min_edge_support_voxels": np.asarray(
            [record["min_edge_support_voxels"] for record in records],
            dtype=np.int32,
        ),
        "edge_support_ab": np.asarray([record["edge_support_ab"] for record in records], dtype=np.int32),
        "edge_support_bc": np.asarray([record["edge_support_bc"] for record in records], dtype=np.int32),
        "edge_support_ca": np.asarray([record["edge_support_ca"] for record in records], dtype=np.int32),
        "fit_cost": np.asarray([record["fit_cost"] for record in records], dtype=np.float32),
        "face_area": np.asarray([record["area"] for record in records], dtype=np.float32),
    }
    return vertices, edges, faces, stats


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {ALGO_NAME}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            f.write(f"f {a} {b} {c}\n")


def write_ply(path: str | Path, vertices: np.ndarray, edges: np.ndarray, faces: np.ndarray) -> None:
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
    scalar_keys = [key for key, value in stats.items() if not isinstance(value, np.ndarray)]
    array_keys = [
        "component_id",
        "component_size",
        "seed_voxels",
        "sampled_voxels",
        "candidate_vertices",
        "inside_fraction",
        "mean_plane_distance_voxels",
        "mean_outside_deficit",
        "max_edge_length_voxels",
        "max_vertex_distance_voxels",
        "min_edge_support_voxels",
        "edge_support_ab",
        "edge_support_bc",
        "edge_support_ca",
        "fit_cost",
        "face_area",
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
    print(f"Vertices: {stats['vertices']}")
    print(f"Face-derived edges: {stats['face_edges']}")
    print(f"Accepted faces: {stats['accepted_faces']}")
    print("Face voxel extraction:")
    print(f"  grow d_tri >= {stats['face_dtri_grow_threshold']}")
    print(f"  seed d_tri >= {stats['face_dtri_seed_threshold']}")
    print(f"  d_vert <= {stats['face_max_dvert']}")
    print(f"  grow voxels = {stats['face_grow_voxels']}")
    print(f"  seed voxels = {stats['face_seed_voxels']}")
    print(f"  face components = {stats['face_components']}")
    print(f"  component mask = {stats['face_component_mask']}")
    print("Face assignment:")
    print(f"  candidate vertices = {stats['face_candidate_vertices']}")
    print(f"  candidate search radius = {stats['face_candidate_search_radius_voxels']} voxel(s)")
    print(f"  barycentric margin = {stats['face_barycentric_margin']}")
    print(f"  min inside fraction = {stats['face_min_inside_fraction']}")
    print(f"  max mean plane distance = {stats['face_max_mean_plane_distance_voxels']} voxel(s)")
    print(f"  max edge length = {stats['face_max_edge_length_voxels']} voxel(s)")
    print(f"  max vertex-to-blob distance = {stats['face_max_vertex_distance_voxels']} voxel(s)")
    print("Edge support:")
    print(f"  edge support voxels = {stats['edge_support_voxels']}")
    print(f"  edge support d_tri <= {stats['edge_support_dtri_threshold']}")
    print(f"  edge support d_vert in [{stats['edge_support_min_dvert']}, {stats['edge_support_max_dvert']}]")
    print(f"  edge support local search radius = {stats['edge_support_search_radius_voxels']} voxel(s)")
    print(f"  edge support max segment distance = {stats['edge_support_max_distance_voxels']} voxel(s)")
    print(f"  min edge support voxels per edge = {stats['edge_support_min_voxels_per_edge']}")
    print(f"  rejected no seed = {stats['rejected_no_seed']}")
    print(f"  rejected small = {stats['rejected_small']}")
    print(f"  rejected large = {stats['rejected_large']}")
    print(f"  rejected candidates = {stats['rejected_candidates']}")
    print(f"  rejected fit = {stats['rejected_fit']}")
    print(f"  duplicate faces collapsed = {stats['duplicate_faces']}")
    print("Vertex extraction:")
    print(
        f"  vertex d_vert >= {stats['vertex_dvert_threshold']} "
        f"(raw max barycentric >= {stats['vertex_raw_max_barycentric']:.6f})"
    )
    print(f"  vertex core mode = {stats['vertex_core_mode']}")
    print(f"  vertex blob voxels = {stats['vertex_blob_voxels']}")
    print(f"  vertex core voxels = {stats['vertex_core_voxels']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO_NAME}: build faces from high-d_tri voxel support.")
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

    parser.add_argument("--face_dtri_seed_threshold", type=float, default=0.60)
    parser.add_argument("--face_dtri_grow_threshold", type=float, default=0.35)
    parser.add_argument("--face_max_dvert", type=float, default=0.55)
    parser.add_argument("--face_component_mask", choices=["seed", "grow"], default="seed")
    parser.add_argument("--face_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--face_min_component_size", type=int, default=4)
    parser.add_argument("--face_max_component_size", type=int, default=0)
    parser.add_argument("--exclude_full_vertex_blobs", action="store_true")

    parser.add_argument("--face_candidate_vertices", type=int, default=16)
    parser.add_argument("--face_candidate_search_radius_voxels", type=float, default=56.0)
    parser.add_argument("--face_max_points_per_component", type=int, default=256)
    parser.add_argument("--face_barycentric_margin", type=float, default=0.08)
    parser.add_argument("--face_min_inside_fraction", type=float, default=0.60)
    parser.add_argument("--face_max_mean_plane_distance_voxels", type=float, default=3.0)
    parser.add_argument("--face_max_outside_deficit", type=float, default=0.18)
    parser.add_argument("--face_max_edge_length_voxels", type=float, default=64.0)
    parser.add_argument("--face_max_vertex_distance_voxels", type=float, default=42.0)
    parser.add_argument("--edge_support_dtri_threshold", type=float, default=0.25)
    parser.add_argument("--edge_support_min_dvert", type=float, default=0.10)
    parser.add_argument("--edge_support_max_dvert", type=float, default=0.84)
    parser.add_argument("--edge_support_search_radius_voxels", type=float, default=40.0)
    parser.add_argument("--edge_support_max_distance_voxels", type=float, default=3.0)
    parser.add_argument("--edge_support_projection_margin_voxels", type=float, default=3.0)
    parser.add_argument("--edge_support_min_voxels_per_edge", type=int, default=3)
    parser.add_argument("--min_face_area", type=float, default=1e-12)
    parser.add_argument("--no_orient_outward", action="store_true")

    parser.add_argument("--out_obj", type=str, default=None)
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--out_npz", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)

    vertices, edges, faces, stats = extract_face_support_mesh(coords, features, resolution, args)

    stem = default_output_stem(path)
    out_dir = default_results_dir(path)
    out_obj = Path(args.out_obj) if args.out_obj else out_dir / f"{stem}_face_support_mesh.obj"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_face_support_mesh.ply"
    out_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{stem}_face_support_mesh.npz"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"{stem}_face_support_mesh_faces.csv"

    write_obj(out_obj, vertices, faces)
    write_ply(out_ply, vertices, edges, faces)
    write_npz(out_npz, vertices, edges, faces, stats)
    write_csv(out_csv, faces, stats)

    print_summary(stats)
    print(f"Saved {out_obj}")
    print(f"Saved {out_ply}")
    print(f"Saved {out_npz}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
