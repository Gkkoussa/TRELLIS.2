#!/usr/bin/env python3
"""Run voxel-constrained QEM on one saved triangle-field example and visualize it.

This is an isolated prototype. It reads the existing triangle-field support and
PBR mesh dump, writes only to the requested output directory, and does not
modify dataset metadata or existing voxel payloads.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import io
import itertools
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


DEFAULT_ROOT = Path("/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose one saved triangle-field object, collapse mesh edges whose "
            "endpoints share an active voxel using QEM, and render a comparison."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument(
        "--sha256",
        default=None,
        help="Specific object. If omitted, choose reproducibly from stage metadata.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/qem_voxel_collapse_demo_r<resolution>.",
    )
    parser.add_argument(
        "--min-input-faces",
        type=int,
        default=2000,
        help="Random selection skips very coarse meshes that have almost nothing to collapse.",
    )
    parser.add_argument(
        "--max-input-faces",
        type=int,
        default=50000,
        help="Random selection skips larger meshes so the Python reference QEM remains practical.",
    )
    parser.add_argument(
        "--max-active-voxels",
        type=int,
        default=15000,
        help="Random selection skips examples above this support size.",
    )
    parser.add_argument("--max-selection-attempts", type=int, default=2000)
    parser.add_argument(
        "--max-collapses",
        type=int,
        default=0,
        help="Zero means continue until no valid same-voxel edge remains.",
    )
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument(
        "--normal-dot-min",
        type=float,
        default=0.0,
        help="Minimum old/new face-normal dot product. Zero rejects flips over 90 degrees.",
    )
    parser.add_argument("--render-face-limit", type=int, default=50000)
    parser.add_argument("--render-edge-limit", type=int, default=60000)
    parser.add_argument("--render-voxel-limit", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def find_voxel_path(voxel_root: Path, sha256: str) -> Path:
    for suffix in (".npz.zst", ".npz"):
        path = voxel_root / f"{sha256}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No voxel payload for {sha256} under {voxel_root}")


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


def load_voxel_payload(voxel_root: Path, sha256: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = find_voxel_path(voxel_root, sha256)
    with load_npz(path) as data:
        coords = np.asarray(data["coords"], dtype=np.int32)
        features = np.asarray(data["features"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} has invalid coords shape {coords.shape}")
    if features.ndim != 2 or features.shape[0] != coords.shape[0]:
        raise ValueError(f"{path} has invalid features shape {features.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path} contains no active voxels")
    return coords, features, path


def normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) * 0.5
    extent = float(np.max(upper - lower))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Mesh bounding box is empty or invalid")
    scale = 0.99999 / extent
    return (vertices - center) * scale, center, scale


def remove_unreferenced_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces.reshape(-1))
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    return vertices[used], remap[faces]


def load_normalized_pbr_mesh(
    pbr_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with pbr_path.open("rb") as handle:
        dump = pickle.load(handle)

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    offset = 0
    for obj in dump.get("objects", []):
        vertices = np.asarray(obj.get("vertices"), dtype=np.float64)
        faces = np.asarray(obj.get("faces"), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            continue
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            continue
        valid_indices = (faces >= 0).all(axis=1) & (faces < vertices.shape[0]).all(axis=1)
        faces = faces[valid_indices]
        if faces.shape[0] == 0:
            continue
        vertices_parts.append(vertices)
        faces_parts.append(faces + offset)
        offset += vertices.shape[0]

    if not vertices_parts:
        raise ValueError(f"{pbr_path} contains no usable triangle mesh")

    vertices = np.concatenate(vertices_parts, axis=0)
    faces = np.concatenate(faces_parts, axis=0)
    vertices, center, scale = normalize_vertices(vertices)

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    valid = np.isfinite(cross).all(axis=1) & (np.linalg.norm(cross, axis=1) > 1e-12)
    faces = faces[valid]
    if faces.shape[0] == 0:
        raise ValueError(f"{pbr_path} has no non-degenerate triangles after normalization")
    vertices, faces = remove_unreferenced_vertices(vertices, faces)
    return vertices, faces, {
        "normalization_center": center.tolist(),
        "normalization_scale": scale,
    }


def read_stage_rows(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if truthy(row.get("triangle_field_voxelized", False))
        and int(float(row.get("num_triangle_field_voxels") or 0)) > 0
    ]


def choose_example(
    *,
    root: Path,
    resolution: int,
    requested_sha256: str | None,
    seed: int,
    min_input_faces: int,
    max_input_faces: int,
    max_active_voxels: int,
    max_attempts: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, Path, dict[str, object]]:
    voxel_root = root / f"triangle_field_voxels_{resolution}"
    metadata_path = voxel_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing stage metadata: {metadata_path}")

    rows = read_stage_rows(metadata_path)
    by_sha = {str(row["sha256"]): row for row in rows}
    if requested_sha256 is not None:
        if requested_sha256 not in by_sha:
            raise ValueError(f"{requested_sha256} is not successful in {metadata_path}")
        candidates = [requested_sha256]
    else:
        candidates = [
            str(row["sha256"])
            for row in rows
            if int(float(row["num_triangle_field_voxels"])) <= max_active_voxels
        ]
        random.Random(seed).shuffle(candidates)

    errors: list[str] = []
    for sha256 in candidates[:max_attempts]:
        pbr_path = root / "pbr_dumps" / f"{sha256}.pickle"
        if not pbr_path.is_file():
            errors.append(f"{sha256}: missing PBR dump")
            continue
        try:
            coords, features, voxel_path = load_voxel_payload(voxel_root, sha256)
            vertices, faces, mesh_meta = load_normalized_pbr_mesh(pbr_path)
            if requested_sha256 is None:
                if faces.shape[0] < min_input_faces:
                    errors.append(f"{sha256}: {faces.shape[0]} faces is below selection minimum")
                    continue
                if faces.shape[0] > max_input_faces:
                    errors.append(f"{sha256}: {faces.shape[0]} faces exceeds selection limit")
                    continue
            return sha256, coords, features, vertices, faces, voxel_path, pbr_path, mesh_meta
        except Exception as exc:
            errors.append(f"{sha256}: {exc}")

    preview = "\n".join(errors[:20])
    raise RuntimeError(
        f"Could not select an eligible example after {min(len(candidates), max_attempts)} attempts.\n{preview}"
    )


def coords_to_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords64 = np.asarray(coords, dtype=np.int64)
    return (coords64[:, 0] * resolution + coords64[:, 1]) * resolution + coords64[:, 2]


def assign_vertices_to_support(
    vertices: np.ndarray,
    support_coords: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, int]:
    coords = np.floor((vertices + 0.5) * resolution).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    support_keys = set(int(key) for key in coords_to_keys(support_coords, resolution))

    reassigned = 0
    offsets = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int32)
    for index, coord in enumerate(coords):
        key = int((int(coord[0]) * resolution + int(coord[1])) * resolution + int(coord[2]))
        if key in support_keys:
            continue

        candidates = coord[None, :] + offsets
        in_bounds = ((candidates >= 0) & (candidates < resolution)).all(axis=1)
        candidates = candidates[in_bounds]
        hits = [
            candidate
            for candidate in candidates
            if int(
                (int(candidate[0]) * resolution + int(candidate[1])) * resolution
                + int(candidate[2])
            )
            in support_keys
        ]
        if not hits:
            raise ValueError(
                f"Vertex {index} at {vertices[index].tolist()} maps to absent voxel "
                f"{coord.tolist()} and has no active voxel within one cell"
            )
        hit_array = np.asarray(hits, dtype=np.int32)
        centers = (hit_array.astype(np.float64) + 0.5) / resolution - 0.5
        best = int(np.argmin(np.sum((centers - vertices[index]) ** 2, axis=1)))
        coords[index] = hit_array[best]
        reassigned += 1
    return coords, reassigned


def unique_edges(faces: np.ndarray) -> np.ndarray:
    if faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def compute_vertex_quadrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_weight: float,
) -> np.ndarray:
    triangles = vertices[faces]
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    norm = np.linalg.norm(raw_normals, axis=1)
    unit_normals = raw_normals / np.maximum(norm[:, None], 1e-30)
    planes = np.concatenate(
        [unit_normals, -np.sum(unit_normals * triangles[:, 0], axis=1, keepdims=True)],
        axis=1,
    )
    face_quadrics = np.einsum("fi,fj->fij", planes, planes)
    face_quadrics *= (0.5 * norm)[:, None, None]

    quadrics = np.zeros((vertices.shape[0], 4, 4), dtype=np.float64)
    for corner in range(3):
        np.add.at(quadrics, faces[:, corner], face_quadrics)

    if boundary_weight <= 0:
        return quadrics

    all_edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    sorted_edges = np.sort(all_edges, axis=1)
    face_ids = np.tile(np.arange(faces.shape[0], dtype=np.int64), 3)
    boundary_edges, first, counts = np.unique(
        sorted_edges,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_mask = counts == 1
    boundary_edges = boundary_edges[boundary_mask]
    boundary_face_ids = face_ids[first[boundary_mask]]

    for edge, face_id in zip(boundary_edges, boundary_face_ids):
        start = vertices[edge[0]]
        end = vertices[edge[1]]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-15:
            continue
        direction /= length
        boundary_normal = np.cross(direction, unit_normals[face_id])
        boundary_norm = float(np.linalg.norm(boundary_normal))
        if boundary_norm <= 1e-15:
            continue
        boundary_normal /= boundary_norm
        plane = np.append(boundary_normal, -float(np.dot(boundary_normal, start)))
        quadric = np.outer(plane, plane) * (boundary_weight * length)
        quadrics[edge[0]] += quadric
        quadrics[edge[1]] += quadric
    return quadrics


def quadric_cost(quadric: np.ndarray, position: np.ndarray) -> float:
    homogeneous = np.append(position, 1.0)
    return float(homogeneous @ quadric @ homogeneous)


def box_constrained_quadric_minimum(
    quadric: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fallbacks: Iterable[np.ndarray],
) -> tuple[float, np.ndarray]:
    matrix = quadric[:3, :3]
    linear = quadric[:3, 3]
    best_cost = math.inf
    best_position: np.ndarray | None = None
    tolerance = 1e-10

    # Per axis: 0 means free, -1 lower bound, +1 upper bound.
    for state in itertools.product((0, -1, 1), repeat=3):
        fixed = np.asarray([axis for axis, value in enumerate(state) if value != 0], dtype=np.int64)
        free = np.asarray([axis for axis, value in enumerate(state) if value == 0], dtype=np.int64)
        position = np.zeros(3, dtype=np.float64)
        if fixed.size:
            position[fixed] = [
                lower[axis] if state[axis] < 0 else upper[axis]
                for axis in fixed
            ]
        if free.size:
            rhs = -linear[free]
            if fixed.size:
                rhs -= matrix[np.ix_(free, fixed)] @ position[fixed]
            try:
                solution, *_ = np.linalg.lstsq(matrix[np.ix_(free, free)], rhs, rcond=None)
            except np.linalg.LinAlgError:
                continue
            position[free] = solution
        if not np.isfinite(position).all():
            continue
        if np.any(position < lower - tolerance) or np.any(position > upper + tolerance):
            continue
        position = np.clip(position, lower, upper)
        cost = quadric_cost(quadric, position)
        if cost < best_cost:
            best_cost = cost
            best_position = position

    for fallback in fallbacks:
        position = np.clip(np.asarray(fallback, dtype=np.float64), lower, upper)
        cost = quadric_cost(quadric, position)
        if cost < best_cost:
            best_cost = cost
            best_position = position

    if best_position is None:
        best_position = (lower + upper) * 0.5
        best_cost = quadric_cost(quadric, best_position)
    return max(best_cost, 0.0), best_position


@dataclass
class CollapseStats:
    initial_candidate_edges: int = 0
    accepted: int = 0
    rejected_link: int = 0
    rejected_geometry: int = 0
    stale_heap_entries: int = 0


class VoxelConstrainedQEM:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_voxels: np.ndarray,
        resolution: int,
        boundary_weight: float,
        normal_dot_min: float,
        verbose: bool,
    ):
        self.vertices = np.asarray(vertices, dtype=np.float64).copy()
        self.faces = np.asarray(faces, dtype=np.int64).copy()
        self.vertex_voxels = np.asarray(vertex_voxels, dtype=np.int32).copy()
        self.resolution = int(resolution)
        self.normal_dot_min = float(normal_dot_min)
        self.verbose = bool(verbose)

        self.active_vertices = np.ones(self.vertices.shape[0], dtype=bool)
        self.active_faces = np.ones(self.faces.shape[0], dtype=bool)
        self.quadrics = compute_vertex_quadrics(self.vertices, self.faces, boundary_weight)
        self.vertex_faces = [set() for _ in range(self.vertices.shape[0])]
        self.neighbors = [set() for _ in range(self.vertices.shape[0])]
        self.versions = np.zeros(self.vertices.shape[0], dtype=np.int64)
        self.heap: list[tuple[float, int, int, int, int, int, np.ndarray]] = []
        self.heap_counter = 0
        self.stats = CollapseStats()
        self.minimum_area2 = 1e-14

        for face_id, face in enumerate(self.faces):
            a, b, c = (int(value) for value in face)
            self.vertex_faces[a].add(face_id)
            self.vertex_faces[b].add(face_id)
            self.vertex_faces[c].add(face_id)
            self.neighbors[a].update((b, c))
            self.neighbors[b].update((a, c))
            self.neighbors[c].update((a, b))

        for u, v in unique_edges(self.faces):
            if np.array_equal(self.vertex_voxels[u], self.vertex_voxels[v]):
                self._push_edge(int(u), int(v))
                self.stats.initial_candidate_edges += 1

    def _edge_candidate(self, u: int, v: int) -> tuple[float, np.ndarray]:
        voxel = self.vertex_voxels[u].astype(np.float64)
        lower = voxel / self.resolution - 0.5
        upper = (voxel + 1.0) / self.resolution - 0.5
        quadric = self.quadrics[u] + self.quadrics[v]
        return box_constrained_quadric_minimum(
            quadric,
            lower,
            upper,
            (self.vertices[u], self.vertices[v], (self.vertices[u] + self.vertices[v]) * 0.5),
        )

    def _push_edge(self, u: int, v: int) -> None:
        if u == v or not self.active_vertices[u] or not self.active_vertices[v]:
            return
        if v not in self.neighbors[u]:
            return
        if not np.array_equal(self.vertex_voxels[u], self.vertex_voxels[v]):
            return
        if u > v:
            u, v = v, u
        cost, position = self._edge_candidate(u, v)
        self.heap_counter += 1
        heapq.heappush(
            self.heap,
            (
                cost,
                self.heap_counter,
                u,
                v,
                int(self.versions[u]),
                int(self.versions[v]),
                position,
            ),
        )

    def _is_link_valid(self, u: int, v: int) -> bool:
        edge_faces = self.vertex_faces[u] & self.vertex_faces[v]
        edge_faces = {face_id for face_id in edge_faces if self.active_faces[face_id]}
        if len(edge_faces) == 0 or len(edge_faces) > 2:
            return False
        common = {
            vertex
            for vertex in (self.neighbors[u] & self.neighbors[v])
            if self.active_vertices[vertex]
        }
        opposite: set[int] = set()
        for face_id in edge_faces:
            opposite.update(int(vertex) for vertex in self.faces[face_id] if vertex not in (u, v))
        return common == opposite

    def _is_geometry_valid(self, u: int, v: int, position: np.ndarray) -> bool:
        affected = self.vertex_faces[u] | self.vertex_faces[v]
        for face_id in affected:
            if not self.active_faces[face_id]:
                continue
            old_face = self.faces[face_id]
            new_face = np.where(old_face == v, u, old_face)
            if len(set(int(value) for value in new_face)) < 3:
                continue

            old_triangle = self.vertices[old_face]
            new_triangle = self.vertices[new_face].copy()
            new_triangle[new_face == u] = position
            old_normal = np.cross(
                old_triangle[1] - old_triangle[0],
                old_triangle[2] - old_triangle[0],
            )
            new_normal = np.cross(
                new_triangle[1] - new_triangle[0],
                new_triangle[2] - new_triangle[0],
            )
            old_norm = float(np.linalg.norm(old_normal))
            new_norm = float(np.linalg.norm(new_normal))
            if not np.isfinite(new_norm) or new_norm <= self.minimum_area2:
                return False
            if old_norm > self.minimum_area2:
                normal_dot = float(np.dot(old_normal, new_normal) / (old_norm * new_norm))
                if normal_dot < self.normal_dot_min:
                    return False
        return True

    def _rebuild_neighbors(self, vertices: set[int]) -> None:
        for vertex in vertices:
            if not self.active_vertices[vertex]:
                self.neighbors[vertex].clear()
                continue
            rebuilt: set[int] = set()
            for face_id in self.vertex_faces[vertex]:
                if self.active_faces[face_id]:
                    rebuilt.update(
                        int(other)
                        for other in self.faces[face_id]
                        if int(other) != vertex and self.active_vertices[int(other)]
                    )
            self.neighbors[vertex] = rebuilt

    def _collapse(self, u: int, v: int, position: np.ndarray) -> None:
        affected_faces = {
            face_id
            for face_id in (self.vertex_faces[u] | self.vertex_faces[v])
            if self.active_faces[face_id]
        }
        impacted_vertices: set[int] = {u, v}
        for face_id in affected_faces:
            impacted_vertices.update(int(vertex) for vertex in self.faces[face_id])
        impacted_vertices.update(self.neighbors[u])
        impacted_vertices.update(self.neighbors[v])

        for face_id in affected_faces:
            for vertex in self.faces[face_id]:
                self.vertex_faces[int(vertex)].discard(face_id)

        self.vertices[u] = position
        self.quadrics[u] += self.quadrics[v]
        self.active_vertices[v] = False

        for face_id in affected_faces:
            face = np.where(self.faces[face_id] == v, u, self.faces[face_id])
            if len(set(int(value) for value in face)) < 3:
                self.active_faces[face_id] = False
                continue
            self.faces[face_id] = face
            for vertex in face:
                self.vertex_faces[int(vertex)].add(face_id)

        self.vertex_faces[v].clear()
        self._rebuild_neighbors(impacted_vertices)

        for vertex in impacted_vertices:
            self.versions[vertex] += 1

        queued: set[tuple[int, int]] = set()
        for vertex in impacted_vertices:
            if not self.active_vertices[vertex]:
                continue
            for neighbor in self.neighbors[vertex]:
                edge = (min(vertex, neighbor), max(vertex, neighbor))
                if edge in queued:
                    continue
                queued.add(edge)
                self._push_edge(*edge)

    def run(self, max_collapses: int = 0) -> CollapseStats:
        while self.heap and (max_collapses <= 0 or self.stats.accepted < max_collapses):
            _, _, u, v, version_u, version_v, position = heapq.heappop(self.heap)
            if (
                not self.active_vertices[u]
                or not self.active_vertices[v]
                or version_u != self.versions[u]
                or version_v != self.versions[v]
                or v not in self.neighbors[u]
            ):
                self.stats.stale_heap_entries += 1
                continue
            if not np.array_equal(self.vertex_voxels[u], self.vertex_voxels[v]):
                self.stats.stale_heap_entries += 1
                continue
            if not self._is_link_valid(u, v):
                self.stats.rejected_link += 1
                continue
            if not self._is_geometry_valid(u, v, position):
                self.stats.rejected_geometry += 1
                continue
            self._collapse(u, v, position)
            self.stats.accepted += 1
            if self.verbose and self.stats.accepted % 1000 == 0:
                print(
                    f"Accepted {self.stats.accepted} collapses; "
                    f"active vertices={int(self.active_vertices.sum())}; "
                    f"heap={len(self.heap)}",
                    flush=True,
                )
        return self.stats

    def compact(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        active_ids = np.flatnonzero(self.active_vertices)
        remap = np.full(self.vertices.shape[0], -1, dtype=np.int64)
        remap[active_ids] = np.arange(active_ids.shape[0], dtype=np.int64)
        out_vertices = self.vertices[active_ids]
        out_voxels = self.vertex_voxels[active_ids]

        faces = self.faces[self.active_faces]
        if faces.shape[0]:
            faces = remap[faces]
            valid = (faces >= 0).all(axis=1)
            valid &= (
                (faces[:, 0] != faces[:, 1])
                & (faces[:, 1] != faces[:, 2])
                & (faces[:, 2] != faces[:, 0])
            )
            faces = faces[valid]
            if faces.shape[0]:
                triangles = out_vertices[faces]
                valid = np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=1,
                ) > self.minimum_area2
                faces = faces[valid]
            if faces.shape[0]:
                keys = np.sort(faces, axis=1)
                _, first = np.unique(keys, axis=0, return_index=True)
                faces = faces[np.sort(first)]
        out_edges = unique_edges(faces)
        return out_vertices, faces, out_edges, out_voxels


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray) -> None:
    with path.open("w") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            handle.write(
                f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n"
            )
        if faces.shape[0] == 0:
            for edge in edges:
                handle.write(f"l {int(edge[0]) + 1} {int(edge[1]) + 1}\n")


def configure_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_xlim(-0.52, 0.52)
    axis.set_ylim(-0.52, 0.52)
    axis.set_zlim(-0.52, 0.52)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=38)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.grid(False)


def subsample_indices(count: int, limit: int, seed: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.random.default_rng(seed).choice(count, size=limit, replace=False)


def draw_mesh(
    axis,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    face_limit: int,
    seed: int,
    color: str,
    alpha: float,
) -> None:
    if faces.shape[0] == 0:
        axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2, c=color)
        return
    selected = subsample_indices(faces.shape[0], face_limit, seed)
    collection = Poly3DCollection(
        vertices[faces[selected]],
        facecolor=color,
        edgecolor=(0.1, 0.1, 0.1, 0.08),
        linewidth=0.08,
        alpha=alpha,
    )
    axis.add_collection3d(collection)


def draw_graph(
    axis,
    vertices: np.ndarray,
    edges: np.ndarray,
    *,
    edge_limit: int,
    seed: int,
) -> None:
    selected = subsample_indices(edges.shape[0], edge_limit, seed)
    if selected.size:
        axis.add_collection3d(
            Line3DCollection(
                vertices[edges[selected]],
                colors=(0.08, 0.12, 0.18, 0.65),
                linewidths=0.35,
            )
        )
    vertex_ids = subsample_indices(vertices.shape[0], max(edge_limit // 2, 1), seed + 1)
    axis.scatter(
        vertices[vertex_ids, 0],
        vertices[vertex_ids, 1],
        vertices[vertex_ids, 2],
        s=2.0,
        c="#e34a33",
        alpha=0.9,
        depthshade=False,
    )


def draw_voxel_support(
    axis,
    coords: np.ndarray,
    values: np.ndarray,
    resolution: int,
) -> None:
    filled = np.zeros((resolution, resolution, resolution), dtype=bool)
    colors = np.zeros((resolution, resolution, resolution, 4), dtype=np.float32)
    filled[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    colors[coords[:, 0], coords[:, 1], coords[:, 2]] = plt.get_cmap("viridis")(
        np.clip(values, 0.0, 1.0)
    )
    grid = np.linspace(-0.5, 0.5, resolution + 1)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    axis.voxels(
        x,
        y,
        z,
        filled,
        facecolors=colors,
        edgecolor=(0.08, 0.1, 0.12, 0.18),
        linewidth=0.08,
        shade=True,
    )


def render_comparison(
    path: Path,
    *,
    sha256: str,
    resolution: int,
    original_vertices: np.ndarray,
    original_faces: np.ndarray,
    support_coords: np.ndarray,
    support_features: np.ndarray,
    collapsed_vertices: np.ndarray,
    collapsed_faces: np.ndarray,
    collapsed_edges: np.ndarray,
    face_limit: int,
    edge_limit: int,
    voxel_limit: int,
    seed: int,
) -> None:
    figure = plt.figure(figsize=(16, 13), dpi=150)
    axes = [
        figure.add_subplot(2, 2, index + 1, projection="3d")
        for index in range(4)
    ]

    draw_mesh(
        axes[0],
        original_vertices,
        original_faces,
        face_limit=face_limit,
        seed=seed,
        color="#8ecae6",
        alpha=0.88,
    )
    configure_axis(
        axes[0],
        f"Original normalized mesh\nV={len(original_vertices):,}, F={len(original_faces):,}",
    )

    voxel_ids = subsample_indices(len(support_coords), voxel_limit, seed + 2)
    d_tri = (
        support_features[voxel_ids, 0]
        if support_features.shape[1]
        else np.zeros(len(voxel_ids))
    )
    if len(voxel_ids) == len(support_coords) and resolution <= 64:
        draw_voxel_support(
            axes[1],
            support_coords[voxel_ids],
            d_tri,
            resolution,
        )
    else:
        centers = (
            support_coords[voxel_ids].astype(np.float64) + 0.5
        ) / resolution - 0.5
        axes[1].scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c=d_tri,
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=max(3.0, 100.0 / resolution),
            alpha=0.8,
            depthshade=False,
        )
    configure_axis(
        axes[1],
        f"Saved resolution-{resolution} active support\nN={len(support_coords):,}, color=d_tri",
    )

    draw_mesh(
        axes[2],
        collapsed_vertices,
        collapsed_faces,
        face_limit=face_limit,
        seed=seed + 3,
        color="#90be6d",
        alpha=0.9,
    )
    configure_axis(
        axes[2],
        f"Voxel-constrained QEM surface\nV={len(collapsed_vertices):,}, F={len(collapsed_faces):,}",
    )

    support_graph_ids = subsample_indices(len(support_coords), min(voxel_limit, 5000), seed + 4)
    graph_centers = (
        support_coords[support_graph_ids].astype(np.float64) + 0.5
    ) / resolution - 0.5
    axes[3].scatter(
        graph_centers[:, 0],
        graph_centers[:, 1],
        graph_centers[:, 2],
        s=0.5,
        c="#adb5bd",
        alpha=0.12,
        depthshade=False,
    )
    draw_graph(
        axes[3],
        collapsed_vertices,
        collapsed_edges,
        edge_limit=edge_limit,
        seed=seed + 5,
    )
    configure_axis(
        axes[3],
        f"Collapsed vertex-edge graph over support\nE={len(collapsed_edges):,}",
    )

    figure.suptitle(
        f"Voxel-constrained QEM demo | {sha256} | resolution {resolution}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("outputs") / f"qem_voxel_collapse_demo_r{args.resolution}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        sha256,
        support_coords,
        support_features,
        vertices,
        faces,
        voxel_path,
        pbr_path,
        mesh_meta,
    ) = choose_example(
        root=args.root,
        resolution=args.resolution,
        requested_sha256=args.sha256,
        seed=args.seed,
        min_input_faces=args.min_input_faces,
        max_input_faces=args.max_input_faces,
        max_active_voxels=args.max_active_voxels,
        max_attempts=args.max_selection_attempts,
    )
    prefix = output_dir / f"{sha256}_r{args.resolution}"
    output_paths = {
        "payload": prefix.with_suffix(".qem.npz"),
        "original_obj": prefix.with_suffix(".original.obj"),
        "collapsed_obj": prefix.with_suffix(".collapsed.obj"),
        "visualization": prefix.with_suffix(".png"),
        "summary": prefix.with_suffix(".json"),
    }
    if not args.overwrite:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing outputs; pass --overwrite:\n"
                + "\n".join(str(path) for path in existing)
            )

    print(f"Selected {sha256}")
    print(f"Voxel payload: {voxel_path}")
    print(f"PBR mesh: {pbr_path}")
    print(
        f"Input: vertices={len(vertices):,}, faces={len(faces):,}, "
        f"active_voxels={len(support_coords):,}",
        flush=True,
    )

    vertex_voxels, reassigned = assign_vertices_to_support(
        vertices,
        support_coords,
        args.resolution,
    )
    # Use the dataset-stage algorithm: all vertices assigned to one active
    # voxel become one box-constrained QEM representative.
    from qem_edge_collapse import compute_normals_and_areas, voxel_qem_collapse

    out_vertices, out_faces, out_edges, out_voxels, collapse_stats = voxel_qem_collapse(
        vertices,
        faces,
        vertex_voxels,
        support_coords,
        args.resolution,
        args.boundary_weight,
    )
    vertex_normals, face_normals, face_areas = compute_normals_and_areas(
        out_vertices, out_faces
    )
    local_offsets = (out_vertices + 0.5) * args.resolution - out_voxels

    support_key_set = set(int(key) for key in coords_to_keys(support_coords, args.resolution))
    output_voxel_keys = set(int(key) for key in coords_to_keys(out_voxels, args.resolution))
    vertices_per_voxel = np.unique(out_voxels, axis=0, return_counts=True)[1]
    active_with_vertices = len(output_voxel_keys & support_key_set)
    active_without_vertices = len(support_key_set - output_voxel_keys)

    summary = {
        "sha256": sha256,
        "resolution": args.resolution,
        "seed": args.seed,
        "source_voxel_path": str(voxel_path),
        "source_pbr_path": str(pbr_path),
        "input_vertices": int(vertices.shape[0]),
        "input_faces": int(faces.shape[0]),
        "input_edges": int(unique_edges(faces).shape[0]),
        "active_voxels": int(support_coords.shape[0]),
        "vertices_reassigned_to_touching_active_voxel": int(reassigned),
        "collapse_mode": "one_qem_representative_per_vertex_occupied_active_voxel",
        "accepted_collapses": collapse_stats["num_group_collapses"],
        "output_vertices": int(out_vertices.shape[0]),
        "output_faces": int(out_faces.shape[0]),
        "output_edges": int(out_edges.shape[0]),
        "active_voxels_with_output_vertices": int(active_with_vertices),
        "active_voxels_without_output_vertices": int(active_without_vertices),
        "max_vertices_per_voxel": int(vertices_per_voxel.max(initial=0)),
        "mean_vertices_per_occupied_vertex_voxel": (
            float(vertices_per_voxel.mean()) if vertices_per_voxel.size else 0.0
        ),
        "local_offset_min": local_offsets.min(axis=0).tolist() if local_offsets.size else [],
        "local_offset_max": local_offsets.max(axis=0).tolist() if local_offsets.size else [],
        "boundary_weight": args.boundary_weight,
        "vertices_outside_sparse_support": 0,
        "physical_saved_voxel_mismatches": 0,
        **collapse_stats,
        **mesh_meta,
    }

    np.savez_compressed(
        output_paths["payload"],
        vertices=out_vertices.astype(np.float32),
        faces=out_faces.astype(np.int32),
        edges=out_edges.astype(np.int32),
        vertex_normals=vertex_normals.astype(np.float32),
        face_normals=face_normals.astype(np.float32),
        face_areas=face_areas.astype(np.float32),
        vertex_voxel_coords=out_voxels.astype(np.int32),
        vertex_local_offsets=local_offsets.astype(np.float32),
        active_voxel_coords=support_coords.astype(np.int32),
        metadata_json=np.asarray(json.dumps(summary)),
    )
    write_obj(output_paths["original_obj"], vertices, faces, unique_edges(faces))
    write_obj(output_paths["collapsed_obj"], out_vertices, out_faces, out_edges)
    write_obj(output_dir / "original.obj", vertices, faces, unique_edges(faces))
    write_obj(output_dir / "collapsed.obj", out_vertices, out_faces, out_edges)
    render_comparison(
        output_paths["visualization"],
        sha256=sha256,
        resolution=args.resolution,
        original_vertices=vertices,
        original_faces=faces,
        support_coords=support_coords,
        support_features=support_features,
        collapsed_vertices=out_vertices,
        collapsed_faces=out_faces,
        collapsed_edges=out_edges,
        face_limit=args.render_face_limit,
        edge_limit=args.render_edge_limit,
        voxel_limit=args.render_voxel_limit,
        seed=args.seed,
    )
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"Output: vertices={len(out_vertices):,}, faces={len(out_faces):,}, "
        f"edges={len(out_edges):,}, "
        f"accepted_collapses={collapse_stats['num_group_collapses']:,}",
        flush=True,
    )
    print(f"Visualization: {output_paths['visualization']}")
    print(f"Summary: {output_paths['summary']}")
    print(f"QEM payload: {output_paths['payload']}")


if __name__ == "__main__":
    main()
