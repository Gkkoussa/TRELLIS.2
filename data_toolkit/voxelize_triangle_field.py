import os
import sys
import json
import time
import pickle
import argparse
import importlib
import io
from contextlib import contextmanager
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

import o_voxel


FORMAT_NAME = 'triangle_field_voxel_npz'
FORMAT_VERSION = 2
INPUT_LAYOUT = [
    ['d_tri', 1],
    ['d_vert', 1],
    ['offset_to_v0', 3],
    ['offset_to_v1', 3],
    ['offset_to_v2', 3],
    ['offset_to_centroid', 3],
    ['face_normal', 3],
    ['offset_to_projection', 3],
]
DEBUG_VERBOSE = False
BENCHMARK_ENABLED = False


def debug_log(message: str):
    if DEBUG_VERBOSE:
        print(f'[triangle-field-debug] {message}', flush=True)


def benchmark_log(message: str):
    if BENCHMARK_ENABLED:
        print(f'[triangle-field-benchmark] {message}', flush=True)


def _format_shape(value) -> str:
    if value is None:
        return 'None'
    shape = getattr(value, 'shape', None)
    dtype = getattr(value, 'dtype', None)
    return f'shape={shape}, dtype={dtype}'


def make_default_material_pack():
    return {
        'baseColorFactor': [1.0, 1.0, 1.0, 1.0],
        'alphaFactor': 1.0,
        'metallicFactor': 0.0,
        'roughnessFactor': 1.0,
        'alphaMode': 'OPAQUE',
        'alphaCutoff': 0.5,
        'baseColorTexture': None,
        'alphaTexture': None,
        'metallicTexture': None,
        'roughnessTexture': None,
    }


def sanitize_dump_for_volumetric_convert(dump):
    changes = []
    materials = dump.get('materials')
    if materials is None:
        dump['materials'] = []
        materials = dump['materials']
        changes.append('created empty materials list')

    if len(materials) == 0:
        materials.append(make_default_material_pack())
        changes.append('added default material because dump had no materials')

    num_materials = len(materials)
    for object_idx, obj in enumerate(dump.get('objects', [])):
        faces = obj.get('faces')
        mat_ids = obj.get('mat_ids')
        if not isinstance(faces, np.ndarray) or faces.ndim != 2:
            continue
        num_faces = faces.shape[0]

        if mat_ids is None:
            obj['mat_ids'] = np.zeros((num_faces,), dtype=np.int32)
            changes.append(f'object[{object_idx}]: created mat_ids filled with 0 for {num_faces} faces')
            continue

        if isinstance(mat_ids, np.ndarray) and mat_ids.shape == (num_faces,) and np.issubdtype(mat_ids.dtype, np.integer):
            invalid_mask = (mat_ids < 0) | (mat_ids >= num_materials)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                remapped = mat_ids.astype(np.int32, copy=True)
                remapped[invalid_mask] = 0
                obj['mat_ids'] = remapped
                changes.append(
                    f'object[{object_idx}]: remapped {invalid_count} invalid mat_ids to 0 '
                    f'(valid range 0..{num_materials - 1})'
                )

    return changes


def validate_dump_for_volumetric_convert(dump):
    report = {
        'num_materials': len(dump.get('materials', [])),
        'summaries': [],
        'warnings': [],
        'errors': [],
    }

    objects = dump.get('objects', [])
    if len(objects) == 0:
        report['errors'].append('dump has no objects after normalization')
        return report

    for object_idx, obj in enumerate(objects):
        prefix = f'object[{object_idx}]'
        vertices = obj.get('vertices')
        faces = obj.get('faces')
        normals = obj.get('normals')
        uvs = obj.get('uvs')
        mat_ids = obj.get('mat_ids')

        report['summaries'].append(
            f'{prefix}: '
            f'vertices={_format_shape(vertices)} '
            f'faces={_format_shape(faces)} '
            f'normals={_format_shape(normals)} '
            f'uvs={_format_shape(uvs)} '
            f'mat_ids={_format_shape(mat_ids)}'
        )

        if not isinstance(vertices, np.ndarray) or vertices.ndim != 2 or vertices.shape[1] != 3:
            report['errors'].append(f'{prefix}: vertices must have shape [N, 3], got {_format_shape(vertices)}')
            continue
        if vertices.shape[0] == 0:
            report['errors'].append(f'{prefix}: vertices is empty')
        elif not np.isfinite(vertices).all():
            report['errors'].append(f'{prefix}: vertices contains NaN/Inf values')

        if not isinstance(faces, np.ndarray) or faces.ndim != 2 or faces.shape[1] != 3:
            report['errors'].append(f'{prefix}: faces must have shape [F, 3], got {_format_shape(faces)}')
            continue
        num_faces = faces.shape[0]
        if num_faces == 0:
            report['errors'].append(f'{prefix}: faces is empty')
        else:
            if np.issubdtype(faces.dtype, np.integer):
                face_min = int(faces.min())
                face_max = int(faces.max())
                if face_min < 0:
                    report['errors'].append(f'{prefix}: faces contains negative vertex index {face_min}')
                if face_max >= vertices.shape[0]:
                    report['errors'].append(
                        f'{prefix}: faces references vertex index {face_max} but only {vertices.shape[0]} vertices exist'
                    )
            else:
                report['errors'].append(f'{prefix}: faces dtype must be integer, got {faces.dtype}')

        if not isinstance(normals, np.ndarray) or normals.shape != (num_faces, 3, 3):
            report['errors'].append(
                f'{prefix}: normals must have shape {(num_faces, 3, 3)}, got {_format_shape(normals)}'
            )
        elif not np.isfinite(normals).all():
            report['errors'].append(f'{prefix}: normals contains NaN/Inf values')

        if uvs is not None:
            if not isinstance(uvs, np.ndarray) or uvs.shape != (num_faces, 3, 2):
                report['errors'].append(
                    f'{prefix}: uvs must have shape {(num_faces, 3, 2)} when present, got {_format_shape(uvs)}'
                )
            elif not np.isfinite(uvs).all():
                report['errors'].append(f'{prefix}: uvs contains NaN/Inf values')
        else:
            report['warnings'].append(f'{prefix}: uvs is None; converter will substitute zeros')

        if not isinstance(mat_ids, np.ndarray) or mat_ids.shape != (num_faces,):
            report['errors'].append(
                f'{prefix}: mat_ids must have shape {(num_faces,)}, got {_format_shape(mat_ids)}'
            )
        elif not np.issubdtype(mat_ids.dtype, np.integer):
            report['errors'].append(f'{prefix}: mat_ids dtype must be integer, got {mat_ids.dtype}')
        elif num_faces > 0:
            mat_min = int(mat_ids.min())
            mat_max = int(mat_ids.max())
            if mat_min < 0:
                report['errors'].append(
                    f'{prefix}: mat_ids contains negative value {mat_min}; native converter may segfault on this'
                )
            if report['num_materials'] == 0:
                report['errors'].append(f'{prefix}: dump has no materials but faces reference material ids')
            elif mat_max >= report['num_materials']:
                report['errors'].append(
                    f'{prefix}: mat_ids references material {mat_max} but dump only has {report["num_materials"]} materials'
                )

    return report


def normalize_dump(dump):
    dump = pickle.loads(pickle.dumps(dump))
    dump['objects'] = [
        obj for obj in dump['objects']
        if obj['vertices'].size != 0 and obj['faces'].size != 0
    ]
    if len(dump['objects']) == 0:
        return dump

    vertices = torch.from_numpy(
        np.concatenate([obj['vertices'] for obj in dump['objects']], axis=0)
    ).float()
    vertices_min = vertices.min(dim=0)[0]
    vertices_max = vertices.max(dim=0)[0]
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()

    for obj in dump['objects']:
        obj['vertices'] = ((torch.from_numpy(obj['vertices']).float() - center) * scale).numpy()

    return dump


def build_global_triangles(dump, filter_degenerate: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    vertices = []
    faces = []
    offset = 0
    for obj in dump['objects']:
        v = torch.from_numpy(obj['vertices']).float()
        f = torch.from_numpy(obj['faces']).long()
        if v.numel() == 0 or f.numel() == 0:
            continue
        vertices.append(v)
        faces.append(f + offset)
        offset += v.shape[0]

    if not vertices:
        return torch.empty((0, 3), dtype=torch.float32), torch.empty((0, 3), dtype=torch.long)

    vertices = torch.cat(vertices, dim=0)
    faces = torch.cat(faces, dim=0)
    if not filter_degenerate:
        return vertices, faces

    tri = vertices[faces]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    area2 = torch.linalg.norm(normals, dim=1)
    valid = torch.isfinite(area2) & (area2 > 1e-12)
    return vertices, faces[valid]


def voxel_coords_to_centers(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.float()
    return (coords + 0.5) / resolution - 0.5


def barycentric_coordinates(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-20)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


def closest_point_on_segment(
    points: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    bary_start: torch.Tensor,
    bary_end: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge = end - start
    denom = (edge * edge).sum(dim=-1).clamp_min(1e-20)
    t = ((points - start) * edge).sum(dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    closest = start + t[..., None] * edge
    bary = (1.0 - t[..., None]) * bary_start + t[..., None] * bary_end
    dist2 = ((points - closest) ** 2).sum(dim=-1)
    return closest, bary, dist2


def closest_point_on_triangles(
    points: torch.Tensor,
    tri_a: torch.Tensor,
    tri_b: torch.Tensor,
    tri_c: torch.Tensor,
    normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = points[:, None, :]
    a = tri_a[None, :, :]
    b = tri_b[None, :, :]
    c = tri_c[None, :, :]
    n = normals[None, :, :]

    n_norm2 = (n * n).sum(dim=-1).clamp_min(1e-20)
    signed = ((p - a) * n).sum(dim=-1) / n_norm2
    projected = p - signed[..., None] * n

    bary_projected = barycentric_coordinates(projected, a, b, c)
    inside = (bary_projected >= -1e-6).all(dim=-1)

    edge_ab_closest, edge_ab_bary, edge_ab_dist2 = closest_point_on_segment(
        p,
        a,
        b,
        torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype),
        torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype),
    )
    edge_bc_closest, edge_bc_bary, edge_bc_dist2 = closest_point_on_segment(
        p,
        b,
        c,
        torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype),
        torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype),
    )
    edge_ca_closest, edge_ca_bary, edge_ca_dist2 = closest_point_on_segment(
        p,
        c,
        a,
        torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype),
        torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype),
    )

    edge_dist2 = torch.stack([edge_ab_dist2, edge_bc_dist2, edge_ca_dist2], dim=-1)
    edge_idx = edge_dist2.argmin(dim=-1)
    edge_closest = torch.stack([edge_ab_closest, edge_bc_closest, edge_ca_closest], dim=-2)
    edge_bary = torch.stack([edge_ab_bary, edge_bc_bary, edge_ca_bary], dim=-2)

    gather_idx_3 = edge_idx[..., None, None].expand(-1, -1, 1, 3)
    closest_edge = edge_closest.gather(dim=-2, index=gather_idx_3).squeeze(-2)
    bary_edge = edge_bary.gather(dim=-2, index=gather_idx_3).squeeze(-2)
    dist2_edge = edge_dist2.gather(dim=-1, index=edge_idx[..., None]).squeeze(-1)

    dist2_projected = ((p - projected) ** 2).sum(dim=-1)
    closest = torch.where(inside[..., None], projected, closest_edge)
    bary = torch.where(inside[..., None], bary_projected, bary_edge)
    dist2 = torch.where(inside, dist2_projected, dist2_edge)
    return closest, bary, dist2


def closest_point_on_candidate_triangles(
    points: torch.Tensor,
    tri_a: torch.Tensor,
    tri_b: torch.Tensor,
    tri_c: torch.Tensor,
    normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = points[:, None, :]
    a = tri_a
    b = tri_b
    c = tri_c
    n = normals

    n_norm2 = (n * n).sum(dim=-1).clamp_min(1e-20)
    signed = ((p - a) * n).sum(dim=-1) / n_norm2
    projected = p - signed[..., None] * n

    bary_projected = barycentric_coordinates(projected, a, b, c)
    inside = (bary_projected >= -1e-6).all(dim=-1)

    edge_ab_closest, edge_ab_bary, edge_ab_dist2 = closest_point_on_segment(
        p,
        a,
        b,
        torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype),
        torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype),
    )
    edge_bc_closest, edge_bc_bary, edge_bc_dist2 = closest_point_on_segment(
        p,
        b,
        c,
        torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype),
        torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype),
    )
    edge_ca_closest, edge_ca_bary, edge_ca_dist2 = closest_point_on_segment(
        p,
        c,
        a,
        torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype),
        torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype),
    )

    edge_dist2 = torch.stack([edge_ab_dist2, edge_bc_dist2, edge_ca_dist2], dim=-1)
    edge_idx = edge_dist2.argmin(dim=-1)
    edge_closest = torch.stack([edge_ab_closest, edge_bc_closest, edge_ca_closest], dim=-2)
    edge_bary = torch.stack([edge_ab_bary, edge_bc_bary, edge_ca_bary], dim=-2)

    gather_idx_3 = edge_idx[..., None, None].expand(-1, -1, 1, 3)
    closest_edge = edge_closest.gather(dim=-2, index=gather_idx_3).squeeze(-2)
    bary_edge = edge_bary.gather(dim=-2, index=gather_idx_3).squeeze(-2)
    dist2_edge = edge_dist2.gather(dim=-1, index=edge_idx[..., None]).squeeze(-1)

    dist2_projected = ((p - projected) ** 2).sum(dim=-1)
    closest = torch.where(inside[..., None], projected, closest_edge)
    bary = torch.where(inside[..., None], bary_projected, bary_edge)
    dist2 = torch.where(inside, dist2_projected, dist2_edge)
    return closest, bary, dist2


def closest_point_from_plane_barycentric_candidates(
    points: torch.Tensor,
    tri_a: torch.Tensor,
    tri_b: torch.Tensor,
    tri_c: torch.Tensor,
    plane_barycentric: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = points[:, None, :]
    a = tri_a
    b = tri_b
    c = tri_c
    bary_plane = plane_barycentric[..., :3]
    projected = (
        bary_plane[..., 0:1] * a
        + bary_plane[..., 1:2] * b
        + bary_plane[..., 2:3] * c
    )
    inside = (bary_plane >= -1e-6).all(dim=-1)

    edge_bc_closest, edge_bc_bary, edge_bc_dist2 = closest_point_on_segment(
        p,
        b,
        c,
        torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype),
        torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype),
    )
    edge_ca_closest, edge_ca_bary, edge_ca_dist2 = closest_point_on_segment(
        p,
        c,
        a,
        torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype),
        torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype),
    )
    edge_ab_closest, edge_ab_bary, edge_ab_dist2 = closest_point_on_segment(
        p,
        a,
        b,
        torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype),
        torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype),
    )

    outside_edge_mask = torch.stack([
        bary_plane[..., 0] < 0.0,
        bary_plane[..., 1] < 0.0,
        bary_plane[..., 2] < 0.0,
    ], dim=-1)
    edge_dist2 = torch.stack([edge_bc_dist2, edge_ca_dist2, edge_ab_dist2], dim=-1)
    edge_dist2 = torch.where(outside_edge_mask, edge_dist2, torch.full_like(edge_dist2, float('inf')))
    edge_idx = edge_dist2.argmin(dim=-1)
    edge_closest = torch.stack([edge_bc_closest, edge_ca_closest, edge_ab_closest], dim=-2)
    edge_bary = torch.stack([edge_bc_bary, edge_ca_bary, edge_ab_bary], dim=-2)

    gather_idx_3 = edge_idx[..., None, None].expand(-1, -1, 1, 3)
    closest_edge = edge_closest.gather(dim=-2, index=gather_idx_3).squeeze(-2)
    bary_edge = edge_bary.gather(dim=-2, index=gather_idx_3).squeeze(-2)
    dist2_edge = edge_dist2.gather(dim=-1, index=edge_idx[..., None]).squeeze(-1)

    dist2_projected = ((p - projected) ** 2).sum(dim=-1)
    closest = torch.where(inside[..., None], projected, closest_edge)
    bary = torch.where(inside[..., None], bary_plane, bary_edge)
    dist2 = torch.where(inside, dist2_projected, dist2_edge)
    return closest, bary, dist2


def assemble_triangle_features(
    points: torch.Tensor,
    tri_a: torch.Tensor,
    tri_b: torch.Tensor,
    tri_c: torch.Tensor,
    centroid: torch.Tensor,
    normal: torch.Tensor,
    closest: torch.Tensor,
    bary: torch.Tensor,
) -> torch.Tensor:
    bary_clamped = bary.clamp(0.0, 1.0)
    d_tri = (3.0 * bary_clamped.min(dim=1).values).clamp(0.0, 1.0)
    d_vert = ((bary_clamped.max(dim=1).values - (1.0 / 3.0)) / (2.0 / 3.0)).clamp(0.0, 1.0)

    offsets = torch.cat([
        tri_a - closest,
        tri_b - closest,
        tri_c - closest,
    ], dim=1)
    offset_to_centroid = centroid - closest
    offset_to_projection = closest - points
    features = torch.cat([
        d_tri[:, None],
        d_vert[:, None],
        offsets,
        offset_to_centroid,
        normal,
        offset_to_projection,
    ], dim=1)
    return features


def nearest_triangle_features(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    device: torch.device,
    point_batch: int,
    triangle_batch: int,
) -> np.ndarray:
    if points.numel() == 0:
        return np.zeros((0, 20), dtype=np.float32)
    if vertices.numel() == 0 or faces.numel() == 0:
        raise ValueError('mesh has no valid triangles')

    vertices = vertices.to(device, non_blocking=True)
    faces = faces.to(device, non_blocking=True)
    tri = vertices[faces]
    tri_a_all = tri[:, 0]
    tri_b_all = tri[:, 1]
    tri_c_all = tri[:, 2]
    normals_all = torch.cross(tri_b_all - tri_a_all, tri_c_all - tri_a_all, dim=1)
    normals_all = normals_all / torch.linalg.norm(normals_all, dim=1, keepdim=True).clamp_min(1e-12)
    centroids_all = (tri_a_all + tri_b_all + tri_c_all) / 3.0

    features = []
    for p0 in range(0, points.shape[0], point_batch):
        p = points[p0:p0 + point_batch].to(device, non_blocking=True)
        best_dist2 = torch.full((p.shape[0],), float('inf'), device=device)
        best_tri = torch.full((p.shape[0],), -1, dtype=torch.long, device=device)
        best_closest = torch.zeros((p.shape[0], 3), dtype=p.dtype, device=device)
        best_bary = torch.zeros((p.shape[0], 3), dtype=p.dtype, device=device)

        for t0 in range(0, faces.shape[0], triangle_batch):
            t1 = min(t0 + triangle_batch, faces.shape[0])
            closest, bary, dist2 = closest_point_on_triangles(
                p,
                tri_a_all[t0:t1],
                tri_b_all[t0:t1],
                tri_c_all[t0:t1],
                normals_all[t0:t1],
            )
            local_best_dist2, local_idx = dist2.min(dim=1)
            update = local_best_dist2 < best_dist2
            if update.any():
                gather_idx = local_idx[:, None, None].expand(-1, 1, 3)
                best_dist2[update] = local_best_dist2[update]
                best_tri[update] = local_idx[update] + t0
                best_closest[update] = closest.gather(dim=1, index=gather_idx).squeeze(1)[update]
                best_bary[update] = bary.gather(dim=1, index=gather_idx).squeeze(1)[update]

        if (best_tri < 0).any():
            raise RuntimeError('nearest triangle search failed for at least one point')

        tri_a = tri_a_all[best_tri]
        tri_b = tri_b_all[best_tri]
        tri_c = tri_c_all[best_tri]
        centroid = centroids_all[best_tri]
        normal = normals_all[best_tri]

        feats = assemble_triangle_features(p, tri_a, tri_b, tri_c, centroid, normal, best_closest, best_bary)
        features.append(feats.cpu().numpy().astype(np.float32))

    return np.concatenate(features, axis=0)


def triangle_features_from_native_candidates(
    coords: torch.Tensor,
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    candidate_offsets: torch.Tensor,
    candidate_triangle_ids: torch.Tensor,
    candidate_barycentric: torch.Tensor,
    device: torch.device,
    point_batch: int,
    projection_mode: str,
) -> Tuple[np.ndarray, dict]:
    if points.numel() == 0:
        return np.zeros((0, 20), dtype=np.float32), {
            'native_empty_candidates': 0,
            'native_mean_candidates': 0.0,
            'native_max_candidates': 0,
            'native_inside_fallback_points': 0,
        }
    if coords.shape[0] != points.shape[0]:
        raise ValueError(f'coords/points length mismatch: {coords.shape[0]} vs {points.shape[0]}')
    if candidate_offsets.shape[0] != points.shape[0] + 1:
        raise ValueError(
            f'candidate_offsets length must be num_voxels + 1, got {candidate_offsets.shape[0]} for {points.shape[0]} voxels'
        )

    counts = (candidate_offsets[1:] - candidate_offsets[:-1]).cpu().numpy()
    empty = int((counts == 0).sum())
    if empty > 0:
        raise RuntimeError(f'native candidate map returned {empty} voxels with no candidates')
    if candidate_triangle_ids.numel() > 0:
        min_tri_id = int(candidate_triangle_ids.min().item())
        max_tri_id = int(candidate_triangle_ids.max().item())
        if min_tri_id < 0 or max_tri_id >= faces.shape[0]:
            raise RuntimeError(
                f'native candidate triangle ids are out of bounds for the feature triangle table: '
                f'min={min_tri_id}, max={max_tri_id}, num_triangles={faces.shape[0]}. '
                f'This usually means candidate IDs and face ordering are not aligned.'
            )

    vertices = vertices.to(device, non_blocking=True)
    faces = faces.to(device, non_blocking=True)
    candidate_offsets = candidate_offsets.to(device, non_blocking=True)
    candidate_triangle_ids = candidate_triangle_ids.long().to(device, non_blocking=True)
    candidate_barycentric = candidate_barycentric.to(device, non_blocking=True)

    tri = vertices[faces]
    tri_a_all = tri[:, 0]
    tri_b_all = tri[:, 1]
    tri_c_all = tri[:, 2]
    normals_all = torch.cross(tri_b_all - tri_a_all, tri_c_all - tri_a_all, dim=1)
    normals_all = normals_all / torch.linalg.norm(normals_all, dim=1, keepdim=True).clamp_min(1e-12)
    centroids_all = (tri_a_all + tri_b_all + tri_c_all) / 3.0

    features = np.zeros((points.shape[0], 20), dtype=np.float32)
    subtimings = {
        'native_setup_s': 0.0,
        'native_projection_s': 0.0,
        'native_select_s': 0.0,
        'native_assemble_copy_s': 0.0,
    }
    inside_fallback_points = 0

    for p0 in range(0, points.shape[0], point_batch):
        p1 = min(p0 + point_batch, points.shape[0])
        stage_t = time.perf_counter()
        starts = candidate_offsets[p0:p1]
        ends = candidate_offsets[p0 + 1:p1 + 1]
        local_counts = ends - starts
        max_candidates = int(local_counts.max().item())
        range_idx = torch.arange(max_candidates, device=device).reshape(1, -1)
        candidate_positions = starts.reshape(-1, 1) + range_idx
        valid_mask = range_idx < local_counts.reshape(-1, 1)
        safe_positions = torch.where(valid_mask, candidate_positions, starts.reshape(-1, 1))

        tri_ids = candidate_triangle_ids[safe_positions]
        plane_barycentric = candidate_barycentric[safe_positions]
        tri_a = tri_a_all[tri_ids]
        tri_b = tri_b_all[tri_ids]
        tri_c = tri_c_all[tri_ids]

        p = points[p0:p1].to(device, non_blocking=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        subtimings['native_setup_s'] += time.perf_counter() - stage_t

        stage_t = time.perf_counter()
        if projection_mode == 'exact':
            closest, bary, dist2 = closest_point_from_plane_barycentric_candidates(
                p,
                tri_a,
                tri_b,
                tri_c,
                plane_barycentric,
            )
        elif projection_mode == 'inside_barycentric':
            bary = plane_barycentric[..., :3]
            closest = (
                bary[..., 0:1] * tri_a
                + bary[..., 1:2] * tri_b
                + bary[..., 2:3] * tri_c
            )
            dist2 = ((p[:, None, :] - closest) ** 2).sum(dim=-1)
            dist2 = torch.where(
                (bary >= -1e-6).all(dim=-1),
                dist2,
                torch.full_like(dist2, float('inf')),
            )
        else:
            raise ValueError(f'Unsupported projection_mode: {projection_mode}')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        subtimings['native_projection_s'] += time.perf_counter() - stage_t

        stage_t = time.perf_counter()
        dist2 = torch.where(valid_mask, dist2, torch.full_like(dist2, float('inf')))
        best_dist2, best_local = dist2.min(dim=1)
        needs_fallback = torch.isinf(best_dist2)
        if needs_fallback.any():
            if projection_mode != 'inside_barycentric':
                raise RuntimeError('native candidate list produced no valid triangle for at least one point')
            inside_fallback_points += int(needs_fallback.sum().item())
            fallback_closest, fallback_bary, fallback_dist2 = closest_point_from_plane_barycentric_candidates(
                p,
                tri_a,
                tri_b,
                tri_c,
                plane_barycentric,
            )
            fallback_dist2 = torch.where(valid_mask, fallback_dist2, torch.full_like(fallback_dist2, float('inf')))
            fallback_best_dist2, fallback_best_local = fallback_dist2.min(dim=1)
            if torch.isinf(fallback_best_dist2).any():
                raise RuntimeError('native exact fallback produced no valid triangle for at least one point')
            closest = torch.where(needs_fallback[:, None, None], fallback_closest, closest)
            bary = torch.where(needs_fallback[:, None, None], fallback_bary, bary)
            dist2 = torch.where(needs_fallback[:, None], fallback_dist2, dist2)
            best_dist2 = torch.where(needs_fallback, fallback_best_dist2, best_dist2)
            best_local = torch.where(needs_fallback, fallback_best_local, best_local)

        gather_idx = best_local[:, None, None].expand(-1, 1, 3)
        best_closest = closest.gather(dim=1, index=gather_idx).squeeze(1)
        best_bary = bary.gather(dim=1, index=gather_idx).squeeze(1)
        best_tri = tri_ids.gather(dim=1, index=best_local[:, None]).squeeze(1)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        subtimings['native_select_s'] += time.perf_counter() - stage_t

        stage_t = time.perf_counter()
        tri_a = tri_a_all[best_tri]
        tri_b = tri_b_all[best_tri]
        tri_c = tri_c_all[best_tri]
        centroid = centroids_all[best_tri]
        normal = normals_all[best_tri]

        feats = assemble_triangle_features(p, tri_a, tri_b, tri_c, centroid, normal, best_closest, best_bary)
        features[p0:p1] = feats.cpu().numpy().astype(np.float32)
        subtimings['native_assemble_copy_s'] += time.perf_counter() - stage_t

    return features, {
        'native_empty_candidates': empty,
        'native_mean_candidates': float(counts.mean()) if counts.size else 0.0,
        'native_max_candidates': int(counts.max()) if counts.size else 0,
        'native_inside_fallback_points': inside_fallback_points,
        **subtimings,
    }


def coords_to_linear_np(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = coords.astype(np.int64, copy=False)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def build_aabb_candidate_lists(
    coords: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    padding: int,
):
    coords_np = coords.cpu().numpy().astype(np.int32, copy=False)
    sparse_linear = coords_to_linear_np(coords_np, resolution)
    sparse_lookup = {int(key): idx for idx, key in enumerate(sparse_linear.tolist())}
    candidate_lists = [[] for _ in range(coords_np.shape[0])]

    tri_np = vertices[faces].cpu().numpy().astype(np.float32, copy=False)
    tri_min = tri_np.min(axis=1)
    tri_max = tri_np.max(axis=1)
    min_idx = np.floor((tri_min + 0.5) * resolution).astype(np.int32) - padding
    max_idx = np.floor((tri_max + 0.5) * resolution).astype(np.int32) + padding
    min_idx = np.clip(min_idx, 0, resolution - 1)
    max_idx = np.clip(max_idx, 0, resolution - 1)

    for tri_idx, (lo, hi) in enumerate(zip(min_idx, max_idx)):
        for x in range(int(lo[0]), int(hi[0]) + 1):
            base_x = x * resolution * resolution
            for y in range(int(lo[1]), int(hi[1]) + 1):
                base_xy = base_x + y * resolution
                for z in range(int(lo[2]), int(hi[2]) + 1):
                    row = sparse_lookup.get(base_xy + z)
                    if row is not None:
                        candidate_lists[row].append(tri_idx)

    counts = np.array([len(c) for c in candidate_lists], dtype=np.int32)
    stats = {
        'aabb_empty_candidates': int((counts == 0).sum()),
        'aabb_mean_candidates': float(counts.mean()) if counts.size else 0.0,
        'aabb_max_candidates': int(counts.max()) if counts.size else 0,
    }
    return candidate_lists, stats


def triangle_features_from_candidate_lists(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    candidate_lists,
    device: torch.device,
    point_batch: int,
    fallback: str,
) -> Tuple[np.ndarray, dict]:
    if points.numel() == 0:
        return np.zeros((0, 20), dtype=np.float32), {
            'aabb_fallback_points': 0,
        }
    if vertices.numel() == 0 or faces.numel() == 0:
        raise ValueError('mesh has no valid triangles')

    empty_indices = [idx for idx, c in enumerate(candidate_lists) if len(c) == 0]
    if empty_indices and fallback == 'error':
        raise RuntimeError(f'AABB candidate map missed {len(empty_indices)} sparse voxel(s)')

    vertices = vertices.to(device, non_blocking=True)
    faces = faces.to(device, non_blocking=True)
    tri = vertices[faces]
    tri_a_all = tri[:, 0]
    tri_b_all = tri[:, 1]
    tri_c_all = tri[:, 2]
    normals_all = torch.cross(tri_b_all - tri_a_all, tri_c_all - tri_a_all, dim=1)
    normals_all = normals_all / torch.linalg.norm(normals_all, dim=1, keepdim=True).clamp_min(1e-12)
    centroids_all = (tri_a_all + tri_b_all + tri_c_all) / 3.0

    features = np.zeros((points.shape[0], 20), dtype=np.float32)

    for p0 in range(0, points.shape[0], point_batch):
        p1 = min(p0 + point_batch, points.shape[0])
        local_candidates = candidate_lists[p0:p1]
        nonempty_local = [idx for idx, candidates in enumerate(local_candidates) if len(candidates) > 0]
        if not nonempty_local:
            continue

        batch_rows = np.array(nonempty_local, dtype=np.int64)
        max_candidates = max(len(local_candidates[idx]) for idx in nonempty_local)
        candidate_idx_np = np.full((len(nonempty_local), max_candidates), -1, dtype=np.int64)
        for row_idx, local_idx in enumerate(nonempty_local):
            candidates = local_candidates[local_idx]
            candidate_idx_np[row_idx, :len(candidates)] = candidates

        p = points[p0:p1][batch_rows].to(device, non_blocking=True)
        candidate_idx = torch.from_numpy(candidate_idx_np).to(device, non_blocking=True)
        valid_mask = candidate_idx >= 0
        safe_idx = candidate_idx.clamp_min(0)

        tri_a = tri_a_all[safe_idx]
        tri_b = tri_b_all[safe_idx]
        tri_c = tri_c_all[safe_idx]
        normals = normals_all[safe_idx]

        closest, bary, dist2 = closest_point_on_candidate_triangles(p, tri_a, tri_b, tri_c, normals)
        dist2 = torch.where(valid_mask, dist2, torch.full_like(dist2, float('inf')))
        best_dist2, best_local = dist2.min(dim=1)
        if torch.isinf(best_dist2).any():
            raise RuntimeError('candidate list produced no valid nearest triangle for at least one point')

        gather_idx = best_local[:, None, None].expand(-1, 1, 3)
        best_closest = closest.gather(dim=1, index=gather_idx).squeeze(1)
        best_bary = bary.gather(dim=1, index=gather_idx).squeeze(1)
        best_tri = candidate_idx.gather(dim=1, index=best_local[:, None]).squeeze(1)

        tri_a = tri_a_all[best_tri]
        tri_b = tri_b_all[best_tri]
        tri_c = tri_c_all[best_tri]
        centroid = centroids_all[best_tri]
        normal = normals_all[best_tri]

        feats = assemble_triangle_features(p, tri_a, tri_b, tri_c, centroid, normal, best_closest, best_bary)
        global_rows = p0 + batch_rows
        features[global_rows] = feats.cpu().numpy().astype(np.float32)

    if empty_indices:
        fallback_points = points[empty_indices]
        fallback_features = nearest_triangle_features(
            fallback_points,
            vertices.cpu(),
            faces.cpu(),
            device=device,
            point_batch=point_batch,
            triangle_batch=opt.triangle_batch,
        )
        features[np.array(empty_indices, dtype=np.int64)] = fallback_features

    return features, {
        'aabb_fallback_points': len(empty_indices),
    }


def feature_dtype_to_numpy(feature_dtype: str):
    if feature_dtype == 'float16':
        return np.float16
    if feature_dtype == 'float32':
        return np.float32
    raise ValueError(f'Unsupported feature dtype: {feature_dtype}')


def build_metadata_json(resolution: int, feature_dtype: str, npz_compression: str) -> str:
    return json.dumps({
        'format': FORMAT_NAME,
        'version': FORMAT_VERSION,
        'resolution': resolution,
        'feature_dtype': feature_dtype,
        'npz_compression': npz_compression,
        'coordinate_frame': 'normalized_object_space',
        'normalization': 'same_as_voxelize_gaussian_distance.normalize_dump',
        'feature_layout': INPUT_LAYOUT,
        'definitions': {
            'voxel_center': '(coords + 0.5) / resolution - 0.5',
            'projected_point': 'closest point on nearest triangle',
            'd_tri': '3 * min(barycentric(projected_point))',
            'd_vert': '(max(barycentric(projected_point)) - 1/3) / (2/3)',
        },
    })


def compress_zstd(payload: bytes, level: int) -> bytes:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError('zstd triangle-field output requires the zstandard package') from exc
    return zstd.ZstdCompressor(level=level).compress(payload)


def decompress_zstd(payload: bytes) -> bytes:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError('zstd triangle-field input requires the zstandard package') from exc
    return zstd.ZstdDecompressor().decompress(payload)


@contextmanager
def load_triangle_field_npz(path: str):
    if path.endswith('.zst'):
        with open(path, 'rb') as f:
            payload = decompress_zstd(f.read())
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            yield data
    else:
        with np.load(path, allow_pickle=False) as data:
            yield data


def triangle_field_output_path(root: str, sha256: str, resolution: int, npz_compression: str) -> str:
    suffix = '.npz.zst' if npz_compression == 'zstd' else '.npz'
    return os.path.join(root, f'triangle_field_voxels_{resolution}', f'{sha256}{suffix}')


def matching_or_requested_output_path(root: str, sha256: str, resolution: int, feature_dtype: str, npz_compression: str) -> Tuple[str, bool]:
    requested = triangle_field_output_path(root, sha256, resolution, npz_compression)
    legacy = triangle_field_output_path(root, sha256, resolution, 'none')
    candidates = [requested]
    if legacy != requested:
        candidates.append(legacy)
    for path in candidates:
        if output_matches_expected_layout(path, resolution, feature_dtype):
            return path, False
    return requested, True


def save_triangle_field_npz(out_path: str, coords: torch.Tensor, features: np.ndarray, resolution: int):
    np_dtype = feature_dtype_to_numpy(opt.feature_dtype)
    coords_np = coords.cpu().numpy().astype(np.int32, copy=False)
    features_np = features.astype(np_dtype, copy=False)
    metadata_json = np.array(build_metadata_json(resolution, opt.feature_dtype, opt.npz_compression))

    if opt.npz_compression == 'compressed':
        tmp_path = out_path + '.tmp.npz'
        np.savez_compressed(
            tmp_path,
            coords=coords_np,
            features=features_np,
            metadata_json=metadata_json,
        )
    elif opt.npz_compression == 'none':
        tmp_path = out_path + '.tmp.npz'
        np.savez(
            tmp_path,
            coords=coords_np,
            features=features_np,
            metadata_json=metadata_json,
        )
    elif opt.npz_compression == 'zstd':
        tmp_path = out_path + '.tmp'
        buffer = io.BytesIO()
        np.savez(
            buffer,
            coords=coords_np,
            features=features_np,
            metadata_json=metadata_json,
        )
        with open(tmp_path, 'wb') as f:
            f.write(compress_zstd(buffer.getvalue(), opt.zstd_level))
    else:
        raise ValueError(f'Unsupported npz_compression: {opt.npz_compression}')
    os.replace(tmp_path, out_path)


def output_matches_expected_layout(out_path: str, resolution: int, feature_dtype: str) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        with load_triangle_field_npz(out_path) as data:
            if set(data.files) != {'coords', 'features', 'metadata_json'}:
                return False
            coords = data['coords']
            features = data['features']
            metadata = json.loads(str(data['metadata_json']))
            expected_dtype = feature_dtype_to_numpy(feature_dtype)
            return (
                coords.ndim == 2
                and coords.shape[1] == 3
                and features.ndim == 2
                and features.shape[1] == 20
                and features.shape[0] == coords.shape[0]
                and features.dtype == expected_dtype
                and metadata.get('format') == FORMAT_NAME
                and metadata.get('version') == FORMAT_VERSION
                and metadata.get('resolution') == resolution
                and metadata.get('feature_dtype') == feature_dtype
            )
    except Exception:
        return False


def read_num_voxels(out_path: str) -> int:
    with load_triangle_field_npz(out_path) as data:
        return int(data['coords'].shape[0])


def _triangle_field_voxelize(file, metadatum, pbr_dump_root, root, device):
    sha256 = metadatum['sha256']
    try:
        object_start_t = time.perf_counter()
        stage_timings = {
            'prep_s': 0.0,
            'voxelize_s': 0.0,
            'coords_to_points_s': 0.0,
            'candidate_s': 0.0,
            'nearest_triangle_s': 0.0,
            'write_s': 0.0,
        }
        debug_log(f'{sha256}: start object')
        pack = {'sha256': sha256}
        dump = None
        vertices = None
        faces = None

        for res in opt.resolution:
            debug_log(f'{sha256}: resolution={res} start')
            out_path, need_process = matching_or_requested_output_path(
                root,
                sha256,
                res,
                opt.feature_dtype,
                opt.npz_compression,
            )

            if not need_process:
                num_voxels = read_num_voxels(out_path)
                pack[f'triangle_field_voxelized_{res}'] = True
                pack[f'num_triangle_field_voxels_{res}'] = num_voxels
                debug_log(f'{sha256}: resolution={res} skip done num_voxel={num_voxels}')
                continue

            if dump is None:
                prep_start_t = time.perf_counter()
                debug_log(f'{sha256}: loading pbr dump')
                with open(os.path.join(pbr_dump_root, 'pbr_dumps', f'{sha256}.pickle'), 'rb') as f:
                    dump = pickle.load(f)
                debug_log(f'{sha256}: normalizing dump')
                dump = normalize_dump(dump)
                sanitize_changes = sanitize_dump_for_volumetric_convert(dump)
                for change in sanitize_changes:
                    debug_log(f'{sha256}: sanitize: {change}')
                validation = validate_dump_for_volumetric_convert(dump)
                for warning in validation['warnings']:
                    debug_log(f'{sha256}: validation warning: {warning}')
                if validation['errors']:
                    raise ValueError('volumetric validation failed: ' + '; '.join(validation['errors'][:8]))
                debug_log(f'{sha256}: building global triangles')
                vertices, faces = build_global_triangles(
                    dump,
                    filter_degenerate=(opt.candidate_source != 'native'),
                )
                debug_log(f'{sha256}: mesh stats vertices={vertices.shape[0]} triangles={faces.shape[0]}')
                if vertices.numel() == 0 or faces.numel() == 0:
                    raise ValueError('mesh has no valid non-degenerate triangles')
                stage_timings['prep_s'] += time.perf_counter() - prep_start_t

            debug_log(f'{sha256}: resolution={res} voxelizing active coords')
            stage_start_t = time.perf_counter()
            if opt.candidate_source == 'native':
                coords, candidate_offsets, candidate_triangle_ids, candidate_barycentric = (
                    o_voxel.convert.blender_dump_to_voxel_triangle_candidates(
                        dump,
                        grid_size=res,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        verbose=False,
                        timing=False,
                    )
                )
                stage_timings['voxelize_s'] += time.perf_counter() - stage_start_t
            else:
                coords, _ = o_voxel.convert.blender_dump_to_volumetric_attr(
                    dump,
                    grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    mip_level_offset=0,
                    verbose=False,
                    timing=False,
                )
                candidate_offsets = None
                candidate_triangle_ids = None
                candidate_barycentric = None
                stage_timings['voxelize_s'] += time.perf_counter() - stage_start_t
            debug_log(f'{sha256}: resolution={res} voxelized active coords count={len(coords)}')

            if len(coords) == 0:
                empty_features = np.zeros((0, 20), dtype=np.float32)
                stage_start_t = time.perf_counter()
                save_triangle_field_npz(out_path, coords, empty_features, res)
                stage_timings['write_s'] += time.perf_counter() - stage_start_t
                pack[f'triangle_field_voxelized_{res}'] = True
                pack[f'num_triangle_field_voxels_{res}'] = 0
                debug_log(f'{sha256}: resolution={res} empty npz write done')
                continue

            debug_log(f'{sha256}: resolution={res} converting active coords to voxel centers')
            stage_start_t = time.perf_counter()
            points = voxel_coords_to_centers(coords, res)
            stage_timings['coords_to_points_s'] += time.perf_counter() - stage_start_t

            if opt.candidate_source == 'native':
                debug_log(f'{sha256}: resolution={res} computing triangle features from native candidates')
                stage_start_t = time.perf_counter()
                features, native_stats = triangle_features_from_native_candidates(
                    coords=coords,
                    points=points,
                    vertices=vertices,
                    faces=faces,
                    candidate_offsets=candidate_offsets,
                    candidate_triangle_ids=candidate_triangle_ids,
                    candidate_barycentric=candidate_barycentric,
                    device=device,
                    point_batch=opt.point_batch,
                    projection_mode=opt.projection_mode,
                )
                stage_timings['nearest_triangle_s'] += time.perf_counter() - stage_start_t
                pack[f'native_empty_candidates_{res}'] = native_stats['native_empty_candidates']
                pack[f'native_mean_candidates_{res}'] = native_stats['native_mean_candidates']
                pack[f'native_max_candidates_{res}'] = native_stats['native_max_candidates']
                pack[f'native_inside_fallback_points_{res}'] = native_stats['native_inside_fallback_points']
                pack[f'native_setup_s_{res}'] = native_stats['native_setup_s']
                pack[f'native_projection_s_{res}'] = native_stats['native_projection_s']
                pack[f'native_select_s_{res}'] = native_stats['native_select_s']
                pack[f'native_assemble_copy_s_{res}'] = native_stats['native_assemble_copy_s']
            elif opt.candidate_source == 'global':
                debug_log(
                    f'{sha256}: resolution={res} computing triangle features from all triangles '
                    f'point_batch={opt.point_batch} triangle_batch={opt.triangle_batch}'
                )
                stage_start_t = time.perf_counter()
                features = nearest_triangle_features(
                    points=points,
                    vertices=vertices,
                    faces=faces,
                    device=device,
                    point_batch=opt.point_batch,
                    triangle_batch=opt.triangle_batch,
                )
                stage_timings['nearest_triangle_s'] += time.perf_counter() - stage_start_t
            else:
                debug_log(
                    f'{sha256}: resolution={res} building AABB candidate map '
                    f'aabb_padding={opt.aabb_padding}'
                )
                stage_start_t = time.perf_counter()
                candidate_lists, candidate_stats = build_aabb_candidate_lists(
                    coords=coords,
                    vertices=vertices,
                    faces=faces,
                    resolution=res,
                    padding=opt.aabb_padding,
                )
                stage_timings['candidate_s'] += time.perf_counter() - stage_start_t
                debug_log(
                    f'{sha256}: resolution={res} candidate map done '
                    f'empty={candidate_stats["aabb_empty_candidates"]} '
                    f'mean={candidate_stats["aabb_mean_candidates"]:.2f} '
                    f'max={candidate_stats["aabb_max_candidates"]}'
                )

                debug_log(
                    f'{sha256}: resolution={res} computing triangle features from candidates '
                    f'point_batch={opt.point_batch} fallback={opt.aabb_fallback}'
                )
                stage_start_t = time.perf_counter()
                features, exact_stats = triangle_features_from_candidate_lists(
                    points=points,
                    vertices=vertices,
                    faces=faces,
                    candidate_lists=candidate_lists,
                    device=device,
                    point_batch=opt.point_batch,
                    fallback=opt.aabb_fallback,
                )
                stage_timings['nearest_triangle_s'] += time.perf_counter() - stage_start_t
                pack[f'aabb_empty_candidates_{res}'] = candidate_stats['aabb_empty_candidates']
                pack[f'aabb_mean_candidates_{res}'] = candidate_stats['aabb_mean_candidates']
                pack[f'aabb_max_candidates_{res}'] = candidate_stats['aabb_max_candidates']
                pack[f'aabb_fallback_points_{res}'] = exact_stats['aabb_fallback_points']

            debug_log(f'{sha256}: resolution={res} writing npz {out_path}')
            stage_start_t = time.perf_counter()
            save_triangle_field_npz(out_path, coords, features, res)
            stage_timings['write_s'] += time.perf_counter() - stage_start_t
            pack[f'triangle_field_voxelized_{res}'] = True
            pack[f'num_triangle_field_voxels_{res}'] = len(coords)
            debug_log(f'{sha256}: resolution={res} write done num_voxel={len(coords)}')

        if BENCHMARK_ENABLED:
            total_time_s = time.perf_counter() - object_start_t
            pack['benchmark_total_s'] = total_time_s
            for key, value in stage_timings.items():
                pack[f'benchmark_{key}'] = value
            benchmark_log(
                f'{sha256}: total={total_time_s:.3f}s '
                + ' '.join(f'{key}={value:.3f}s' for key, value in stage_timings.items())
                + ' '
                + ' '.join(
                    f'{key}={pack[key]}'
                    for key in sorted(pack)
                    if (
                        (key.startswith('aabb_') or key.startswith('native_'))
                        and key.endswith(tuple(str(res) for res in opt.resolution))
                    )
                )
            )
        debug_log(f'{sha256}: object done')
        return pack
    except Exception as e:
        print(f'Error voxelizing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save/load metadata')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to load pbr dumps')
    parser.add_argument('--triangle_field_voxel_root', type=str, default=None,
                        help='Directory to save triangle-field voxel .npz files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    parser.add_argument('--feature_dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='Stored dtype for input/target feature arrays')
    parser.add_argument('--npz_compression', type=str, default='compressed', choices=['compressed', 'none', 'zstd'],
                        help='Use numpy compressed npz, uncompressed npz, or zstd-compressed npz.zst output')
    parser.add_argument('--zstd_level', type=int, default=3,
                        help='Compression level for --npz_compression zstd')
    parser.add_argument('--candidate_source', type=str, default='native', choices=['native', 'aabb', 'global'],
                        help='Use native o_voxel triangle candidates, Python AABB candidates, or all triangles')
    parser.add_argument('--projection_mode', type=str, default='inside_barycentric',
                        choices=['inside_barycentric', 'exact'],
                        help='Projection rule for native candidates')
    parser.add_argument('--point_batch', type=int, default=2048)
    parser.add_argument('--triangle_batch', type=int, default=4096)
    parser.add_argument('--aabb_padding', type=int, default=1,
                        help='Conservative voxel padding around each triangle AABB when building candidates')
    parser.add_argument('--aabb_fallback', type=str, default='global', choices=['global', 'error'],
                        help='Behavior when an active sparse voxel receives no AABB triangle candidates')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed per-object debug logging')
    parser.add_argument('--benchmark', action='store_true',
                        help='Print per-object timings and an aggregate timing summary')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.resolution = sorted([int(x) for x in opt.resolution.split(',')], reverse=True)
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.triangle_field_voxel_root = opt.triangle_field_voxel_root or opt.root
    DEBUG_VERBOSE = bool(opt.verbose)
    BENCHMARK_ENABLED = bool(opt.benchmark)

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.triangle_field_voxel_root, f'triangle_field_voxels_{res}', 'new_records'), exist_ok=True)

    metadata_path = os.path.join(opt.root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(metadata_path).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(
            pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256')
        )
    if os.path.exists(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(
            pd.read_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')).set_index('sha256')
        )
    for res in opt.resolution:
        voxel_meta_path = os.path.join(opt.triangle_field_voxel_root, f'triangle_field_voxels_{res}', 'metadata.csv')
        if os.path.exists(voxel_meta_path):
            voxel_meta = pd.read_csv(voxel_meta_path).set_index('sha256')
            voxel_meta = voxel_meta.rename(columns={
                'triangle_field_voxelized': f'triangle_field_voxelized_{res}',
                'num_triangle_field_voxels': f'num_triangle_field_voxels_{res}',
            })
            metadata = metadata.combine_first(voxel_meta)
    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['pbr_dumped'] == True]
        mask = np.zeros(len(metadata), dtype=bool)
        for res in opt.resolution:
            col = f'triangle_field_voxelized_{res}'
            if col in metadata.columns:
                mask |= metadata[col] != True
            else:
                mask[:] = True
                break
        metadata = metadata[mask]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    print(f'Processing {len(metadata)} objects...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Feature dtype: {opt.feature_dtype}')
    print(f'NPZ compression: {opt.npz_compression}')
    print(f'Projection mode: {opt.projection_mode}')
    if opt.npz_compression == 'zstd':
        print(f'Zstd level: {opt.zstd_level}')
    if len(metadata) > 0:
        debug_log(f'first sha256 in shard: {metadata.iloc[0]["sha256"]}')

    func = partial(
        _triangle_field_voxelize,
        pbr_dump_root=opt.pbr_dump_root,
        root=opt.triangle_field_voxel_root,
        device=device,
    )
    voxelized = dataset_utils.foreach_instance(
        metadata,
        None,
        func,
        max_workers=opt.max_workers,
        no_file=True,
        desc='Voxelizing triangle-field features',
    )

    if 'error' in voxelized.columns:
        errors = voxelized[voxelized['error'].notna()]
        if len(errors) > 0:
            with open('triangle_field_errors.txt', 'w') as f:
                f.write('\n'.join(errors['sha256'].tolist()))

    if opt.benchmark and len(voxelized) > 0 and 'benchmark_total_s' in voxelized.columns:
        benchmark_cols = [
            'benchmark_total_s',
            'benchmark_prep_s',
            'benchmark_voxelize_s',
            'benchmark_coords_to_points_s',
            'benchmark_nearest_triangle_s',
            'benchmark_write_s',
        ]
        benchmark_df = voxelized[[c for c in benchmark_cols if c in voxelized.columns]].dropna(how='all')
        if len(benchmark_df) > 0:
            means = benchmark_df.mean(numeric_only=True)
            totals = benchmark_df.sum(numeric_only=True)
            benchmark_log(
                f'aggregate over {len(benchmark_df)} object(s): '
                + ' '.join(f'{col} mean={means[col]:.3f}s total={totals[col]:.3f}s' for col in benchmark_df.columns)
            )

    for res in opt.resolution:
        col_ok = f'triangle_field_voxelized_{res}'
        col_n = f'num_triangle_field_voxels_{res}'
        if col_ok in voxelized.columns:
            voxel_meta = voxelized[voxelized[col_ok] == True]
            if len(voxel_meta) > 0:
                voxel_meta = voxel_meta[['sha256', col_ok, col_n]].rename(columns={
                    col_ok: 'triangle_field_voxelized',
                    col_n: 'num_triangle_field_voxels',
                })
                voxel_meta.to_csv(
                    os.path.join(
                        opt.triangle_field_voxel_root,
                        f'triangle_field_voxels_{res}',
                        'new_records',
                        f'part_{opt.rank}.csv',
                    ),
                    index=False,
                )
