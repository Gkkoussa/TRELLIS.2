"""Online surface-point sampling for point-density flow matching."""

import hashlib
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import trimesh

from .components import StandardDatasetBase


def load_normalized_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh and apply the triangle-field unit-extent convention."""
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f'Could not load a triangle mesh from {path}')

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    valid_faces = (
        (faces >= 0).all(axis=1)
        & (faces < len(vertices)).all(axis=1)
    )
    faces = faces[valid_faces]
    triangles = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    faces = faces[area2 > 1e-12]
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise ValueError(f'Mesh has no finite, non-degenerate triangles: {path}')

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extent = float((bounds_max - bounds_min).max())
    if extent <= 0:
        raise ValueError(f'Mesh has zero bounding-box extent: {path}')
    center = (bounds_min + bounds_max) / 2.0
    vertices = (vertices - center) * (0.99999 / extent)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)


def sample_surface_field(
    mesh: trimesh.Trimesh,
    count: int,
    *,
    field_name: str,
    resolution: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points, face normals, and a barycentrically interpolated field."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    face_areas = 0.5 * area2
    face_normals = cross / area2[:, None]

    if field_name == 'density':
        face_values = face_areas
    elif field_name == 'elongation':
        edge_sq_sum = (
            np.square(triangles[:, 1] - triangles[:, 0]).sum(axis=1)
            + np.square(triangles[:, 2] - triangles[:, 1]).sum(axis=1)
            + np.square(triangles[:, 0] - triangles[:, 2]).sum(axis=1)
        )
        quality = np.clip(
            4.0 * np.sqrt(3.0) * face_areas / np.maximum(edge_sq_sum, 1e-20),
            1e-8,
            1.0,
        )
        face_values = -np.log(quality)
    else:
        raise ValueError(f'Unsupported point field: {field_name}')

    vertex_value_sum = np.zeros(len(vertices), dtype=np.float64)
    vertex_face_count = np.zeros(len(vertices), dtype=np.int64)
    for corner in range(3):
        np.add.at(vertex_value_sum, faces[:, corner], face_values)
        np.add.at(vertex_face_count, faces[:, corner], 1)
    vertex_mean_value = np.divide(
        vertex_value_sum,
        vertex_face_count,
        out=np.zeros_like(vertex_value_sum),
        where=vertex_face_count > 0,
    )

    points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=seed)
    bary = trimesh.triangles.points_to_barycentric(
        triangles[face_indices],
        points,
    )
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-8)
    field = (
        vertex_mean_value[faces[face_indices]] * bary
    ).sum(axis=1, keepdims=True)
    if field_name == 'density':
        voxel_size = 1.0 / float(resolution)
        field = -np.log(
            np.maximum(field, 1e-12) / (voxel_size * voxel_size) + 1e-8
        )
    return (
        points.astype(np.float32),
        face_normals[face_indices].astype(np.float32),
        field.astype(np.float32),
    )


def sample_surface_density(
    mesh: trimesh.Trimesh,
    count: int,
    *,
    resolution: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible density sampler."""
    return sample_surface_field(
        mesh,
        count,
        field_name='density',
        resolution=resolution,
        seed=seed,
    )


def render_point_density(
    points: torch.Tensor,
    density: torch.Tensor,
    image_size: int = 256,
) -> torch.Tensor:
    """Create a lightweight orthographic density snapshot."""
    batch_size = points.shape[0]
    images = torch.full(
        (batch_size, 3, image_size, image_size),
        0.08,
        device=points.device,
        dtype=torch.float32,
    )
    values = torch.sigmoid(density[..., 0].float())
    for batch_index in range(batch_size):
        xy = ((points[batch_index, :, :2].float() + 0.5) * (image_size - 1)).round().long()
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < image_size)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < image_size)
        )
        order = torch.argsort(points[batch_index, :, 2].float())
        order = order[valid[order]]
        xy_ordered = xy[order]
        value = values[batch_index, order]
        color = torch.stack([
            value,
            0.2 + 0.6 * (1.0 - (2.0 * value - 1.0).abs()),
            1.0 - value,
        ], dim=0)
        images[batch_index, :, image_size - 1 - xy_ordered[:, 1], xy_ordered[:, 0]] = color
    return images


class PointDensityMeshDataset(StandardDatasetBase):
    """Filtered mesh dataset with efficient online P/Q surface sampling."""

    def __init__(
        self,
        roots: str,
        *,
        context_points: int = 81920,
        query_points: int = 16384,
        density_resolution: int = 512,
        density_stats_path: str,
        field_name: str = 'density',
        min_aesthetic_score: float = None,
        deterministic_sampling: bool = False,
        sampling_seed: int = 0,
        include_query_normals: bool = True,
    ):
        self.context_points = int(context_points)
        self.query_points = int(query_points)
        self.density_resolution = int(density_resolution)
        if field_name not in ('density', 'elongation'):
            raise ValueError(f'Unsupported point field: {field_name}')
        self.field_name = field_name
        self.min_aesthetic_score = min_aesthetic_score
        self.deterministic_sampling = bool(deterministic_sampling)
        self.sampling_seed = int(sampling_seed)
        self.include_query_normals = bool(include_query_normals)
        density_stats_path = os.path.expandvars(density_stats_path)
        with open(density_stats_path, 'r') as file:
            stats = json.load(file)
        self.density_mean = float(stats['mean'])
        self.density_std = float(stats['std'])
        if not np.isfinite(self.density_std) or self.density_std <= 0:
            raise ValueError(f'Invalid density std in {density_stats_path}: {self.density_std}')
        self.density_stats_path = density_stats_path
        self.value_range = (0.0, 1.0)
        super().__init__(roots)

    def filter_metadata(self, metadata):
        stats: Dict[str, int] = {}
        metadata = metadata[metadata['local_path'].notna()]
        stats['Mesh available'] = len(metadata)
        if self.min_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats

    def _mesh_path(self, root, instance: str) -> str:
        local_path = str(self.metadata.loc[instance, 'local_path'])
        return local_path if os.path.isabs(local_path) else os.path.join(root['mesh'], local_path)

    def _sample_seed(self, instance: str) -> int:
        if not self.deterministic_sampling:
            return int(np.random.randint(0, np.iinfo(np.int32).max))
        digest = hashlib.sha256(f'{self.sampling_seed}:{instance}'.encode()).digest()
        return int.from_bytes(digest[:4], byteorder='little', signed=False)

    def get_instance(self, root, instance: str):
        mesh = load_normalized_mesh(self._mesh_path(root, instance))
        total_points = self.context_points + self.query_points
        points, normals, density = sample_surface_field(
            mesh,
            total_points,
            field_name=self.field_name,
            resolution=self.density_resolution,
            seed=self._sample_seed(instance),
        )
        density = (density - self.density_mean) / self.density_std
        split = self.context_points
        sample = {
            'context_points': torch.from_numpy(points[:split]),
            'context_normals': torch.from_numpy(normals[:split]),
            'context_density': torch.from_numpy(density[:split]),
            'query_points': torch.from_numpy(points[split:]),
            'query_density': torch.from_numpy(density[split:]),
        }
        if self.include_query_normals:
            sample['query_normals'] = torch.from_numpy(normals[split:])
        return sample

    def visualize_sample(self, sample):
        return {
            'density': render_point_density(
                sample['query_points'],
                sample['query_density'],
            )
        }

    def __str__(self):
        return '\n'.join([
            super().__str__(),
            f'  - Context points: {self.context_points}',
            f'  - Query points: {self.query_points}',
            f'  - Target field: {self.field_name}',
            f'  - Density resolution: {self.density_resolution}',
            f'  - Density normalization: mean={self.density_mean}, std={self.density_std}',
            f'  - Include query normals: {self.include_query_normals}',
            f'  - Deterministic sampling: {self.deterministic_sampling}',
        ])
