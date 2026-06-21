import os
from typing import Union

import numpy as np
import torch
import o_voxel
import utils3d

from .components import StandardDatasetBase
from ..modules import sparse as sp
from ..renderers import VoxelRenderer
from ..representations import Voxel
from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import snapshot_orbit_cameras


class SparseVoxelOccupancyVisMixin:
    @torch.no_grad()
    def visualize_sample(self, x: Union[sp.SparseTensor, dict]):
        x = x if isinstance(x, sp.SparseTensor) else x['x']

        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4

        exts, ints = snapshot_orbit_cameras()

        images = []
        x = x.cuda()
        for i in range(x.shape[0]):
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / self.resolution,
                coords=x[i].coords[:, 1:].contiguous(),
                attrs=None,
                layout={'color': slice(0, 3)},
            )
            voxel_centers = (x[i].coords[:, 1:].float() + 0.5) / self.resolution - 0.5
            dist = torch.linalg.norm(voxel_centers, dim=-1)
            dist = (dist / dist.max().clamp_min(1e-6)).unsqueeze(-1)
            attr = torch.cat([
                dist,
                1.0 - dist,
                0.5 + 0.5 * torch.sin(dist * torch.pi),
            ], dim=-1).clamp(0, 1)
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(rep, ext, intr, colors_overwrite=attr)
                image[
                    :,
                    512 * (j // tile[1]):512 * (j // tile[1] + 1),
                    512 * (j % tile[1]):512 * (j % tile[1] + 1),
                ] = res['color']
            images.append(image)

        return {'occupancy': torch.stack(images)}


class SparseVoxelOccupancyDataset(SparseVoxelOccupancyVisMixin, StandardDatasetBase):
    """
    Sparse occupancy dataset backed by existing voxelized coordinates.

    Each active voxel receives one constant feature. This is intended for
    training a shape/subdivision VAE whose main target is the sparse structure,
    not surface attributes.
    """

    def __init__(
        self,
        roots,
        resolution: int = 256,
        max_active_voxels: int = 1000000,
        max_num_faces: int = None,
        min_aesthetic_score: float = 4.5,
        voxel_root_key: str = 'gaussian_distance_voxel',
        voxel_dirname: str = 'gaussian_distance_voxels',
        voxelized_flag_column: str = 'gaussian_distance_voxelized',
        num_voxels_column: str = 'num_gaussian_distance_voxels',
        feature_value: float = 1.0,
    ):
        self.resolution = resolution
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.min_aesthetic_score = min_aesthetic_score
        self.voxel_root_key = voxel_root_key
        self.voxel_dirname = voxel_dirname
        self.voxelized_flag_column = voxelized_flag_column
        self.num_voxels_column = num_voxels_column
        self.feature_value = feature_value
        self.value_range = (-1, 1)

        super().__init__(roots)
        self.loads = [self.metadata.loc[sha256, self.num_voxels_column] for _, sha256 in self.instances]

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
            f'  - Voxel dirname: {self.voxel_dirname}_{self.resolution}',
            f'  - Feature value: {self.feature_value}',
        ]
        return '\n'.join(lines)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[self.voxelized_flag_column] == True]
        stats[f'{self.voxelized_flag_column} == True'] = len(metadata)
        if self.min_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata[self.num_voxels_column] <= self.max_active_voxels]
        stats[f'Active voxels <= {self.max_active_voxels}'] = len(metadata)
        if self.max_num_faces is not None:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        return metadata, stats

    def read_occupancy_voxel(self, root, instance):
        coords, _ = o_voxel.io.read_vxz(os.path.join(root, f'{instance}.vxz'), num_threads=4)
        feats = torch.full((coords.shape[0], 1), self.feature_value, dtype=torch.float32)
        x = sp.SparseTensor(
            feats,
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        return {'x': x}

    def get_instance(self, root, instance):
        return self.read_occupancy_voxel(root[self.voxel_root_key], instance)

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], sp.SparseTensor):
                    pack[k] = sp.sparse_cat([b[k] for b in sub_batch], dim=0)
                else:
                    pack[k] = [b[k] for b in sub_batch]
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs
