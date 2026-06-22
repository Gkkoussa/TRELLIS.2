from typing import *

import os
import torch
import numpy as np
import pandas as pd

from .components import StandardDatasetBase
from .sparse_voxel_triangle_field import (
    SparseVoxelTriangleFieldVisMixin,
    find_triangle_field_path,
    load_triangle_field_npz,
)
from ..modules import sparse as sp
from ..utils.data_utils import load_balanced_group_indices


class TriangleFieldSuperResolutionDataset(SparseVoxelTriangleFieldVisMixin, StandardDatasetBase):
    """
    Paired low/high-resolution triangle-field dataset for decoder-constrained
    feature-space flow.

    Returns high-resolution x_0 and low-resolution d_tri/d_vert copied onto the
    high-resolution support by parent coordinate lookup.
    """

    def __init__(
        self,
        roots,
        low_resolution: int = 64,
        high_resolution: int = 128,
        max_active_voxels: int = 1000000,
        min_aesthetic_score: float = 4.5,
        max_num_faces: int = None,
        low_voxel_root_key: str = 'low_triangle_field_voxel',
        high_voxel_root_key: str = 'high_triangle_field_voxel',
        voxelized_flag_column: str = 'triangle_field_voxelized',
        num_voxels_column: str = 'num_triangle_field_voxels',
        distance_transform: str = 'minus_one_one',
        return_area_offsets: bool = False,
    ):
        if high_resolution % low_resolution != 0:
            raise ValueError(f'high_resolution must be divisible by low_resolution, got {high_resolution}/{low_resolution}')
        self.low_resolution = low_resolution
        self.high_resolution = high_resolution
        self.resolution = high_resolution
        self.max_active_voxels = max_active_voxels
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_faces = max_num_faces
        self.low_voxel_root_key = low_voxel_root_key
        self.high_voxel_root_key = high_voxel_root_key
        self.voxelized_flag_column = voxelized_flag_column
        self.num_voxels_column = num_voxels_column
        self.value_range = (0, 1)
        self.distance_transform = distance_transform
        self.return_area_offsets = return_area_offsets
        if self.distance_transform not in ('none', 'minus_one_one'):
            raise ValueError(f"distance_transform must be 'none' or 'minus_one_one', got {self.distance_transform}")

        super().__init__(roots)
        self._filter_paired_instances()
        self.loads = [
            self.metadata.loc[sha256, self.num_voxels_column]
            if self.num_voxels_column in self.metadata.columns else 1
            for _, sha256 in self.instances
        ]

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Low resolution: {self.low_resolution}',
            f'  - High resolution: {self.high_resolution}',
            f'  - Max active high-res voxels: {self.max_active_voxels}',
            f'  - Distance transform: {self.distance_transform}',
            f'  - Return area offsets: {self.return_area_offsets}',
        ]
        return '\n'.join(lines)

    def _filter_paired_instances(self) -> None:
        """
        StandardDatasetBase merges metadata across root entries, but this dataset
        needs an actual low/high pair for every instance.
        """
        if not isinstance(self.roots, dict):
            return

        valid_by_source = {}
        for key, root in self.roots.items():
            low_metadata = pd.read_csv(os.path.join(root[self.low_voxel_root_key], 'metadata.csv'))
            high_metadata = pd.read_csv(os.path.join(root[self.high_voxel_root_key], 'metadata.csv'))
            valid = set(low_metadata['sha256'].values).intersection(set(high_metadata['sha256'].values))
            valid_by_source[key] = valid
            if key in self._stats:
                self._stats[key]['Paired low/high metadata'] = len(valid)

        filtered = []
        for root, sha256 in self.instances:
            source_key = None
            for key, candidate in self.roots.items():
                if root is candidate:
                    source_key = key
                    break
            if source_key is None:
                filtered.append((root, sha256))
            elif sha256 in valid_by_source[source_key]:
                filtered.append((root, sha256))
        self.instances = filtered
        if len(self.metadata) > 0:
            valid_union = set().union(*valid_by_source.values()) if valid_by_source else set()
            self.metadata = self.metadata[self.metadata.index.isin(valid_union)]

    def filter_metadata(self, metadata):
        stats = {}
        if self.voxelized_flag_column in metadata.columns:
            metadata = metadata[metadata[self.voxelized_flag_column] == True]
            stats[f'{self.voxelized_flag_column} == True'] = len(metadata)
        if self.min_aesthetic_score is not None and 'aesthetic_score' in metadata.columns:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        if self.num_voxels_column in metadata.columns:
            metadata = metadata[metadata[self.num_voxels_column] > 0]
            stats['Active voxels > 0'] = len(metadata)
            metadata = metadata[metadata[self.num_voxels_column] <= self.max_active_voxels]
            stats[f'Active voxels <= {self.max_active_voxels}'] = len(metadata)
        if self.max_num_faces is not None and 'num_faces' in metadata.columns:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        return metadata, stats

    def _transform_target_channels(self, features: torch.Tensor) -> torch.Tensor:
        target = features[:, :2]
        if self.distance_transform == 'minus_one_one':
            target = target * 2.0 - 1.0
        return target.float()

    @staticmethod
    def _coord_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
        coords = coords.long()
        return coords[:, 0] * (resolution * resolution) + coords[:, 1] * resolution + coords[:, 2]

    def _read_target_features(
        self,
        root: str,
        instance: str,
        return_area_offsets: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = find_triangle_field_path(root, instance)
        with load_triangle_field_npz(path) as data:
            coords = torch.from_numpy(data['coords'].astype(np.int32, copy=False))
            features = torch.from_numpy(data['features'].astype(np.float32, copy=False))
        if features.ndim != 2 or features.shape[1] < 2:
            raise ValueError(f'{path} has invalid feature shape {tuple(features.shape)}')
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'{path} has invalid coords shape {tuple(coords.shape)}')
        if coords.shape[0] != features.shape[0]:
            raise ValueError(f'{path} coords/features length mismatch: {coords.shape[0]} vs {features.shape[0]}')
        area_offsets = None
        if return_area_offsets:
            if features.shape[1] < 11:
                raise ValueError(
                    f'{path} must contain offset_to_v0/v1/v2 channels for inverse-area weighting, '
                    f'got {features.shape[1]} channels'
                )
            area_offsets = features[:, 2:11].float()
        return coords, self._transform_target_channels(features), area_offsets

    def _low_to_high_support(
        self,
        low_coords: torch.Tensor,
        low_target: torch.Tensor,
        high_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        factor = self.high_resolution // self.low_resolution
        parent_coords = torch.div(high_coords, factor, rounding_mode='floor')
        if low_coords.numel() == 0:
            cond = torch.zeros((high_coords.shape[0], low_target.shape[1]), dtype=low_target.dtype)
            return cond, 1.0
        low_keys = self._coord_keys(low_coords, self.low_resolution)
        parent_keys = self._coord_keys(parent_coords, self.low_resolution)
        order = torch.argsort(low_keys)
        sorted_keys = low_keys[order]
        sorted_feats = low_target[order]
        idx = torch.searchsorted(sorted_keys, parent_keys)
        valid = (idx < sorted_keys.numel()) & (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == parent_keys)
        cond = torch.zeros((high_coords.shape[0], low_target.shape[1]), dtype=low_target.dtype)
        if valid.any():
            cond[valid] = sorted_feats[idx[valid]]
        missing_frac = 1.0 - valid.float().mean().item()
        return cond, missing_frac

    def get_instance(self, root, instance):
        low_coords, low_target, _ = self._read_target_features(root[self.low_voxel_root_key], instance)
        high_coords, high_target, area_offsets = self._read_target_features(
            root[self.high_voxel_root_key],
            instance,
            return_area_offsets=self.return_area_offsets,
        )
        cond, missing_frac = self._low_to_high_support(low_coords, low_target, high_coords)

        sparse_coords = torch.cat([torch.zeros_like(high_coords[:, 0:1]), high_coords], dim=-1).int()
        x_0 = sp.SparseTensor(high_target.float(), sparse_coords)
        cond = sp.SparseTensor(cond.float(), sparse_coords)
        pack = {
            'x_0': x_0,
            'cond': cond,
            'missing_low_parent_frac': torch.tensor(missing_frac, dtype=torch.float32),
        }
        if self.return_area_offsets:
            pack['area_offsets'] = sp.SparseTensor(area_offsets.float(), sparse_coords)
        return pack

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x_0'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            for k in sub_batch[0].keys():
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
