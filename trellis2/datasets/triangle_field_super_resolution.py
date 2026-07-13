from typing import *

import os
import torch
import numpy as np
import pandas as pd

from .components import StandardDatasetBase
from .sparse_voxel_triangle_field import (
    EXTENDED_INPUT_LAYOUT,
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
        conditioning_augmentation: dict = None,
        density_conditioning: bool = False,
        density_voxel_root_key: str = 'density_triangle_field_voxel',
        density_channel: str = 'density_field',
        instances_path: str = None,
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
        self.conditioning_augmentation = conditioning_augmentation
        self.density_conditioning = bool(density_conditioning)
        self.density_voxel_root_key = density_voxel_root_key
        self.density_channel = density_channel
        self.instances_path = instances_path
        if self.density_channel not in EXTENDED_INPUT_LAYOUT:
            raise ValueError(
                f'density_channel must be one of {sorted(EXTENDED_INPUT_LAYOUT.keys())}, '
                f'got {self.density_channel}'
            )
        if self.distance_transform not in ('none', 'minus_one_one'):
            raise ValueError(f"distance_transform must be 'none' or 'minus_one_one', got {self.distance_transform}")
        if self.conditioning_augmentation is not None:
            self._validate_conditioning_augmentation(self.conditioning_augmentation)

        super().__init__(roots)
        self._filter_paired_instances()
        self._filter_instances_path()
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
            f'  - Conditioning augmentation: {self.conditioning_augmentation}',
            f'  - Density conditioning: {self.density_conditioning}',
            f'  - Density voxel root key: {self.density_voxel_root_key}',
            f'  - Density channel: {self.density_channel}',
        ]
        return '\n'.join(lines)

    def _filter_instances_path(self) -> None:
        if self.instances_path is None:
            return
        if not os.path.exists(self.instances_path):
            raise FileNotFoundError(f'instances_path not found: {self.instances_path}')
        with open(self.instances_path, 'r') as f:
            keep = {line.strip() for line in f if line.strip()}
        self.instances = [(root, sha256) for root, sha256 in self.instances if sha256 in keep]
        if len(self.metadata) > 0:
            self.metadata = self.metadata[self.metadata.index.isin(keep)]
        for stats in self._stats.values():
            stats[f'Restricted to {os.path.basename(os.path.dirname(self.instances_path))}/{os.path.basename(self.instances_path)}'] = len(self.instances)

    @staticmethod
    def _validate_conditioning_augmentation(cfg: dict) -> None:
        if cfg.get('type') != 'sparse_blur_noise':
            raise ValueError(
                f"Unsupported conditioning_augmentation type {cfg.get('type')}. "
                "Expected 'sparse_blur_noise'."
            )
        blur_sigma = float(cfg.get('blur_sigma', 1.0))
        noise_level = float(cfg.get('noise_level', 0.0))
        apply_prob = float(cfg.get('apply_prob', 1.0))
        if blur_sigma <= 0:
            raise ValueError(f'conditioning_augmentation.blur_sigma must be positive, got {blur_sigma}')
        if not (0.0 <= noise_level <= 1.0):
            raise ValueError(f'conditioning_augmentation.noise_level must be in [0, 1], got {noise_level}')
        if not (0.0 <= apply_prob <= 1.0):
            raise ValueError(f'conditioning_augmentation.apply_prob must be in [0, 1], got {apply_prob}')

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
            if self.density_conditioning:
                if self.density_voxel_root_key not in root:
                    raise KeyError(
                        f'Density conditioning requires root key {self.density_voxel_root_key}. '
                        f'Available keys: {sorted(root.keys())}'
                    )
                density_metadata = pd.read_csv(os.path.join(root[self.density_voxel_root_key], 'metadata.csv'))
                valid = valid.intersection(set(density_metadata['sha256'].values))
            valid_by_source[key] = valid
            if key in self._stats:
                stat_name = 'Paired low/high/density metadata' if self.density_conditioning else 'Paired low/high metadata'
                self._stats[key][stat_name] = len(valid)

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

    def _read_coords(self, root: str, instance: str) -> torch.Tensor:
        path = find_triangle_field_path(root, instance)
        with load_triangle_field_npz(path) as data:
            coords = torch.from_numpy(data['coords'].astype(np.int32, copy=False))
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'{path} has invalid coords shape {tuple(coords.shape)}')
        return coords

    def _features_to_support(
        self,
        source_coords: torch.Tensor,
        source_feats: torch.Tensor,
        target_coords: torch.Tensor,
        resolution: int,
    ) -> Tuple[torch.Tensor, float]:
        if source_coords.numel() == 0:
            feats = torch.zeros(
                (target_coords.shape[0], source_feats.shape[1]),
                dtype=source_feats.dtype,
            )
            return feats, 1.0
        if source_coords.shape == target_coords.shape and torch.equal(source_coords, target_coords):
            return source_feats, 0.0

        source_keys = self._coord_keys(source_coords, resolution)
        target_keys = self._coord_keys(target_coords, resolution)
        order = torch.argsort(source_keys)
        sorted_keys = source_keys[order]
        sorted_feats = source_feats[order]
        idx = torch.searchsorted(sorted_keys, target_keys)
        valid = (
            (idx < sorted_keys.numel()) &
            (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == target_keys)
        )
        feats = torch.zeros(
            (target_coords.shape[0], source_feats.shape[1]),
            dtype=source_feats.dtype,
        )
        if valid.any():
            feats[valid] = sorted_feats[idx[valid]]
        missing_frac = 1.0 - valid.float().mean().item()
        return feats, missing_frac

    def _read_density_conditioning(
        self,
        root: str,
        instance: str,
        high_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        path = find_triangle_field_path(root, instance)
        with load_triangle_field_npz(path) as data:
            coords = torch.from_numpy(data['coords'].astype(np.int32, copy=False))
            features = torch.from_numpy(data['features'].astype(np.float32, copy=False))
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'{path} has invalid coords shape {tuple(coords.shape)}')
        if features.ndim != 2:
            raise ValueError(f'{path} has invalid feature shape {tuple(features.shape)}')
        if coords.shape[0] != features.shape[0]:
            raise ValueError(f'{path} coords/features length mismatch: {coords.shape[0]} vs {features.shape[0]}')

        slc = EXTENDED_INPUT_LAYOUT[self.density_channel]
        if features.shape[1] < slc.stop:
            raise ValueError(
                f'{path} has {features.shape[1]} feature channels, but density channel '
                f'{self.density_channel} requires at least {slc.stop}'
            )
        return self._features_to_support(
            coords,
            features[:, slc].float(),
            high_coords,
            self.high_resolution,
        )

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

    def _sparse_blur_features(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        keys = self._coord_keys(coords, self.high_resolution)
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        sorted_feats = feats[order]

        accum = torch.zeros_like(feats)
        denom = torch.zeros((feats.shape[0], 1), dtype=feats.dtype, device=feats.device)
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
        for dx, dy, dz in offsets:
            offset = torch.tensor([dx, dy, dz], dtype=coords.dtype, device=coords.device)
            neighbor = coords + offset
            valid = (
                (neighbor[:, 0] >= 0) & (neighbor[:, 0] < self.high_resolution) &
                (neighbor[:, 1] >= 0) & (neighbor[:, 1] < self.high_resolution) &
                (neighbor[:, 2] >= 0) & (neighbor[:, 2] < self.high_resolution)
            )
            if not valid.any():
                continue
            query_keys = self._coord_keys(neighbor[valid], self.high_resolution)
            idx = torch.searchsorted(sorted_keys, query_keys)
            hit = (
                (idx < sorted_keys.numel()) &
                (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == query_keys)
            )
            if not hit.any():
                continue
            rows = valid.nonzero(as_tuple=False).reshape(-1)[hit].to(device=feats.device)
            weight = float(np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)))
            accum[rows] += sorted_feats[idx[hit]] * weight
            denom[rows] += weight
        return accum / denom.clamp_min(1e-6)

    def _augment_conditioning(
        self,
        high_coords: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.conditioning_augmentation
        if cfg is None:
            return cond
        apply_prob = float(cfg.get('apply_prob', 1.0))
        if apply_prob < 1.0 and torch.rand(()) >= apply_prob:
            return cond

        blur_sigma = float(cfg.get('blur_sigma', 1.0))
        noise_level = float(cfg.get('noise_level', 0.0))
        cond = self._sparse_blur_features(high_coords, cond, blur_sigma).clamp(-1.0, 1.0)
        if noise_level > 0:
            cond = ((1.0 - noise_level) * cond + noise_level * torch.randn_like(cond)).clamp(-1.0, 1.0)
        return cond

    def get_instance(self, root, instance):
        low_coords, low_target, _ = self._read_target_features(root[self.low_voxel_root_key], instance)
        high_coords, high_target, area_offsets = self._read_target_features(
            root[self.high_voxel_root_key],
            instance,
            return_area_offsets=self.return_area_offsets,
        )
        cond, missing_frac = self._low_to_high_support(low_coords, low_target, high_coords)
        cond = self._augment_conditioning(high_coords, cond)
        density_cond = None
        density_missing_frac = None
        if self.density_conditioning:
            density_cond, density_missing_frac = self._read_density_conditioning(
                root[self.density_voxel_root_key],
                instance,
                high_coords,
            )

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
        if self.density_conditioning:
            pack['density_cond'] = sp.SparseTensor(density_cond.float(), sparse_coords)
            pack['density_missing_parent_frac'] = torch.tensor(density_missing_frac, dtype=torch.float32)
        return pack

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            load_key = 'x_0' if 'x_0' in batch[0] else 'cond'
            group_idx = load_balanced_group_indices([b[load_key].feats.shape[0] for b in batch], split_size)
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


class TriangleFieldLatentSuperResolutionDataset(TriangleFieldSuperResolutionDataset):
    """
    Paired low/high-resolution triangle-field dataset with precomputed high-res
    VAE latents. The feature conditioning path matches
    TriangleFieldSuperResolutionDataset, while `z_0` is loaded from an offline
    full-feature triangle-field encoder pass.
    """

    def __init__(
        self,
        *args,
        latent_root_key: str = 'triangle_field_latent',
        latent_encoded_flag_column: str = 'triangle_field_latent_encoded',
        latent_tokens_column: str = 'triangle_field_latent_tokens',
        max_latent_tokens: int = 32768,
        **kwargs,
    ):
        self.latent_root_key = latent_root_key
        self.latent_encoded_flag_column = latent_encoded_flag_column
        self.latent_tokens_column = latent_tokens_column
        self.max_latent_tokens = max_latent_tokens
        super().__init__(*args, **kwargs)

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Latent root key: {self.latent_root_key}',
            f'  - Max latent tokens: {self.max_latent_tokens}',
        ]
        return '\n'.join(lines)

    def _filter_paired_instances(self) -> None:
        if not isinstance(self.roots, dict):
            return

        valid_by_source = {}
        for key, root in self.roots.items():
            low_metadata = pd.read_csv(os.path.join(root[self.low_voxel_root_key], 'metadata.csv'))
            high_metadata = pd.read_csv(os.path.join(root[self.high_voxel_root_key], 'metadata.csv'))
            latent_metadata = pd.read_csv(os.path.join(root[self.latent_root_key], 'metadata.csv'))
            valid = (
                set(low_metadata['sha256'].values)
                .intersection(set(high_metadata['sha256'].values))
                .intersection(set(latent_metadata['sha256'].values))
            )
            if self.density_conditioning:
                if self.density_voxel_root_key not in root:
                    raise KeyError(
                        f'Density conditioning requires root key {self.density_voxel_root_key}. '
                        f'Available keys: {sorted(root.keys())}'
                    )
                density_metadata = pd.read_csv(os.path.join(root[self.density_voxel_root_key], 'metadata.csv'))
                valid = valid.intersection(set(density_metadata['sha256'].values))
            valid_by_source[key] = valid
            if key in self._stats:
                stat_name = (
                    'Paired low/high/latent/density metadata'
                    if self.density_conditioning else
                    'Paired low/high/latent metadata'
                )
                self._stats[key][stat_name] = len(valid)

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
        metadata, stats = super().filter_metadata(metadata)
        if self.latent_encoded_flag_column in metadata.columns:
            metadata = metadata[metadata[self.latent_encoded_flag_column] == True]
            stats[f'{self.latent_encoded_flag_column} == True'] = len(metadata)
        if self.latent_tokens_column in metadata.columns:
            metadata = metadata[metadata[self.latent_tokens_column] > 0]
            stats['Latent tokens > 0'] = len(metadata)
            metadata = metadata[metadata[self.latent_tokens_column] <= self.max_latent_tokens]
            stats[f'Latent tokens <= {self.max_latent_tokens}'] = len(metadata)
        return metadata, stats

    def _read_latent(self, root: str, instance: str) -> Tuple[sp.SparseTensor, str]:
        latent_path = os.path.join(root, f'{instance}.npz')
        cache_path = os.path.join(root, f'{instance}.cache.pt')
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f'Triangle-field latent file not found: {latent_path}')
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f'Triangle-field latent cache not found: {cache_path}')

        with np.load(latent_path, allow_pickle=False) as data:
            coords = torch.from_numpy(data['coords'].astype(np.int32, copy=False))
            feats = torch.from_numpy(data['feats'].astype(np.float32, copy=False))
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'{latent_path} has invalid coords shape {tuple(coords.shape)}')
        if feats.ndim != 2:
            raise ValueError(f'{latent_path} has invalid feats shape {tuple(feats.shape)}')
        if coords.shape[0] != feats.shape[0]:
            raise ValueError(f'{latent_path} coords/feats length mismatch: {coords.shape[0]} vs {feats.shape[0]}')

        sparse_coords = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1).int()
        return sp.SparseTensor(feats.float(), sparse_coords), cache_path

    @staticmethod
    def _read_latent_cache(cache_path: str) -> Dict[str, Any]:
        try:
            return torch.load(cache_path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(cache_path, map_location='cpu')

    def get_instance(self, root, instance):
        low_coords, low_target, _ = self._read_target_features(root[self.low_voxel_root_key], instance)
        high_coords = self._read_coords(root[self.high_voxel_root_key], instance)
        cond, missing_frac = self._low_to_high_support(low_coords, low_target, high_coords)
        cond = self._augment_conditioning(high_coords, cond)
        sparse_coords = torch.cat([torch.zeros_like(high_coords[:, 0:1]), high_coords], dim=-1).int()
        z_0, cache_path = self._read_latent(root[self.latent_root_key], instance)
        density_cond = None
        density_missing_frac = None
        if self.density_conditioning:
            density_cond, density_missing_frac = self._read_density_conditioning(
                root[self.density_voxel_root_key],
                instance,
                high_coords,
            )
        pack = {
            'cond': sp.SparseTensor(cond.float(), sparse_coords),
            'missing_low_parent_frac': torch.tensor(missing_frac, dtype=torch.float32),
            'z_0': z_0,
        }
        if self.density_conditioning:
            pack['density_cond'] = sp.SparseTensor(density_cond.float(), sparse_coords)
            pack['density_missing_parent_frac'] = torch.tensor(density_missing_frac, dtype=torch.float32)
        pack['triangle_field_slat_cache_path'] = cache_path
        pack['triangle_field_slat_cache'] = self._read_latent_cache(cache_path)
        return pack
