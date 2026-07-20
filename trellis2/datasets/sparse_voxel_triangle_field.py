import bisect
import io
import json
import os
from typing import Union

import numpy as np
import torch
import utils3d

from .components import StandardDatasetBase
from ..modules import sparse as sp
from ..renderers import VoxelRenderer
from ..representations import Voxel
from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import snapshot_orbit_cameras


INPUT_LAYOUT = {
    'd_tri': slice(0, 1),
    'd_vert': slice(1, 2),
    'offset_to_v0': slice(2, 5),
    'offset_to_v1': slice(5, 8),
    'offset_to_v2': slice(8, 11),
    'offset_to_centroid': slice(11, 14),
    'face_normal': slice(14, 17),
    'offset_to_projection': slice(17, 20),
}

EXTENDED_INPUT_LAYOUT = {
    **INPUT_LAYOUT,
    'density_field': slice(20, 21),
}

TARGET_LAYOUT = {
    'd_tri': slice(0, 1),
    'd_vert': slice(1, 2),
}


def load_triangle_field_npz(path: str):
    if path.endswith('.zst'):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError('Reading .npz.zst triangle fields requires the zstandard package') from exc
        with open(path, 'rb') as f:
            payload = zstd.ZstdDecompressor().decompress(f.read())
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def find_triangle_field_path(root: str, instance: str) -> str:
    for suffix in ('.npz.zst', '.npz'):
        path = os.path.join(root, f'{instance}{suffix}')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No triangle-field voxel file found for {instance} in {root}')


class SparseVoxelTriangleFieldVisMixin:
    @staticmethod
    def _scalar_to_color(values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1, 1).clamp(0, 1)
        return values.expand(-1, 3)

    @staticmethod
    def _signed_vector_to_color(values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] >= 3:
            return (values[:, :3] * 0.5 + 0.5).clamp(0, 1)
        return SparseVoxelTriangleFieldVisMixin._scalar_to_color(values.norm(dim=1))

    @staticmethod
    def _magnitude_to_color(values: torch.Tensor) -> torch.Tensor:
        mag = torch.linalg.norm(values, dim=1, keepdim=True)
        denom = torch.quantile(mag.detach().float(), 0.99).clamp_min(1e-6)
        mag = (mag / denom).clamp(0, 1)
        return torch.cat([mag, 1.0 - mag, 0.25 * torch.ones_like(mag)], dim=1)

    @torch.no_grad()
    def visualize_sample(self, sample: Union[sp.SparseTensor, dict]):
        if isinstance(sample, sp.SparseTensor):
            x = sample
        elif 'target' in sample:
            x = sample['target']
        elif 'x_0' in sample:
            x = sample['x_0']
        elif 'cond' in sample:
            x = sample['cond']
        else:
            x = sample['x']

        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4

        exts, ints = snapshot_orbit_cameras()

        x = x.cuda()
        images = {}
        layout = TARGET_LAYOUT if x.feats.shape[1] == 2 else INPUT_LAYOUT
        for key, slc in layout.items():
            if slc.stop > x.feats.shape[1]:
                continue
            rendered = []
            for i in range(x.shape[0]):
                rep = Voxel(
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / self.resolution,
                    coords=x[i].coords[:, 1:].contiguous(),
                    attrs=None,
                    layout={'color': slice(0, 3)},
                )
                values = x[i].feats[:, slc]
                if key in ('d_tri', 'd_vert'):
                    if getattr(self, 'distance_transform', 'none') == 'minus_one_one':
                        values = values * 0.5 + 0.5
                    attr = self._scalar_to_color(values)
                elif key == 'face_normal':
                    attr = self._signed_vector_to_color(values)
                else:
                    attr = self._magnitude_to_color(values)
                attr = attr.float()

                image = torch.zeros(3, 1024, 1024, dtype=torch.float32).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    with torch.autocast(device_type='cuda', enabled=False):
                        res = renderer.render(rep, ext.float(), intr.float(), colors_overwrite=attr)
                    image[
                        :,
                        512 * (j // tile[1]):512 * (j // tile[1] + 1),
                        512 * (j % tile[1]):512 * (j % tile[1] + 1),
                    ] = res['color'].float()
                rendered.append(image)
            images[key] = torch.stack(rendered)

        return images


class SparseVoxelTriangleFieldDataset(SparseVoxelTriangleFieldVisMixin, StandardDatasetBase):
    """
    Sparse triangle-field voxel dataset.

    The encoder input `x` contains all stored triangle-field features. The
    reconstruction target contains only d_tri and d_vert.
    """

    def __init__(
        self,
        roots,
        resolution: int = 256,
        max_active_voxels: int = 1000000,
        max_num_faces: int = None,
        min_aesthetic_score: float = 4.5,
        voxel_root_key: str = 'triangle_field_voxel',
        voxel_dirname: str = 'triangle_field_voxels',
        voxelized_flag_column: str = 'triangle_field_voxelized',
        num_voxels_column: str = 'num_triangle_field_voxels',
        input_feature_scale: Union[float, list[float], None] = None,
        distance_transform: str = 'none',
        include_density_field: bool = False,
    ):
        self.resolution = resolution
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.min_aesthetic_score = min_aesthetic_score
        self.voxel_root_key = voxel_root_key
        self.voxel_dirname = voxel_dirname
        self.voxelized_flag_column = voxelized_flag_column
        self.num_voxels_column = num_voxels_column
        self.include_density_field = include_density_field
        self.input_layout = EXTENDED_INPUT_LAYOUT if include_density_field else INPUT_LAYOUT
        self.target_layout = TARGET_LAYOUT
        self.value_range = (0, 1)
        self.distance_transform = distance_transform
        if self.distance_transform not in ('none', 'minus_one_one'):
            raise ValueError(
                f"distance_transform must be 'none' or 'minus_one_one', got {self.distance_transform}"
            )

        if input_feature_scale is None:
            self.input_feature_scale = None
        else:
            self.input_feature_scale = torch.tensor(input_feature_scale, dtype=torch.float32)

        super().__init__(roots)
        self.loads = [self.metadata.loc[sha256, self.num_voxels_column] for _, sha256 in self.instances]

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
            f'  - Voxel dirname: {self.voxel_dirname}_{self.resolution}',
            f'  - Input channels: {self.num_input_channels}',
            f'  - Target channels: {self.num_target_channels}',
            f'  - Input feature scale: {None if self.input_feature_scale is None else "explicit"}',
            f'  - Distance transform: {self.distance_transform}',
            f'  - Include density field: {self.include_density_field}',
        ]
        return '\n'.join(lines)

    @property
    def num_input_channels(self) -> int:
        return max(slc.stop for slc in self.input_layout.values())

    @property
    def num_target_channels(self) -> int:
        return max(slc.stop for slc in self.target_layout.values())

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[self.voxelized_flag_column] == True]
        stats[f'{self.voxelized_flag_column} == True'] = len(metadata)
        if self.min_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata[self.num_voxels_column] > 0]
        stats['Active voxels > 0'] = len(metadata)
        metadata = metadata[metadata[self.num_voxels_column] <= self.max_active_voxels]
        stats[f'Active voxels <= {self.max_active_voxels}'] = len(metadata)
        if self.max_num_faces is not None:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        return metadata, stats

    def _scale_input_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.input_feature_scale is None:
            return features
        scale = self.input_feature_scale.to(features.device)
        if scale.numel() == 1:
            return features / scale.clamp_min(1e-12)
        if scale.numel() != features.shape[1]:
            raise ValueError(
                f'input_feature_scale must be scalar or length {features.shape[1]}, got {scale.numel()}'
            )
        return features / scale.reshape(1, -1).clamp_min(1e-12)

    def _transform_distance_channels(self, features: torch.Tensor) -> torch.Tensor:
        if self.distance_transform == 'none':
            return features
        features = features.clone()
        features[:, :self.num_target_channels] = features[:, :self.num_target_channels] * 2.0 - 1.0
        return features

    def read_triangle_field_voxel(self, root, instance):
        path = find_triangle_field_path(root, instance)
        with load_triangle_field_npz(path) as data:
            coords = torch.from_numpy(data['coords'].astype(np.int32, copy=False))
            features = torch.from_numpy(data['features'].astype(np.float32, copy=False))

        if features.ndim != 2 or features.shape[1] < self.num_target_channels:
            raise ValueError(f'{path} has invalid feature shape {tuple(features.shape)}')
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'{path} has invalid coords shape {tuple(coords.shape)}')
        if coords.shape[0] != features.shape[0]:
            raise ValueError(f'{path} coords/features length mismatch: {coords.shape[0]} vs {features.shape[0]}')
        if features.shape[1] < self.num_input_channels:
            raise ValueError(
                f'{path} has {features.shape[1]} feature channels, but dataset requires '
                f'{self.num_input_channels}. include_density_field={self.include_density_field}'
            )

        sparse_coords = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
        input_features = self._transform_distance_channels(features[:, :self.num_input_channels])
        target_features = input_features[:, :self.num_target_channels]
        x = sp.SparseTensor(
            self._scale_input_features(input_features).float(),
            sparse_coords,
        )
        target = sp.SparseTensor(
            target_features.float(),
            sparse_coords,
        )
        return {'x': x, 'target': target}

    def get_instance(self, root, instance):
        return self.read_triangle_field_voxel(root[self.voxel_root_key], instance)

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
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


class MultiResolutionSparseVoxelTriangleFieldDataset(SparseVoxelTriangleFieldVisMixin):
    """
    Flatten several SparseVoxelTriangleFieldDataset instances into one dataset.

    Each item is still loaded through the normal single-resolution dataset path,
    so feature transforms, filtering, and sparse collation stay shared.
    """

    def __init__(
        self,
        roots,
        resolutions=(32, 64, 128, 256, 512),
        instances_path: str = None,
        **kwargs,
    ):
        self.resolutions = [int(r) for r in resolutions]
        if len(self.resolutions) == 0:
            raise ValueError('resolutions must be non-empty.')
        self.resolution = max(self.resolutions)
        self.datasets = []
        self._cumulative_sizes = []
        self._stats = {}

        parsed_roots = json.loads(roots)
        keep = None
        if instances_path is not None:
            with open(instances_path, 'r') as f:
                keep = {line.strip() for line in f if line.strip()}

        total = 0
        for resolution in self.resolutions:
            per_resolution_roots = {}
            resolution_key = f'triangle_field_voxel_{resolution}'
            for source_name, source_roots in parsed_roots.items():
                if resolution_key in source_roots:
                    voxel_root = source_roots[resolution_key]
                elif 'triangle_field_voxel' in source_roots:
                    voxel_root = str(source_roots['triangle_field_voxel']).format(resolution=resolution)
                else:
                    raise KeyError(
                        f"Source {source_name} must define '{resolution_key}' or 'triangle_field_voxel'."
                    )
                per_resolution_roots[f'{source_name}_r{resolution}'] = {
                    'triangle_field_voxel': voxel_root,
                }

            dataset = SparseVoxelTriangleFieldDataset(
                json.dumps(per_resolution_roots),
                resolution=resolution,
                **kwargs,
            )
            if keep is not None:
                before = len(dataset.instances)
                dataset.instances = [
                    (root, sha256)
                    for root, sha256 in dataset.instances
                    if sha256 in keep
                ]
                if len(dataset.metadata) > 0:
                    dataset.metadata = dataset.metadata[dataset.metadata.index.astype(str).isin(keep)]
                dataset.loads = [
                    dataset.metadata.loc[sha256, dataset.num_voxels_column]
                    if dataset.num_voxels_column in dataset.metadata.columns else 1
                    for _, sha256 in dataset.instances
                ]
                for stats in dataset._stats.values():
                    stats['Restricted to instances'] = len(dataset.instances)
                    stats['Instances removed'] = before - len(dataset.instances)

            self.datasets.append(dataset)
            total += len(dataset)
            self._cumulative_sizes.append(total)
            self._stats[f'resolution_{resolution}'] = {
                'Total instances': len(dataset),
            }

        self.loads = []
        for dataset in self.datasets:
            self.loads.extend(dataset.loads)
        self.input_layout = self.datasets[0].input_layout
        self.target_layout = self.datasets[0].target_layout
        self.value_range = self.datasets[0].value_range
        self.distance_transform = self.datasets[0].distance_transform

    def __len__(self):
        return self._cumulative_sizes[-1]

    def __getitem__(self, index):
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, index)
        prev_size = 0 if dataset_idx == 0 else self._cumulative_sizes[dataset_idx - 1]
        pack = self.datasets[dataset_idx][index - prev_size]
        pack['resolution'] = torch.tensor(self.datasets[dataset_idx].resolution, dtype=torch.int32)
        return pack

    def __str__(self):
        lines = [
            self.__class__.__name__,
            f'  - Total instances: {len(self)}',
            f'  - Resolutions: {self.resolutions}',
            f'  - Snapshot/render resolution: {self.resolution}',
            f'  - Input channels: {self.datasets[0].num_input_channels}',
            f'  - Target channels: {self.datasets[0].num_target_channels}',
            f'  - Distance transform: {self.distance_transform}',
            '  - Per-resolution datasets:',
        ]
        for resolution, dataset in zip(self.resolutions, self.datasets):
            lines.append(f'    - {resolution}: {len(dataset)}')
        return '\n'.join(lines)

    @staticmethod
    def collate_fn(batch, split_size=None):
        return SparseVoxelTriangleFieldDataset.collate_fn(batch, split_size=split_size)
