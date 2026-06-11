import os
import io
import csv
from typing import Union
import numpy as np
import pickle
import torch
from PIL import Image
import o_voxel
import utils3d
from .components import StandardDatasetBase
from ..modules import sparse as sp
from ..renderers import VoxelRenderer
from ..representations import Voxel
from ..representations.mesh import MeshWithPbrMaterial, TextureFilterMode, TextureWrapMode, AlphaMode, PbrMaterial, Texture

from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import snapshot_orbit_cameras


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def nearest_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    if is_power_of_two(n):
        return n
    lower = 2 ** (n.bit_length() - 1)
    upper = 2 ** n.bit_length()
    if n - lower < upper - n:
        return lower
    else:
        return upper
    

class SparseVoxelPbrVisMixin:
    @torch.no_grad()
    def visualize_sample(self, x: Union[sp.SparseTensor, dict]):
        x = x if isinstance(x, sp.SparseTensor) else x['x']
        
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        
        exts, ints = snapshot_orbit_cameras()

        images = {k: [] for k in self.layout}
        
        # Build each representation
        x = x.cuda()
        for i in range(x.shape[0]):
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1/self.resolution,
                coords=x[i].coords[:, 1:].contiguous(),
                attrs=None,
                layout={
                    'color': slice(0, 3),
                }
            )
            for k in self.layout:
                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    attr = x[i].feats[:, self.layout[k]].expand(-1, 3)
                    res = renderer.render(rep, ext, intr, colors_overwrite=attr)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images[k].append(image)
        
        for k in self.layout:
            images[k] = torch.stack(images[k])
            
        return images


class SparseVoxelShapeDataset(SparseVoxelPbrVisMixin, StandardDatasetBase):
    """
    Sparse voxel occupancy dataset.

    Reads existing sparse .vxz voxel files and returns only the active voxel
    coordinates with a single occupancy feature set to 1.
    """

    def __init__(
        self,
        roots,
        resolution: int = 1024,
        max_active_voxels: int = 1000000,
        max_num_faces: int = None,
        min_aesthetic_score: float = 5.0,
        voxel_root_key: str = 'shape_voxel',
        voxel_dirname: str = 'shape_voxels',
        voxelized_flag_column: str = 'shape_voxelized',
        num_voxels_column: str = 'num_shape_voxels',
        num_read_threads: int = 4,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.voxel_root_key = voxel_root_key
        self.voxel_dirname = voxel_dirname
        self.voxelized_flag_column = voxelized_flag_column
        self.num_voxels_column = num_voxels_column
        self.num_read_threads = num_read_threads
        self.value_range = (0, 1)
        self.layout = {
            'occupancy': slice(0, 1),
        }

        super().__init__(roots)

        self.loads = [self.metadata.loc[sha256, self.num_voxels_column] for _, sha256 in self.instances]

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
            f'  - Voxel dirname: {self.voxel_dirname}_{self.resolution}',
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

    def read_shape_voxel(self, root, instance):
        coords, _ = o_voxel.io.read_vxz(
            os.path.join(root[self.voxel_root_key], f'{instance}.vxz'),
            num_threads=self.num_read_threads,
        )
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        x = sp.SparseTensor(
            feats,
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        return {'x': x}

    def get_instance(self, root, instance):
        return self.read_shape_voxel(root, instance)

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


class DenseGaussianPatchDataset(StandardDatasetBase):
    """
    Dense 6-channel Gaussian-distance patch dataset.

    Reads existing sparse .vxz Gaussian-distance voxel files, samples a local
    dense patch, and returns it as x_0 for dense flow matching.
    """

    def __init__(
        self,
        roots,
        resolution: int = 256,
        patch_size: int = 64,
        max_active_voxels: int = None,
        max_num_faces: int = None,
        min_aesthetic_score: float = 5.0,
        attrs: list[str] = ['base_color', 'emissive'],
        voxel_root_key: str = 'gaussian_distance_voxel',
        voxelized_flag_column: str = 'gaussian_distance_voxelized',
        num_voxels_column: str = 'num_gaussian_distance_voxels',
        foreground_patch_prob: float = 0.8,
        background_value: float = -1.0,
        num_read_threads: int = 4,
        cond_as_token: bool = False,
        zero_cond: bool = False,
        return_origin: bool = False,
        metadata_filter_csv: str = None,
        metadata_filter_column: str = None,
    ):
        if patch_size > resolution:
            raise ValueError(f"patch_size ({patch_size}) must be <= resolution ({resolution})")
        if not 0.0 <= foreground_patch_prob <= 1.0:
            raise ValueError("foreground_patch_prob must be in [0, 1]")
        if max_active_voxels is not None and max_active_voxels > patch_size ** 3:
            raise ValueError(f"max_active_voxels ({max_active_voxels}) must be <= patch_size^3 ({patch_size ** 3})")

        self.resolution = resolution
        self.patch_size = patch_size
        self.min_aesthetic_score = min_aesthetic_score
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.voxel_root_key = voxel_root_key
        self.voxelized_flag_column = voxelized_flag_column
        self.num_voxels_column = num_voxels_column
        self.foreground_patch_prob = foreground_patch_prob
        self.background_value = background_value
        self.num_read_threads = num_read_threads
        self.cond_as_token = cond_as_token
        self.zero_cond = zero_cond
        self.return_origin = return_origin
        self.metadata_filter_csv = metadata_filter_csv
        self.metadata_filter_column = metadata_filter_column
        self._metadata_filter_sha256 = None
        self.value_range = (-1, 1)
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            if attr not in self.channels:
                raise ValueError(f"Unsupported voxel attribute: {attr}")
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]
        self.num_channels = start

        super().__init__(roots)

        self.loads = [self.metadata.loc[sha256, self.num_voxels_column] for _, sha256 in self.instances]

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
            f'  - Patch size: {self.patch_size}',
            f'  - Attributes: {list(self.layout.keys())}',
            f'  - Max patch active voxels: {self.max_active_voxels}',
            f'  - Foreground patch probability: {self.foreground_patch_prob}',
            f'  - Background value: {self.background_value}',
            f'  - Cond as token: {self.cond_as_token}',
            f'  - Zero cond: {self.zero_cond}',
            f'  - Return origin: {self.return_origin}',
            f'  - Metadata filter CSV: {self.metadata_filter_csv}',
            f'  - Metadata filter column: {self.metadata_filter_column}',
        ]
        return '\n'.join(lines)

    def _load_gaussian_metadata_filter(self):
        if self.metadata_filter_csv is None:
            return None
        if self._metadata_filter_sha256 is not None:
            return self._metadata_filter_sha256

        def truthy(value):
            return str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y'}

        allowed = set()
        with open(self.metadata_filter_csv, newline='') as f:
            reader = csv.DictReader(f)
            if 'sha256' not in reader.fieldnames:
                raise ValueError(f"Metadata filter CSV must contain sha256 column: {self.metadata_filter_csv}")
            if self.metadata_filter_column is not None and self.metadata_filter_column not in reader.fieldnames:
                raise ValueError(
                    f"Metadata filter CSV is missing requested column "
                    f"{self.metadata_filter_column}: {self.metadata_filter_csv}"
                )
            for row in reader:
                if self.metadata_filter_column is None or truthy(row[self.metadata_filter_column]):
                    allowed.add(str(row['sha256']))
        self._metadata_filter_sha256 = allowed
        return allowed

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[self.voxelized_flag_column] == True]
        stats[f'{self.voxelized_flag_column} == True'] = len(metadata)
        if self.min_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        if self.max_num_faces is not None:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        allowed_sha256 = self._load_gaussian_metadata_filter()
        if allowed_sha256 is not None:
            metadata = metadata[metadata.index.astype(str).isin(allowed_sha256)]
            stats[f'Metadata filter {self.metadata_filter_column or "sha256"}'] = len(metadata)
        return metadata, stats

    def _sample_patch_origin(self, coords: torch.Tensor) -> torch.Tensor:
        max_origin = self.resolution - self.patch_size
        if coords.numel() == 0 or np.random.rand() >= self.foreground_patch_prob:
            return torch.randint(0, max_origin + 1, (3,), dtype=torch.long)

        coord = coords[torch.randint(0, coords.shape[0], (1,)).item()].long()
        low = torch.clamp(coord - self.patch_size + 1, min=0, max=max_origin)
        high = torch.clamp(coord, min=0, max=max_origin)
        origin = torch.stack([
            torch.randint(low[i].item(), high[i].item() + 1, (1,), dtype=torch.long)[0]
            for i in range(3)
        ])
        return origin

    def _read_gaussian_voxel(self, root, instance):
        coords, attr = o_voxel.io.read_vxz(
            os.path.join(root[self.voxel_root_key], f'{instance}.vxz'),
            num_threads=self.num_read_threads,
        )
        feats = torch.concat([attr[k] for k in self.layout], dim=-1).float() / 255.0 * 2 - 1
        return coords.long(), feats

    def get_instance(self, root, instance):
        coords, feats = self._read_gaussian_voxel(root, instance)
        origin = self._sample_patch_origin(coords)
        patch_max = origin + self.patch_size
        mask = torch.all((coords >= origin) & (coords < patch_max), dim=1)

        patch = torch.full(
            (self.num_channels, self.patch_size, self.patch_size, self.patch_size),
            self.background_value,
            dtype=torch.float32,
        )
        if mask.any():
            selected = mask.nonzero(as_tuple=False).flatten()
            if self.max_active_voxels is not None and selected.numel() > self.max_active_voxels:
                selected = selected[torch.randperm(selected.numel())[:self.max_active_voxels]]
            local_coords = coords[selected] - origin
            patch[
                :,
                local_coords[:, 0],
                local_coords[:, 1],
                local_coords[:, 2],
            ] = feats[selected].t()

        denom = max(self.resolution - self.patch_size, 1)
        cond = torch.zeros(3, dtype=torch.float32) if self.zero_cond else origin.float() / denom
        if self.cond_as_token:
            cond = cond.unsqueeze(0)

        pack = {
            'x_0': patch,
            'cond': cond,
        }
        if self.return_origin:
            pack['patch_origin'] = origin.float()
        return pack

    @staticmethod
    def collate_fn(batch):
        return {
            key: torch.stack([b[key] for b in batch])
            for key in batch[0].keys()
        }

    @torch.no_grad()
    def visualize_sample(self, x: Union[torch.Tensor, dict]):
        x = x if isinstance(x, torch.Tensor) else x['x_0']
        x = x.detach().float().cpu()
        mid = x.shape[2] // 2

        def to_rgb(channels):
            image = x[:, channels, mid, :, :]
            if image.shape[1] == 1:
                image = image.expand(-1, 3, -1, -1)
            elif image.shape[1] > 3:
                image = image[:, :3]
            image = (image + 1) * 0.5
            return image.clamp(0, 1)

        images = {}
        if self.num_channels >= 3:
            images['edge'] = to_rgb(slice(0, 3))
        if self.num_channels >= 6:
            images['vertex'] = to_rgb(slice(3, 6))
        if not images:
            images['patch'] = to_rgb(slice(0, 1))
        return images


class SparseGaussianPatchDataset(DenseGaussianPatchDataset):
    """
    Sparse 6-channel Gaussian-distance patch dataset.

    Uses the active .vxz voxel coordinates inside a sampled local patch as the
    sparse support, matching the original sparse latent flow setup: coordinates
    are provided, and flow matching denoises only the features on those coords.
    """

    def __init__(
        self,
        *args,
        min_patch_active_voxels: int = 1,
        max_resample_attempts: int = 16,
        **kwargs,
    ):
        self.min_patch_active_voxels = min_patch_active_voxels
        self.max_resample_attempts = max_resample_attempts
        super().__init__(*args, **kwargs)

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Min patch active voxels: {self.min_patch_active_voxels}',
            f'  - Max resample attempts: {self.max_resample_attempts}',
        ]
        return '\n'.join(lines)

    def _select_patch(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        origin = None
        selected = torch.empty(0, dtype=torch.long)
        for _ in range(max(self.max_resample_attempts, 1)):
            origin = self._sample_patch_origin(coords)
            patch_max = origin + self.patch_size
            mask = torch.all((coords >= origin) & (coords < patch_max), dim=1)
            selected = mask.nonzero(as_tuple=False).flatten()
            if selected.numel() >= self.min_patch_active_voxels:
                break

        if selected.numel() < self.min_patch_active_voxels and coords.numel() != 0:
            old_foreground_patch_prob = self.foreground_patch_prob
            self.foreground_patch_prob = 1.0
            origin = self._sample_patch_origin(coords)
            self.foreground_patch_prob = old_foreground_patch_prob
            patch_max = origin + self.patch_size
            mask = torch.all((coords >= origin) & (coords < patch_max), dim=1)
            selected = mask.nonzero(as_tuple=False).flatten()

        if self.max_active_voxels is not None and selected.numel() > self.max_active_voxels:
            selected = selected[torch.randperm(selected.numel())[:self.max_active_voxels]]
        return origin, selected

    def get_instance(self, root, instance):
        coords, feats = self._read_gaussian_voxel(root, instance)
        origin, selected = self._select_patch(coords)
        if origin is None:
            origin = torch.zeros(3, dtype=torch.long)

        local_coords = coords[selected] - origin if selected.numel() != 0 else torch.empty(0, 3, dtype=torch.long)
        sparse_coords = torch.cat([
            torch.zeros(local_coords.shape[0], 1, dtype=torch.int32),
            local_coords.int(),
        ], dim=1)
        sparse_feats = feats[selected].float() if selected.numel() != 0 else torch.empty(0, self.num_channels, dtype=torch.float32)
        x_0 = sp.SparseTensor(sparse_feats, sparse_coords)

        denom = max(self.resolution - self.patch_size, 1)
        cond = torch.zeros(3, dtype=torch.float32) if self.zero_cond else origin.float() / denom
        if self.cond_as_token:
            cond = cond.unsqueeze(0)

        pack = {
            'x_0': x_0,
            'cond': cond,
        }
        if self.return_origin:
            pack['patch_origin'] = origin.float()
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
            for key in sub_batch[0].keys():
                if isinstance(sub_batch[0][key], torch.Tensor):
                    pack[key] = torch.stack([b[key] for b in sub_batch])
                elif isinstance(sub_batch[0][key], sp.SparseTensor):
                    pack[key] = sp.sparse_cat([b[key] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][key], list):
                    pack[key] = sum([b[key] for b in sub_batch], [])
                else:
                    pack[key] = [b[key] for b in sub_batch]
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs

    @torch.no_grad()
    def visualize_sample(self, x: Union[sp.SparseTensor, dict]):
        x = x if isinstance(x, sp.SparseTensor) else x['x_0']

        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch_i in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch_i),
                np.cos(yaw) * np.cos(pitch_i),
                np.sin(pitch_i),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = {k: [] for k in self.layout}
        x = x.cuda()
        for i in range(x.shape[0]):
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / self.patch_size,
                coords=x[i].coords[:, 1:].contiguous(),
                attrs=None,
                layout={
                    'color': slice(0, 3),
                }
            )
            for k in self.layout:
                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    attr = x[i].feats[:, self.layout[k]]
                    if attr.shape[1] == 1:
                        attr = attr.expand(-1, 3)
                    elif attr.shape[1] > 3:
                        attr = attr[:, :3]
                    res = renderer.render(rep, ext, intr, colors_overwrite=attr)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images[k].append(image)

        for k in self.layout:
            images[k] = torch.stack(images[k])

        return images


class SparseVoxelPbrDataset(SparseVoxelPbrVisMixin, StandardDatasetBase):
    """
    Sparse Voxel PBR dataset.
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(
        self,
        roots,
        resolution: int = 1024,
        max_active_voxels: int = 1000000,
        max_num_faces: int = None,
        min_aesthetic_score: float = 5.0,
        attrs: list[str] = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha'],
        with_mesh: bool = True,
        voxel_root_key: str = 'pbr_voxel',
        voxel_dirname: str = 'pbr_voxels',
        voxelized_flag_column: str = 'pbr_voxelized',
        num_voxels_column: str = 'num_pbr_voxels',
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.with_mesh = with_mesh
        self.voxel_root_key = voxel_root_key
        self.voxel_dirname = voxel_dirname
        self.voxelized_flag_column = voxelized_flag_column
        self.num_voxels_column = num_voxels_column
        self.value_range = (-1, 1)
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]

        super().__init__(roots)
        
        self.loads = [self.metadata.loc[sha256, self.num_voxels_column] for _, sha256 in self.instances]
        
    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
            f'  - Attributes: {list(self.layout.keys())}',
            f'  - Voxel dirname: {self.voxel_dirname}_{self.resolution}',
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

    @staticmethod
    def _texture_from_dump(pack) -> Texture:
        png_bytes = pack['image']
        image = Image.open(io.BytesIO(png_bytes))
        if image.width != image.height or not is_power_of_two(image.width):
            size = nearest_power_of_two(max(image.width, image.height))
            image = image.resize((size, size), Image.LANCZOS)
        texture = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).reshape(image.height, image.width, -1)
        filter_mode = {
            'Linear': TextureFilterMode.LINEAR,
            'Closest': TextureFilterMode.CLOSEST,
            'Cubic': TextureFilterMode.LINEAR,
            'Smart': TextureFilterMode.LINEAR,
        }[pack['interpolation']]
        wrap_mode = {
            'REPEAT': TextureWrapMode.REPEAT,
            'EXTEND': TextureWrapMode.CLAMP_TO_EDGE,
            'CLIP': TextureWrapMode.CLAMP_TO_EDGE,
            'MIRROR': TextureWrapMode.MIRRORED_REPEAT,
        }[pack['extension']]
        return Texture(texture, filter_mode=filter_mode, wrap_mode=wrap_mode)

    def read_mesh_with_texture(self, root, instance):
        with open(os.path.join(root, f'{instance}.pickle'), 'rb') as f:
            dump = pickle.load(f)
            
        # Fix dump alpha map
        for mat in dump['materials']:
            if mat['alphaTexture'] is not None and mat['alphaMode'] == 'OPAQUE':
                mat['alphaMode'] = 'BLEND'

        # process material
        materials = []
        for mat in dump['materials']:
            materials.append(PbrMaterial(
                base_color_texture=self._texture_from_dump(mat['baseColorTexture']) if mat['baseColorTexture'] is not None else None,
                base_color_factor=mat['baseColorFactor'],
                metallic_texture=self._texture_from_dump(mat['metallicTexture']) if mat['metallicTexture'] is not None else None,
                metallic_factor=mat['metallicFactor'],
                roughness_texture=self._texture_from_dump(mat['roughnessTexture']) if mat['roughnessTexture'] is not None else None,
                roughness_factor=mat['roughnessFactor'],
                alpha_texture=self._texture_from_dump(mat['alphaTexture']) if mat['alphaTexture'] is not None else None,
                alpha_factor=mat['alphaFactor'],
                alpha_mode={
                    'OPAQUE': AlphaMode.OPAQUE,
                    'MASK': AlphaMode.MASK,
                    'BLEND': AlphaMode.BLEND,
                }[mat['alphaMode']],
                alpha_cutoff=mat['alphaCutoff'],
            ))
        materials.append(PbrMaterial(
            base_color_factor=[0.8, 0.8, 0.8],
            alpha_factor=1.0,
            metallic_factor=0.0,
            roughness_factor=0.5,
            alpha_mode=AlphaMode.OPAQUE,
            alpha_cutoff=0.5,
        ))  # append default material

        # process mesh
        start = 0
        vertices = []
        faces = []
        material_ids = []
        uv_coords = []
        for obj in dump['objects']:
            if obj['vertices'].size == 0 or obj['faces'].size == 0:
                continue
            vertices.append(obj['vertices'])
            faces.append(obj['faces'] + start)
            obj['mat_ids'][obj['mat_ids'] == -1] = len(materials) - 1
            material_ids.append(obj['mat_ids'])
            uv_coords.append(obj['uvs'] if obj['uvs'] is not None else np.zeros((obj['faces'].shape[0], 3, 2), dtype=np.float32))
            start += len(obj['vertices'])
        
        vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
        faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
        material_ids = torch.from_numpy(np.concatenate(material_ids, axis=0)).long()
        uv_coords = torch.from_numpy(np.concatenate(uv_coords, axis=0)).float()
        
        # Normalize vertices
        vertices_min = vertices.min(dim=0)[0]
        vertices_max = vertices.max(dim=0)[0]
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
        
        return {'mesh': [MeshWithPbrMaterial(
            vertices=vertices,
            faces=faces,
            material_ids=material_ids,
            uv_coords=uv_coords,
            materials=materials,
        )]}

    def read_pbr_voxel(self, root, instance):
        coords, attr = o_voxel.io.read_vxz(os.path.join(root, f'{instance}.vxz'), num_threads=4)
        feats = torch.concat([attr[k] for k in self.layout], dim=-1) / 255.0 * 2 - 1
        x = sp.SparseTensor(
            feats.float(),
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        return {'x': x}
    
    def get_instance(self, root, instance):
        if self.with_mesh:
            mesh = self.read_mesh_with_texture(root['pbr_dump'], instance)
            pbr_voxel = self.read_pbr_voxel(root[self.voxel_root_key], instance)
            return {**mesh, **pbr_voxel}
        else:
            return self.read_pbr_voxel(root[self.voxel_root_key], instance)
    
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
