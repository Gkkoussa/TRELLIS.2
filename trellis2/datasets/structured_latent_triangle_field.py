import os
import json
from typing import *
import numpy as np
import torch
import utils3d

from .. import models
from .components import StandardDatasetBase
from ..modules.sparse import SparseTensor, sparse_cat
from ..renderers import VoxelRenderer
from ..representations import Voxel
from ..utils.render_utils import snapshot_orbit_cameras
from ..utils.data_utils import load_balanced_group_indices


class TriangleFieldSLatVisMixin:
    def __init__(
        self,
        *args,
        triangle_field_slat_dec_path: Optional[str] = None,
        triangle_field_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: Optional[str] = None,
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.triangle_field_slat_dec = None
        self.triangle_field_slat_dec_path = triangle_field_slat_dec_path
        self.triangle_field_slat_dec_ckpt = triangle_field_slat_dec_ckpt
        self.triangle_field_distance_transform = 'none'
        self.shape_slat_dec = None
        self.pretrained_shape_slat_dec = pretrained_shape_slat_dec
        self.shape_slat_dec_path = shape_slat_dec_path
        self.shape_slat_dec_ckpt = shape_slat_dec_ckpt

    def _loading_triangle_field_slat_dec(self):
        if self.triangle_field_slat_dec is not None:
            if self.shape_slat_dec_path is None and self.pretrained_shape_slat_dec is None:
                return
            if self.shape_slat_dec is not None:
                return
        if self.triangle_field_slat_dec_path is None:
            raise ValueError("A triangle-field decoder path is required for snapshot visualization.")

        cfg = json.load(open(os.path.join(self.triangle_field_slat_dec_path, 'config.json'), 'r'))
        decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
        ckpt_path = os.path.join(self.triangle_field_slat_dec_path, 'ckpts', f'decoder_{self.triangle_field_slat_dec_ckpt}.pt')
        decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        self.triangle_field_slat_dec = decoder.cuda().eval()
        self.triangle_field_distance_transform = (
            cfg.get('dataset', {})
            .get('args', {})
            .get('distance_transform', 'none')
        )

        if self.shape_slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.shape_slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.shape_slat_dec_path, 'ckpts', f'decoder_{self.shape_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
            self.shape_slat_dec = decoder.cuda().eval()
        elif self.pretrained_shape_slat_dec is not None:
            self.shape_slat_dec = models.from_pretrained(self.pretrained_shape_slat_dec).cuda().eval()

    def _delete_triangle_field_slat_dec(self):
        del self.triangle_field_slat_dec
        self.triangle_field_slat_dec = None
        if self.shape_slat_dec is not None:
            del self.shape_slat_dec
            self.shape_slat_dec = None

    def _load_triangle_field_slat_cache(self, cache_path: str, device: torch.device):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Triangle-field latent cache not found: {cache_path}")
        try:
            cache = torch.load(cache_path, map_location=device, weights_only=False)
        except TypeError:
            cache = torch.load(cache_path, map_location=device)
        return cache

    @torch.no_grad()
    def decode_latent(self, z, cache_paths: Optional[list[str]] = None, shape_z: Optional[SparseTensor] = None, batch_size=4):
        self._loading_triangle_field_slat_dec()
        if self.triangle_field_slat_normalization is not None:
            z = z.replace(z.feats * self.triangle_field_slat_std.to(z.device) + self.triangle_field_slat_mean.to(z.device))
        if shape_z is not None and getattr(self, 'shape_slat_normalization', None) is not None:
            shape_z = shape_z.replace(shape_z.feats * self.shape_slat_std.to(shape_z.device) + self.shape_slat_mean.to(shape_z.device))
        if getattr(self.triangle_field_slat_dec, 'pred_subdiv', False):
            voxels = []
            for i in range(0, z.shape[0], batch_size):
                decoded = self.triangle_field_slat_dec(z[i:i + batch_size])
                for j in range(decoded.shape[0]):
                    voxels.append(decoded[j])
            self._delete_triangle_field_slat_dec()
            return voxels
        if cache_paths is None and shape_z is None:
            raise ValueError("Triangle-field latent cache paths are required for decoding pred_subdiv=False latents.")

        voxels = []
        if shape_z is not None:
            if self.shape_slat_dec is None:
                raise ValueError("A shape latent decoder is required to decode with guide_subs.")
            for i in range(0, z.shape[0], batch_size):
                zi = z[i:i + batch_size]
                shape_zi = shape_z[i:i + batch_size]
                _, subs = self.shape_slat_dec(shape_zi, return_subs=True)
                decoded = self.triangle_field_slat_dec(zi, guide_subs=subs)
                for j in range(decoded.shape[0]):
                    voxels.append(decoded[j])
        else:
            for i in range(z.shape[0]):
                zi = z[i]
                cache = self._load_triangle_field_slat_cache(cache_paths[i], zi.device)
                zi._scale = cache['scale']
                zi._spatial_cache = cache['spatial_cache']
                decoded = self.triangle_field_slat_dec(zi)
                for j in range(decoded.shape[0]):
                    voxels.append(decoded[j])
        self._delete_triangle_field_slat_dec()
        return voxels

    @staticmethod
    def _scalar_to_color(values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1, 1).clamp(0, 1)
        return values.expand(-1, 3)

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        z = sample if isinstance(sample, SparseTensor) else sample['x_0']
        if isinstance(sample, SparseTensor):
            cache_paths = None
            shape_z = None
        else:
            cache_paths = sample.get('triangle_field_slat_cache_path', None)
            shape_z = sample.get('concat_cond', None)
        voxels = self.decode_latent(
            z.cuda(),
            cache_paths=cache_paths,
            shape_z=shape_z.cuda() if shape_z is not None else None,
        )

        render_resolution = int(self.snapshot_render_resolution)
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = render_resolution
        renderer.rendering_options.ssaa = 4

        exts, ints = snapshot_orbit_cameras()

        images = {k: [] for k in self.layout}

        for voxel in voxels:
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / self.resolution,
                coords=voxel.coords[:, 1:].contiguous(),
                attrs=None,
                layout={'color': slice(0, 3)},
            )
            feats = voxel.feats
            if self.triangle_field_distance_transform == 'minus_one_one':
                feats = feats * 0.5 + 0.5
            feats = feats.clamp(0, 1)
            for k in self.layout:
                image = torch.zeros(3, render_resolution * 2, render_resolution * 2, dtype=torch.float32).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    attr = self._scalar_to_color(feats[:, self.layout[k]]).float()
                    with torch.autocast(device_type='cuda', enabled=False):
                        res = renderer.render(rep, ext.float(), intr.float(), colors_overwrite=attr)
                    image[
                        :,
                        render_resolution * (j // tile[1]):render_resolution * (j // tile[1] + 1),
                        render_resolution * (j % tile[1]):render_resolution * (j % tile[1] + 1),
                    ] = res['color'].float()
                images[k].append(image)

        for k in self.layout:
            images[k] = torch.stack(images[k])

        return images


class MichelangeloConditionedTriangleFieldSLat(TriangleFieldSLatVisMixin, StandardDatasetBase):
    """
    Triangle-field structured latent dataset conditioned on Michelangelo latents.
    """
    def __init__(
        self,
        roots: str,
        *,
        resolution: int = 256,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        triangle_field_slat_normalization: Optional[dict] = None,
        triangle_field_slat_normalization_path: Optional[str] = None,
        triangle_field_slat_dec_path: Optional[str] = None,
        triangle_field_slat_dec_ckpt: Optional[str] = None,
        snapshot_render_resolution: int = 512,
    ):
        if triangle_field_slat_normalization is not None and triangle_field_slat_normalization_path is not None:
            raise ValueError("Provide either triangle_field_slat_normalization or triangle_field_slat_normalization_path, not both.")
        self.resolution = resolution
        self.triangle_field_slat_normalization = triangle_field_slat_normalization
        self.triangle_field_slat_normalization_path = triangle_field_slat_normalization_path
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.value_range = (0, 1)
        self.snapshot_render_resolution = snapshot_render_resolution
        self.layout = {
            'd_tri': slice(0, 1),
            'd_vert': slice(1, 2),
        }

        super().__init__(
            roots,
            triangle_field_slat_dec_path=triangle_field_slat_dec_path,
            triangle_field_slat_dec_ckpt=triangle_field_slat_dec_ckpt,
        )

        self.loads = [self.metadata.loc[sha256, 'triangle_field_latent_tokens'] for _, sha256 in self.instances]

        if self.triangle_field_slat_normalization is None:
            if self.triangle_field_slat_normalization_path is not None:
                with open(self.triangle_field_slat_normalization_path, 'r') as fp:
                    self.triangle_field_slat_normalization = json.load(fp)
            else:
                self.triangle_field_slat_normalization = self._load_triangle_field_slat_normalization()

        if self.triangle_field_slat_normalization is not None:
            self.triangle_field_slat_mean = torch.tensor(self.triangle_field_slat_normalization['mean']).reshape(1, -1)
            self.triangle_field_slat_std = torch.tensor(self.triangle_field_slat_normalization['std']).reshape(1, -1)

    def _load_triangle_field_slat_normalization(self):
        if not isinstance(self.roots, dict):
            raise ValueError("MichelangeloConditionedTriangleFieldSLat requires JSON roots with a triangle_field_latent entry.")

        paths = []
        for _, root in self.roots.items():
            if 'triangle_field_latent' not in root:
                raise ValueError("Dataset root is missing required key: triangle_field_latent")
            paths.append(os.path.join(root['triangle_field_latent'], 'normalization.json'))

        missing = [path for path in paths if not os.path.exists(path)]
        if len(missing) != 0:
            raise FileNotFoundError(
                "Triangle-field latent normalization stats are required but were not found. "
                f"Missing: {missing}"
            )

        normalizations = []
        for path in paths:
            with open(path, 'r') as fp:
                normalizations.append(json.load(fp))

        reference = normalizations[0]
        for path, normalization in zip(paths[1:], normalizations[1:]):
            if normalization['mean'] != reference['mean'] or normalization['std'] != reference['std']:
                raise ValueError(
                    "All triangle_field_latent roots must use the same normalization stats. "
                    f"Mismatch found at {path}."
                )
        return reference

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['triangle_field_latent_encoded'] == True]
        stats['With triangle-field latent'] = len(metadata)
        metadata = metadata[metadata['michelangelo_latent_encoded'] == True]
        stats['With Michelangelo latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['triangle_field_latent_tokens'] <= self.max_tokens]
        stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        data = np.load(os.path.join(root['triangle_field_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.triangle_field_slat_normalization is not None:
            feats = (feats - self.triangle_field_slat_mean) / self.triangle_field_slat_std
        triangle_field_z = SparseTensor(feats, coords)

        data = np.load(os.path.join(root['michelangelo_latent'], f'{instance}.npz'))
        cond = torch.tensor(data['feats']).float()
        neg_cond = torch.zeros_like(cond)

        return {
            'x_0': triangle_field_z,
            'cond': cond,
            'neg_cond': neg_cond,
            'triangle_field_slat_cache_path': os.path.join(root['triangle_field_latent'], f'{instance}.cache.pt'),
        }

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

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    pack[k] = sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


class MichelangeloShapeConditionedTriangleFieldSLat(TriangleFieldSLatVisMixin, StandardDatasetBase):
    """
    Triangle-field structured latent dataset conditioned on both:

    - dense Michelangelo latents through cross-attention (`cond`)
    - sparse occupancy shape latents through token-wise concatenation (`concat_cond`)
    """
    def __init__(
        self,
        roots: str,
        *,
        resolution: int = 256,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        triangle_field_slat_normalization: Optional[dict] = None,
        triangle_field_slat_normalization_path: Optional[str] = None,
        shape_slat_normalization: Optional[dict] = None,
        shape_slat_normalization_path: Optional[str] = None,
        triangle_field_slat_dec_path: Optional[str] = None,
        triangle_field_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: Optional[str] = None,
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        snapshot_render_resolution: int = 512,
    ):
        if triangle_field_slat_normalization is not None and triangle_field_slat_normalization_path is not None:
            raise ValueError("Provide either triangle_field_slat_normalization or triangle_field_slat_normalization_path, not both.")
        if shape_slat_normalization is not None and shape_slat_normalization_path is not None:
            raise ValueError("Provide either shape_slat_normalization or shape_slat_normalization_path, not both.")
        self.resolution = resolution
        self.triangle_field_slat_normalization = triangle_field_slat_normalization
        self.triangle_field_slat_normalization_path = triangle_field_slat_normalization_path
        self.shape_slat_normalization = shape_slat_normalization
        self.shape_slat_normalization_path = shape_slat_normalization_path
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.value_range = (0, 1)
        self.snapshot_render_resolution = snapshot_render_resolution
        self.layout = {
            'd_tri': slice(0, 1),
            'd_vert': slice(1, 2),
        }

        super().__init__(
            roots,
            triangle_field_slat_dec_path=triangle_field_slat_dec_path,
            triangle_field_slat_dec_ckpt=triangle_field_slat_dec_ckpt,
            pretrained_shape_slat_dec=pretrained_shape_slat_dec,
            shape_slat_dec_path=shape_slat_dec_path,
            shape_slat_dec_ckpt=shape_slat_dec_ckpt,
        )

        self.loads = [self.metadata.loc[sha256, 'triangle_field_latent_tokens'] for _, sha256 in self.instances]

        if self.triangle_field_slat_normalization is None:
            if self.triangle_field_slat_normalization_path is not None:
                with open(self.triangle_field_slat_normalization_path, 'r') as fp:
                    self.triangle_field_slat_normalization = json.load(fp)
            else:
                self.triangle_field_slat_normalization = self._load_latent_normalization('triangle_field_latent')

        if self.shape_slat_normalization is None:
            if self.shape_slat_normalization_path is not None:
                with open(self.shape_slat_normalization_path, 'r') as fp:
                    self.shape_slat_normalization = json.load(fp)
            else:
                self.shape_slat_normalization = self._load_latent_normalization('shape_latent')

        if self.triangle_field_slat_normalization is not None:
            self.triangle_field_slat_mean = torch.tensor(self.triangle_field_slat_normalization['mean']).reshape(1, -1)
            self.triangle_field_slat_std = torch.tensor(self.triangle_field_slat_normalization['std']).reshape(1, -1)
        if self.shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(self.shape_slat_normalization['mean']).reshape(1, -1)
            self.shape_slat_std = torch.tensor(self.shape_slat_normalization['std']).reshape(1, -1)

    def _load_latent_normalization(self, key):
        if not isinstance(self.roots, dict):
            raise ValueError("MichelangeloShapeConditionedTriangleFieldSLat requires JSON roots.")

        paths = []
        for _, root in self.roots.items():
            if key not in root:
                raise ValueError(f"Dataset root is missing required key: {key}")
            paths.append(os.path.join(root[key], 'normalization.json'))

        missing = [path for path in paths if not os.path.exists(path)]
        if len(missing) != 0:
            raise FileNotFoundError(
                f"{key} normalization stats are required but were not found. "
                f"Missing: {missing}"
            )

        normalizations = []
        for path in paths:
            with open(path, 'r') as fp:
                normalizations.append(json.load(fp))

        reference = normalizations[0]
        for path, normalization in zip(paths[1:], normalizations[1:]):
            if normalization['mean'] != reference['mean'] or normalization['std'] != reference['std']:
                raise ValueError(
                    f"All {key} roots must use the same normalization stats. "
                    f"Mismatch found at {path}."
                )
        return reference

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['triangle_field_latent_encoded'] == True]
        stats['With triangle-field latent'] = len(metadata)
        metadata = metadata[metadata['michelangelo_latent_encoded'] == True]
        stats['With Michelangelo latent'] = len(metadata)
        metadata = metadata[metadata['shape_latent_encoded'] == True]
        stats['With shape latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['triangle_field_latent_tokens'] <= self.max_tokens]
        stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        data = np.load(os.path.join(root['triangle_field_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.triangle_field_slat_normalization is not None:
            feats = (feats - self.triangle_field_slat_mean) / self.triangle_field_slat_std
        triangle_field_z = SparseTensor(feats, coords)

        data = np.load(os.path.join(root['shape_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.shape_slat_normalization is not None:
            feats = (feats - self.shape_slat_mean) / self.shape_slat_std
        shape_z = SparseTensor(feats, coords)

        if not torch.equal(shape_z.coords, triangle_field_z.coords):
            raise ValueError(
                f"Shape latent and triangle-field latent have different coordinates for {instance}: "
                f"{shape_z.coords.shape} vs {triangle_field_z.coords.shape}"
            )

        data = np.load(os.path.join(root['michelangelo_latent'], f'{instance}.npz'))
        cond = torch.tensor(data['feats']).float()
        neg_cond = torch.zeros_like(cond)

        return {
            'x_0': triangle_field_z,
            'concat_cond': shape_z,
            'cond': cond,
            'neg_cond': neg_cond,
            'triangle_field_slat_cache_path': os.path.join(root['triangle_field_latent'], f'{instance}.cache.pt'),
        }

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

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    pack[k] = sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs
