from typing import *

import copy
import functools
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from .flow_matching import FlowMatchingTrainer
from ..vae.triangle_field_vae import TriangleFieldVaeTrainer
from ... import models
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler


class TriangleFieldSuperResolutionFlowTrainer(FlowMatchingTrainer):
    """
    Decoder-constrained feature-space flow for triangle-field super-resolution.

    The trainable model is a time-FiLM VAE encoder. It consumes noisy high-res
    d_tri/d_vert plus low-res d_tri/d_vert copied to high-res support, and the
    frozen VAE decoder maps its latent output back to clean high-res d_tri/d_vert.
    """

    def __init__(
        self,
        *args,
        decoder_model: dict,
        decoder_ckpt: str,
        num_workers: int = None,
        cond_drop_prob: float = 0.1,
        loss_type: str = 'l1',
        sample_posterior: bool = False,
        snapshot_t: float = 0.5,
        voxel_loss_weight: dict = None,
        conditioning_augmentation: dict = None,
        **kwargs,
    ):
        self.decoder_model_config = decoder_model
        self.decoder_ckpt = decoder_ckpt
        self.num_workers = num_workers
        self.cond_drop_prob = float(cond_drop_prob)
        self.loss_type = loss_type
        self.sample_posterior = sample_posterior
        self.snapshot_t = float(snapshot_t)
        self.voxel_loss_weight = voxel_loss_weight
        self.conditioning_augmentation = conditioning_augmentation
        self._cond_blur_cache = {}
        if not (0.0 <= self.cond_drop_prob <= 1.0):
            raise ValueError(f'cond_drop_prob must be in [0, 1], got {self.cond_drop_prob}')
        if self.loss_type not in ('l1', 'l2'):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {self.loss_type}")
        if self.conditioning_augmentation is not None:
            self._validate_conditioning_augmentation(self.conditioning_augmentation)
        if self.voxel_loss_weight is not None:
            weight_type = self.voxel_loss_weight.get('type', None)
            if weight_type != 'inverse_triangle_area':
                raise ValueError(
                    f"Unsupported voxel_loss_weight type {weight_type}. "
                    "Expected 'inverse_triangle_area'."
                )
            normalize = self.voxel_loss_weight.get('normalize', 'mean')
            if normalize not in ('mean', 'none'):
                raise ValueError("voxel_loss_weight.normalize must be 'mean' or 'none'.")
        super().__init__(*args, **kwargs)
        self._build_frozen_decoder()

    def _build_frozen_decoder(self):
        cfg = self.decoder_model_config
        self.decoder = getattr(models, cfg['name'])(**cfg['args']).to(self.device)
        ckpt = torch.load(self.decoder_ckpt, map_location=self.device, weights_only=True)
        self.decoder.load_state_dict(ckpt)
        self.decoder.eval()
        # Keep decoder params requiring grad so flex_gemm can backprop through
        # sparse convs into z. The decoder is still frozen because it is not in
        # self.models/master_params and its grads are cleared before each loss.
        for param in self.decoder.parameters():
            param.requires_grad_(True)

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Frozen decoder: {getattr(self, "decoder", None).__class__.__name__ if hasattr(self, "decoder") else "pending init"}',
            f'  - Decoder checkpoint: {self.decoder_ckpt}',
            f'  - Cond drop prob: {self.cond_drop_prob}',
            f'  - Loss type: {self.loss_type}',
            f'  - Sample posterior: {self.sample_posterior}',
            f'  - Snapshot t: {self.snapshot_t}',
            f'  - Voxel loss weight: {self.voxel_loss_weight}',
            f'  - Conditioning augmentation: {self.conditioning_augmentation}',
        ]
        return '\n'.join(lines)

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

    def _get_cond_blur_convs(
        self,
        channels: int,
        sigma: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        key = (channels, float(sigma), dtype, device.type, device.index)
        if key in self._cond_blur_cache:
            return self._cond_blur_cache[key]

        value_conv = sp.SparseConv3d(channels, channels, 3, bias=False).to(device=device, dtype=dtype)
        norm_conv = sp.SparseConv3d(1, 1, 3, bias=False).to(device=device, dtype=dtype)
        for conv in (value_conv, norm_conv):
            conv.eval()
            for param in conv.parameters():
                param.requires_grad_(False)

        kernel = torch.empty((3, 3, 3), device=device, dtype=torch.float32)
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    dx, dy, dz = x - 1, y - 1, z - 1
                    kernel[x, y, z] = float(np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)))
        kernel = kernel.to(dtype=dtype)

        value_weight = torch.zeros_like(value_conv.weight.data)
        for channel in range(channels):
            value_weight[channel, :, :, :, channel] = kernel
        value_conv.weight.data.copy_(value_weight)
        norm_conv.weight.data.copy_(kernel.reshape(1, 3, 3, 3, 1))

        self._cond_blur_cache[key] = (value_conv, norm_conv)
        return value_conv, norm_conv

    @torch.no_grad()
    def _augment_conditioning(self, cond: sp.SparseTensor) -> sp.SparseTensor:
        cfg = self.conditioning_augmentation
        if cfg is None:
            return cond

        apply_prob = float(cfg.get('apply_prob', 1.0))
        if apply_prob <= 0.0:
            return cond
        if apply_prob < 1.0:
            apply = torch.rand(cond.shape[0], device=cond.feats.device) < apply_prob
            if not apply.any():
                return cond
        else:
            apply = torch.ones(cond.shape[0], device=cond.feats.device, dtype=torch.bool)

        blur_sigma = float(cfg.get('blur_sigma', 1.0))
        noise_level = float(cfg.get('noise_level', 0.0))
        value_conv, norm_conv = self._get_cond_blur_convs(
            cond.feats.shape[1],
            blur_sigma,
            cond.feats.dtype,
            cond.feats.device,
        )

        blurred = value_conv(cond)
        ones = cond.replace(torch.ones((cond.feats.shape[0], 1), device=cond.feats.device, dtype=cond.feats.dtype))
        denom = norm_conv(ones).feats.clamp_min(1e-6)
        aug_feats = (blurred.feats / denom).clamp(-1.0, 1.0)
        if noise_level > 0:
            aug_feats = ((1.0 - noise_level) * aug_feats + noise_level * torch.randn_like(aug_feats)).clamp(-1.0, 1.0)

        feats = cond.feats.clone()
        feats[apply[cond.coords[:, 0].long()]] = aug_feats[apply[cond.coords[:, 0].long()]]
        return cond.replace(feats)

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers if self.num_workers is not None else int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def finetune_from(self, finetune_ckpt):
        """
        Partially load pretrained VAE encoder weights. New FiLM parameters and
        mismatched input-layer tensors are left initialized by the new model.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')

        model_ckpts = {}
        for name, model in self.models.items():
            model_state = model.state_dict()
            if name not in finetune_ckpt:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state
                continue
            raw = torch.load(finetune_ckpt[name], map_location=self.device, weights_only=True)
            loadable = {}
            skipped = []
            for k, v in raw.items():
                if k not in model_state:
                    skipped.append((k, 'missing_in_model'))
                elif v.shape != model_state[k].shape:
                    skipped.append((k, f'shape {tuple(v.shape)} != {tuple(model_state[k].shape)}'))
                else:
                    loadable[k] = v
            missing, unexpected = model.load_state_dict(loadable, strict=False)
            if self.is_master:
                for k, reason in skipped:
                    print(f'Warning: skipped {name}.{k}: {reason}')
                for k in missing:
                    print(f'Warning: left initialized {name}.{k}')
                for k in unexpected:
                    print(f'Warning: unexpected {name}.{k}')
            merged = model.state_dict()
            model_ckpts[name] = merged
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, _ in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')
        if self.world_size > 1:
            self.check_ddp()

    def _diffuse_sparse(
        self,
        x_0: sp.SparseTensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[sp.SparseTensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0.feats)
        batch_t = t[x_0.coords[:, 0].long()].reshape(-1, 1)
        x_t = (1.0 - batch_t) * x_0.feats + (self.sigma_min + (1.0 - self.sigma_min) * batch_t) * noise
        return x_0.replace(x_t), noise

    def _drop_cond(self, cond: sp.SparseTensor) -> Tuple[sp.SparseTensor, torch.Tensor]:
        batch_size = cond.shape[0]
        drop = torch.rand(batch_size, device=cond.feats.device) < self.cond_drop_prob
        if drop.any():
            feats = cond.feats.clone()
            feats[drop[cond.coords[:, 0].long()]] = 0
            cond = cond.replace(feats)
        return cond, drop

    @staticmethod
    def _make_encoder_input(x_t: sp.SparseTensor, cond: sp.SparseTensor) -> sp.SparseTensor:
        if not torch.equal(x_t.coords, cond.coords):
            raise ValueError(f'x_t and cond coords must match, got {x_t.coords.shape} vs {cond.coords.shape}')
        return x_t.replace(torch.cat([x_t.feats, cond.feats], dim=-1))

    def _predict_x0(self, x_t: sp.SparseTensor, cond: sp.SparseTensor, t: torch.Tensor) -> sp.SparseTensor:
        enc_in = self._make_encoder_input(x_t, cond)
        z = self.training_models['encoder'](
            enc_in,
            t * 1000.0,
            sample_posterior=self.sample_posterior,
        )
        return self.decoder(z)

    def _inverse_triangle_area_weight(self, area_offsets: sp.SparseTensor) -> torch.Tensor:
        if area_offsets.feats.shape[1] != 9:
            raise ValueError(
                f'inverse_triangle_area weighting expects 9 offset channels, got {area_offsets.feats.shape[1]}'
            )

        cfg = self.voxel_loss_weight or {}
        eps = float(cfg.get('eps', 1e-8))
        clamp_max = cfg.get('clamp_max', None)
        normalize = cfg.get('normalize', 'mean')

        with torch.autocast(device_type='cuda', enabled=False):
            offset0 = area_offsets.feats[:, 0:3].float()
            offset1 = area_offsets.feats[:, 3:6].float()
            offset2 = area_offsets.feats[:, 6:9].float()
            area = 0.5 * torch.linalg.norm(
                torch.cross(offset1 - offset0, offset2 - offset0, dim=-1),
                dim=-1,
            )
            weight = area.clamp_min(eps).reciprocal()
            if normalize == 'mean':
                weight = weight / weight.mean().clamp_min(eps)
            if clamp_max is not None:
                weight = weight.clamp_max(float(clamp_max))
        return weight

    def _reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.loss_type == 'l1':
            err = torch.abs(pred - target)
        elif self.loss_type == 'l2':
            err = (pred - target) ** 2
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        if weight is not None:
            err = err * weight.reshape(-1, 1).to(device=err.device, dtype=err.dtype)
        return err.mean()

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond: sp.SparseTensor,
        missing_low_parent_frac: torch.Tensor = None,
        area_offsets: sp.SparseTensor = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        self.decoder.zero_grad(set_to_none=True)
        t = self.sample_t(x_0.shape[0]).to(x_0.feats.device).float()
        x_t, _ = self._diffuse_sparse(x_0, t)
        cond = self._augment_conditioning(cond)
        cond, cond_drop = self._drop_cond(cond)
        pred_x0 = self._predict_x0(x_t, cond, t)
        if pred_x0.feats.shape != x_0.feats.shape:
            raise ValueError(f'Prediction shape must match x_0, got {pred_x0.feats.shape} vs {x_0.feats.shape}')

        loss_weight = None
        if self.voxel_loss_weight is not None:
            if area_offsets is None:
                raise ValueError(
                    'voxel_loss_weight requires area_offsets from the dataset. '
                    'Set dataset.args.return_area_offsets=true.'
                )
            if not torch.equal(area_offsets.coords, x_0.coords):
                raise ValueError('area_offsets coords must match x_0 coords')
            loss_weight = self._inverse_triangle_area_weight(area_offsets)

        terms = edict()
        loss_name = self.loss_type
        terms[loss_name] = self._reconstruction_loss(pred_x0.feats, x_0.feats, loss_weight)
        terms['loss'] = terms[loss_name]

        status = {
            'cond/drop_frac': cond_drop.float().mean(),
        }
        if missing_low_parent_frac is not None:
            missing_low_parent_frac = missing_low_parent_frac.float()
            status['cond/missing_low_parent_frac'] = missing_low_parent_frac.mean()
            status['cond/missing_low_parent_frac_max'] = missing_low_parent_frac.max()

        with torch.no_grad():
            l1_per_channel = (pred_x0.feats - x_0.feats).abs().mean(dim=0)
            status['recon/d_tri_l1'] = l1_per_channel[0]
            status['recon/d_vert_l1'] = l1_per_channel[1]
            if loss_weight is not None:
                status['loss_weight/mean'] = loss_weight.float().mean()
                status['loss_weight/max'] = loss_weight.float().max()
            time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
            err = torch.segment_reduce(
                (pred_x0.feats - x_0.feats).pow(2).mean(dim=1),
                reduce='mean',
                lengths=x_0.seqlen,
            ).detach().cpu().numpy()
            for i in range(10):
                if (time_bin == i).sum() != 0:
                    terms[f'bin_{i}'] = {'mse': err[time_bin == i].mean()}

        return terms, status

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        iterator = iter(dataloader)

        gt_images = {}
        cond_images = {}
        noisy_images = {}
        pred_images = {}
        self.models['encoder'].eval()
        self.decoder.eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iterator)
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            t = torch.full((args['x_0'].shape[0],), self.snapshot_t, device=self.device)
            x_t, _ = self._diffuse_sparse(args['x_0'], t)
            enc_in = self._make_encoder_input(x_t, args['cond'])
            z = self.models['encoder'](enc_in, t * 1000.0, sample_posterior=False)
            y = self.decoder(z)

            gt_vis = self.dataset.visualize_sample({'target': args['x_0']})
            cond_vis = self.dataset.visualize_sample({'target': args['cond']})
            noisy_vis = self.dataset.visualize_sample({'target': x_t})
            pred_vis = self.dataset.visualize_sample({'target': y})
            for k, v in gt_vis.items():
                gt_images.setdefault(k, []).append(v[:batch])
            for k, v in cond_vis.items():
                cond_images.setdefault(k, []).append(v[:batch])
            for k, v in noisy_vis.items():
                noisy_images.setdefault(k, []).append(v[:batch])
            for k, v in pred_vis.items():
                pred_images.setdefault(k, []).append(v[:batch])

        self.models['encoder'].train()

        sample_dict = {}
        for k in gt_images:
            sample_dict[f'gt_{k}'] = {'value': torch.cat(gt_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'cond_{k}'] = {'value': torch.cat(cond_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'noisy_{k}'] = {'value': torch.cat(noisy_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'pred_{k}'] = {'value': torch.cat(pred_images[k], dim=0)[:num_samples], 'type': 'image'}
        return sample_dict


class TriangleFieldLatentSuperResolutionFlowTrainer(TriangleFieldSuperResolutionFlowTrainer):
    """
    Latent-space super-resolution flow with feature-space model inputs.

    Clean high-resolution latents are loaded from an offline full-feature VAE
    encoding pass. Training samples latent z_t on the encoded support, decodes
    z_t without gradients, concatenates decoded features with the existing
    low-resolution conditioning, and trains the time-FiLM encoder to predict z_0.
    """

    def __init__(
        self,
        *args,
        latent_loss_parameterization: str = 'velocity',
        latent_loss_t_min: float = 0.05,
        batched_cache_decode: bool = False,
        latent_self_conditioning: dict = None,
        **kwargs,
    ):
        self.latent_loss_parameterization = latent_loss_parameterization
        self.latent_loss_t_min = float(latent_loss_t_min)
        self.batched_cache_decode = bool(batched_cache_decode)
        self.latent_self_conditioning = latent_self_conditioning or {'mode': 'none'}
        self.latent_self_conditioning_mode = self.latent_self_conditioning.get('mode', 'none')
        self.latent_self_conditioning_upsample_factor = self.latent_self_conditioning.get('upsample_factor', None)
        if self.latent_loss_parameterization not in ('z0', 'velocity'):
            raise ValueError(
                "latent_loss_parameterization must be 'z0' or 'velocity', "
                f"got {self.latent_loss_parameterization}"
            )
        if self.latent_loss_t_min <= 0:
            raise ValueError(f'latent_loss_t_min must be positive, got {self.latent_loss_t_min}')
        if self.latent_self_conditioning_mode not in ('none', 'input', 'bottleneck'):
            raise ValueError(
                "latent_self_conditioning.mode must be 'none', 'input', or 'bottleneck', "
                f"got {self.latent_self_conditioning_mode}"
            )
        super().__init__(*args, **kwargs)

    def _build_frozen_decoder(self):
        cfg = self.decoder_model_config
        self.decoder = getattr(models, cfg['name'])(**cfg['args']).to(self.device)
        ckpt = torch.load(self.decoder_ckpt, map_location=self.device, weights_only=True)
        self.decoder.load_state_dict(ckpt)
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad_(False)

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Latent loss parameterization: {self.latent_loss_parameterization}',
            f'  - Latent loss t min: {self.latent_loss_t_min}',
            f'  - Batched cache decode: {self.batched_cache_decode}',
            f'  - Latent self-conditioning: {self.latent_self_conditioning}',
        ]
        return '\n'.join(lines)

    def _load_latent_cache(self, cache_path: str) -> Dict[str, Any]:
        try:
            return torch.load(cache_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(cache_path, map_location=self.device)

    @staticmethod
    def _shape_to_size(shape: Any) -> torch.Size:
        if isinstance(shape, torch.Size):
            return shape
        if isinstance(shape, (tuple, list)):
            return torch.Size([int(v) for v in shape])
        raise ValueError(f'Unsupported cached shape type: {type(shape)}')

    def _merge_latent_spatial_cache(
        self,
        z: sp.SparseTensor,
        caches: List[Dict[str, Any]],
    ) -> sp.SparseTensor:
        if len(caches) != z.shape[0]:
            raise ValueError(f'Expected {z.shape[0]} latent caches, got {len(caches)}')
        if len(caches) == 0:
            return z

        scale = caches[0]['scale']
        for cache in caches[1:]:
            if cache['scale'] != scale:
                raise ValueError(f'Cannot merge latent caches with different scales: {scale} vs {cache["scale"]}')

        merged_cache = {}
        first_spatial_cache = caches[0]['spatial_cache']
        for scale_key in first_spatial_cache.keys():
            scale_entries = [cache['spatial_cache'].get(scale_key, {}) for cache in caches]
            expected_keys = set(scale_entries[0].keys())
            for entries in scale_entries[1:]:
                if set(entries.keys()) != expected_keys:
                    raise ValueError(
                        f'Cannot merge latent caches with different keys at scale {scale_key}: '
                        f'{expected_keys} vs {set(entries.keys())}'
                    )

            merged_entries = {}
            for key in expected_keys:
                values = [entries[key] for entries in scale_entries]
                if key == 'shape':
                    shapes = [self._shape_to_size(value) for value in values]
                    merged_entries[key] = torch.Size([
                        max(shape[dim] for shape in shapes)
                        for dim in range(len(shapes[0]))
                    ])
                elif key.startswith('channel2spatial_'):
                    coords_list = []
                    idx_list = []
                    subidx_list = []
                    idx_offset = 0
                    for batch_idx, value in enumerate(values):
                        if not (isinstance(value, tuple) and len(value) == 3):
                            raise ValueError(f'Cache key {key} must be a (coords, idx, subidx) tuple')
                        coords, idx, subidx = value
                        coords = coords.to(device=z.device).clone()
                        idx = idx.to(device=z.device)
                        subidx = subidx.to(device=z.device)
                        coords[:, 0] += batch_idx
                        coords_list.append(coords)
                        idx_list.append(idx + idx_offset)
                        subidx_list.append(subidx)
                        idx_offset += int(idx.max().item()) + 1 if idx.numel() > 0 else 0
                    merged_entries[key] = (
                        torch.cat(coords_list, dim=0),
                        torch.cat(idx_list, dim=0),
                        torch.cat(subidx_list, dim=0),
                    )
                else:
                    raise ValueError(
                        f'Unsupported cached key {key}. Batched cache decode only merges '
                        "'shape' and 'channel2spatial_*' entries."
                    )
            merged_cache[scale_key] = merged_entries

        out = z.replace(z.feats)
        out._scale = scale
        out._spatial_cache = merged_cache
        return out

    @torch.no_grad()
    def _decode_latents_with_cache(
        self,
        z: sp.SparseTensor,
        caches: List[Dict[str, Any]] = None,
        cache_paths: List[str] = None,
    ) -> sp.SparseTensor:
        if caches is None:
            if cache_paths is None:
                raise ValueError('Either caches or cache_paths must be provided.')
            caches = [self._load_latent_cache(path) for path in cache_paths]
        if len(caches) != z.shape[0]:
            raise ValueError(f'Expected {z.shape[0]} latent caches, got {len(caches)}')

        if self.batched_cache_decode:
            decoded = self.decoder(self._merge_latent_spatial_cache(z, caches))
            decoded.clear_spatial_cache()
            return decoded

        decoded = []
        for i, cache in enumerate(caches):
            zi = z[i]
            zi._scale = cache['scale']
            zi._spatial_cache = cache['spatial_cache']
            decoded.append(self.decoder(zi))
        decoded = sp.sparse_cat(decoded, dim=0)
        decoded.clear_spatial_cache()
        return decoded

    def _diffuse_latent(
        self,
        z_0: sp.SparseTensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[sp.SparseTensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(z_0.feats)
        batch_t = t[z_0.coords[:, 0].long()].reshape(-1, 1)
        z_t = (1.0 - batch_t) * z_0.feats + batch_t * noise
        return z_0.replace(z_t), noise

    def _latent_reconstruction_loss(
        self,
        pred_z0: sp.SparseTensor,
        z_0: sp.SparseTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        err = pred_z0.feats - z_0.feats
        if self.latent_loss_parameterization == 'velocity':
            denom = t[z_0.coords[:, 0].long()].reshape(-1, 1).clamp_min(self.latent_loss_t_min)
            err = err / denom

        if self.loss_type == 'l1':
            return err.abs().mean()
        if self.loss_type == 'l2':
            return err.pow(2).mean()
        raise ValueError(f'Invalid loss type {self.loss_type}')

    @staticmethod
    def _coords_to_keys(coords: torch.Tensor, spatial_size: torch.Tensor) -> torch.Tensor:
        coords = coords.long()
        sx, sy, sz = spatial_size.long()
        return (((coords[:, 0] * sx + coords[:, 1]) * sy + coords[:, 2]) * sz + coords[:, 3])

    def _infer_latent_to_field_factor(self, z_t: sp.SparseTensor, x_t: sp.SparseTensor) -> int:
        if self.latent_self_conditioning_upsample_factor is not None:
            return int(self.latent_self_conditioning_upsample_factor)
        up_block_type = self.decoder_model_config.get('args', {}).get('up_block_type', [])
        if len(up_block_type) == 0:
            raise ValueError(
                'latent_self_conditioning.upsample_factor must be set when decoder_model.args.up_block_type is unavailable.'
            )
        return 2 ** len(up_block_type)

    def _latent_to_field_support(self, z_t: sp.SparseTensor, x_t: sp.SparseTensor) -> Tuple[sp.SparseTensor, torch.Tensor]:
        factor = self._infer_latent_to_field_factor(z_t, x_t)
        field_parent = torch.div(x_t.coords[:, 1:], factor, rounding_mode='floor')
        query_coords = torch.cat([x_t.coords[:, 0:1], field_parent], dim=1)
        source_coords = z_t.coords

        max_spatial = torch.maximum(
            query_coords[:, 1:].amax(dim=0),
            source_coords[:, 1:].amax(dim=0),
        ) + 1
        query_keys = self._coords_to_keys(query_coords, max_spatial)
        source_keys = self._coords_to_keys(source_coords, max_spatial)
        order = torch.argsort(source_keys)
        sorted_keys = source_keys[order]
        sorted_feats = z_t.feats[order]
        idx = torch.searchsorted(sorted_keys, query_keys)
        valid = (
            (idx < sorted_keys.numel()) &
            (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == query_keys)
        )
        feats = torch.zeros(
            (x_t.feats.shape[0], z_t.feats.shape[1]),
            dtype=z_t.feats.dtype,
            device=z_t.feats.device,
        )
        if valid.any():
            feats[valid] = sorted_feats[idx[valid]]
        missing = 1.0 - torch.segment_reduce(valid.float(), reduce='mean', lengths=x_t.seqlen)
        return x_t.replace(feats), missing

    def training_losses(
        self,
        z_0: sp.SparseTensor,
        cond: sp.SparseTensor,
        triangle_field_slat_cache: List[Dict[str, Any]] = None,
        triangle_field_slat_cache_path: List[str] = None,
        x_0: sp.SparseTensor = None,
        missing_low_parent_frac: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        t = self.sample_t(z_0.shape[0]).to(z_0.feats.device).float()
        z_t, _ = self._diffuse_latent(z_0, t)
        x_t = self._decode_latents_with_cache(
            z_t,
            caches=triangle_field_slat_cache,
            cache_paths=triangle_field_slat_cache_path,
        )
        if not torch.equal(x_t.coords, cond.coords):
            raise ValueError(
                f'Decoded z_t coords must match cond coords, got {x_t.coords.shape} vs {cond.coords.shape}'
            )

        cond = self._augment_conditioning(cond)
        cond, cond_drop = self._drop_cond(cond)
        latent_cond_missing = None
        if self.latent_self_conditioning_mode == 'input':
            latent_cond_input, latent_cond_missing = self._latent_to_field_support(z_t, x_t)
            enc_in = self._make_encoder_input(x_t, cond)
            enc_in = enc_in.replace(torch.cat([enc_in.feats, latent_cond_input.feats], dim=-1))
            encoder_kwargs = {}
        elif self.latent_self_conditioning_mode == 'bottleneck':
            enc_in = self._make_encoder_input(x_t, cond)
            encoder_kwargs = {'latent_cond': z_t}
        else:
            enc_in = self._make_encoder_input(x_t, cond)
            encoder_kwargs = {}
        pred_z0 = self.training_models['encoder'](
            enc_in,
            t * 1000.0,
            sample_posterior=self.sample_posterior,
            **encoder_kwargs,
        )
        if not torch.equal(pred_z0.coords, z_0.coords):
            raise ValueError(
                f'Predicted latent coords must match z_0 coords, got {pred_z0.coords.shape} vs {z_0.coords.shape}'
            )

        terms = edict()
        loss_name = f'latent_{self.loss_type}'
        terms[loss_name] = self._latent_reconstruction_loss(pred_z0, z_0, t)
        terms['loss'] = terms[loss_name]

        status = {
            'cond/drop_frac': cond_drop.float().mean(),
        }
        if missing_low_parent_frac is not None:
            missing_low_parent_frac = missing_low_parent_frac.float()
            status['cond/missing_low_parent_frac'] = missing_low_parent_frac.mean()
            status['cond/missing_low_parent_frac_max'] = missing_low_parent_frac.max()
        if latent_cond_missing is not None:
            status['latent_cond/missing_parent_frac'] = latent_cond_missing.mean()
            status['latent_cond/missing_parent_frac_max'] = latent_cond_missing.max()

        with torch.no_grad():
            raw_l1 = (pred_z0.feats - z_0.feats).abs()
            status['latent/raw_l1'] = raw_l1.mean()
            status['latent/raw_l2'] = (pred_z0.feats - z_0.feats).pow(2).mean()
            status['latent/pred_abs_mean'] = pred_z0.feats.abs().mean()
            status['latent/target_abs_mean'] = z_0.feats.abs().mean()
            time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
            err = torch.segment_reduce(
                (pred_z0.feats - z_0.feats).pow(2).mean(dim=1),
                reduce='mean',
                lengths=z_0.seqlen,
            ).detach().cpu().numpy()
            for i in range(10):
                if (time_bin == i).sum() != 0:
                    terms[f'bin_{i}'] = {'latent_mse': err[time_bin == i].mean()}

        return terms, status

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        iterator = iter(dataloader)

        gt_images = {}
        cond_images = {}
        noisy_images = {}
        pred_images = {}
        self.models['encoder'].eval()
        self.decoder.eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iterator)
            args = {k: v[:batch] if hasattr(v, '__getitem__') and not isinstance(v, str) else v for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            t = torch.full((args['z_0'].shape[0],), self.snapshot_t, device=self.device)
            z_t, _ = self._diffuse_latent(args['z_0'], t)
            gt = self._decode_latents_with_cache(
                args['z_0'],
                caches=args.get('triangle_field_slat_cache', None),
                cache_paths=args.get('triangle_field_slat_cache_path', None),
            )
            x_t = self._decode_latents_with_cache(
                z_t,
                caches=args.get('triangle_field_slat_cache', None),
                cache_paths=args.get('triangle_field_slat_cache_path', None),
            )
            if self.latent_self_conditioning_mode == 'input':
                latent_cond_input, _ = self._latent_to_field_support(z_t, x_t)
                enc_in = self._make_encoder_input(x_t, args['cond'])
                enc_in = enc_in.replace(torch.cat([enc_in.feats, latent_cond_input.feats], dim=-1))
                encoder_kwargs = {}
            elif self.latent_self_conditioning_mode == 'bottleneck':
                enc_in = self._make_encoder_input(x_t, args['cond'])
                encoder_kwargs = {'latent_cond': z_t}
            else:
                enc_in = self._make_encoder_input(x_t, args['cond'])
                encoder_kwargs = {}
            pred_z0 = self.models['encoder'](
                enc_in,
                t * 1000.0,
                sample_posterior=False,
                **encoder_kwargs,
            )
            y = self._decode_latents_with_cache(
                pred_z0,
                caches=args.get('triangle_field_slat_cache', None),
                cache_paths=args.get('triangle_field_slat_cache_path', None),
            )

            gt_vis = self.dataset.visualize_sample({'target': gt})
            cond_vis = self.dataset.visualize_sample({'target': args['cond']})
            noisy_vis = self.dataset.visualize_sample({'target': x_t})
            pred_vis = self.dataset.visualize_sample({'target': y})
            for k, v in gt_vis.items():
                gt_images.setdefault(k, []).append(v[:batch])
            for k, v in cond_vis.items():
                cond_images.setdefault(k, []).append(v[:batch])
            for k, v in noisy_vis.items():
                noisy_images.setdefault(k, []).append(v[:batch])
            for k, v in pred_vis.items():
                pred_images.setdefault(k, []).append(v[:batch])

        self.models['encoder'].train()

        sample_dict = {}
        for k in gt_images:
            sample_dict[f'gt_{k}'] = {'value': torch.cat(gt_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'cond_{k}'] = {'value': torch.cat(cond_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'noisy_{k}'] = {'value': torch.cat(noisy_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'pred_{k}'] = {'value': torch.cat(pred_images[k], dim=0)[:num_samples], 'type': 'image'}
        return sample_dict
