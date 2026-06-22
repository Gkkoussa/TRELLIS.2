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
        if not (0.0 <= self.cond_drop_prob <= 1.0):
            raise ValueError(f'cond_drop_prob must be in [0, 1], got {self.cond_drop_prob}')
        if self.loss_type not in ('l1', 'l2'):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {self.loss_type}")
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
        ]
        return '\n'.join(lines)

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
