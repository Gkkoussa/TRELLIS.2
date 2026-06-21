from typing import Dict, Tuple

import copy
import functools
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from .pbr_vae import PbrVaeTrainer
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler


class TriangleFieldVaeTrainer(PbrVaeTrainer):
    """
    VAE trainer for triangle-field voxels.

    The encoder consumes the full triangle-field input tensor `x`, while the
    decoder reconstructs only the two-channel `target` tensor: d_tri and d_vert.
    """

    def __init__(
        self,
        *args,
        num_workers: int = None,
        loss_type: str = 'l1',
        lambda_kl: float = 1e-6,
        lambda_subdiv: float = 0.0,
        debug_nans: bool = False,
        voxel_loss_weight: dict = None,
        aux_feature_dropout: dict = None,
        **kwargs,
    ):
        self.debug_nans = debug_nans
        self.voxel_loss_weight = voxel_loss_weight
        self.aux_feature_dropout = aux_feature_dropout
        self.lambda_subdiv = lambda_subdiv
        self._zero_decoder_grad_for_step = False
        self._decoder_grad_hooks = []
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
        self._validate_aux_feature_dropout()
        lambda_render = kwargs.pop('lambda_render', 0.0)
        if lambda_render != 0.0:
            raise ValueError('TriangleFieldVaeTrainer does not support render loss; set lambda_render to 0.0 or omit it.')
        super().__init__(
            *args,
            num_workers=num_workers,
            loss_type=loss_type,
            lambda_kl=lambda_kl,
            lambda_render=0.0,
            **kwargs,
        )
        self._register_decoder_freeze_hooks()

    def _validate_aux_feature_dropout(self) -> None:
        if self.aux_feature_dropout is None:
            return
        enabled = self.aux_feature_dropout.get('enabled', True)
        if not enabled:
            return
        full_prob = float(self.aux_feature_dropout.get('full_prob', 0.5))
        drop_all_prob = float(self.aux_feature_dropout.get('drop_all_prob', 0.25))
        drop_group_prob = float(self.aux_feature_dropout.get('drop_group_prob', 0.25))
        if min(full_prob, drop_all_prob, drop_group_prob) < 0:
            raise ValueError('aux_feature_dropout probabilities must be non-negative.')
        total = full_prob + drop_all_prob + drop_group_prob
        if total <= 0:
            raise ValueError('At least one aux_feature_dropout probability must be positive.')
        group_drop_prob = float(self.aux_feature_dropout.get('group_drop_prob', 0.5))
        if not (0 < group_drop_prob <= 1):
            raise ValueError('aux_feature_dropout.group_drop_prob must be in (0, 1].')
        freeze_decoder = bool(self.aux_feature_dropout.get('freeze_decoder_on_drop', True))
        if not freeze_decoder:
            raise ValueError('Only freeze_decoder_on_drop=True is supported for this trainer option.')

    def _register_decoder_freeze_hooks(self) -> None:
        if self.aux_feature_dropout is None or not self.aux_feature_dropout.get('enabled', True):
            return
        decoder = self.training_models.get('decoder', None)
        if decoder is None:
            return

        def maybe_zero_decoder_grad(grad):
            if self._zero_decoder_grad_for_step:
                return torch.zeros_like(grad)
            return grad

        self._decoder_grad_hooks = [
            param.register_hook(maybe_zero_decoder_grad)
            for param in decoder.parameters()
            if param.requires_grad
        ]

    def _aux_dropout_groups(self):
        layout = getattr(self.dataset, 'input_layout', None)
        target_channels = getattr(self.dataset, 'num_target_channels', 2)
        if layout is None:
            return [('aux', slice(target_channels, None))]

        configured = None
        if self.aux_feature_dropout is not None:
            configured = self.aux_feature_dropout.get('groups', None)

        groups = []
        for name, slc in layout.items():
            if slc.stop <= target_channels:
                continue
            if configured is not None and name not in configured:
                continue
            groups.append((name, slc))
        return groups

    def _sample_aux_dropout_mode(self):
        if self.aux_feature_dropout is None or not self.aux_feature_dropout.get('enabled', True):
            return 'full', []

        full_prob = float(self.aux_feature_dropout.get('full_prob', 0.5))
        drop_all_prob = float(self.aux_feature_dropout.get('drop_all_prob', 0.25))
        drop_group_prob = float(self.aux_feature_dropout.get('drop_group_prob', 0.25))
        total = full_prob + drop_all_prob + drop_group_prob
        full_prob /= total
        drop_all_prob /= total

        seed = int(self.aux_feature_dropout.get('seed', 0)) + int(self.step)
        rng = np.random.default_rng(seed)
        r = rng.random()
        groups = self._aux_dropout_groups()
        if r < full_prob or not groups:
            return 'full', []
        if r < full_prob + drop_all_prob:
            return 'drop_all', groups

        group_drop_prob = float(self.aux_feature_dropout.get('group_drop_prob', 0.5))
        selected = [group for group in groups if rng.random() < group_drop_prob]
        if not selected:
            selected = [groups[int(rng.integers(0, len(groups)))]]
        return 'drop_groups', selected

    def _apply_aux_feature_dropout(self, x: sp.SparseTensor):
        mode, groups = self._sample_aux_dropout_mode()
        self._zero_decoder_grad_for_step = mode != 'full'
        if mode == 'full':
            return x, mode, []

        feats = x.feats.clone()
        for _, slc in groups:
            feats[:, slc] = 0
        return x.replace(feats), mode, [name for name, _ in groups]

    def _debug_tensor_stats(self, name: str, tensor: torch.Tensor) -> Dict[str, float]:
        tensor = tensor.detach().float()
        finite = torch.isfinite(tensor)
        stats = {
            f'{name}/finite_frac': finite.float().mean().item(),
            f'{name}/has_nan': torch.isnan(tensor).any().float().item(),
            f'{name}/has_inf': torch.isinf(tensor).any().float().item(),
        }
        if finite.any().item():
            finite_tensor = tensor[finite]
            stats.update({
                f'{name}/min': finite_tensor.min().item(),
                f'{name}/max': finite_tensor.max().item(),
                f'{name}/mean': finite_tensor.mean().item(),
                f'{name}/std': finite_tensor.std(unbiased=False).item(),
                f'{name}/absmax': finite_tensor.abs().max().item(),
            })
        else:
            stats.update({
                f'{name}/min': float('nan'),
                f'{name}/max': float('nan'),
                f'{name}/mean': float('nan'),
                f'{name}/std': float('nan'),
                f'{name}/absmax': float('nan'),
            })
        return stats

    def _debug_sparse_stats(self, name: str, tensor: sp.SparseTensor) -> Dict[str, float]:
        stats = self._debug_tensor_stats(name, tensor.feats)
        stats[f'{name}/tokens'] = float(tensor.feats.shape[0])
        stats[f'{name}/channels'] = float(tensor.feats.shape[1])
        return stats

    def _debug_abort_if_nonfinite(self, stage: str, status: Dict[str, float]) -> None:
        if not self.debug_nans:
            return
        bad_keys = [
            key for key, value in status.items()
            if (key.endswith('/has_nan') or key.endswith('/has_inf')) and value != 0.0
        ]
        if not bad_keys:
            return
        lines = [
            f'Non-finite value detected in TriangleFieldVaeTrainer at step={self.step} rank={self.rank} stage={stage}.',
            f'Bad flags: {bad_keys}',
        ]
        for key in sorted(status):
            if (
                key.endswith('/finite_frac') or
                key.endswith('/min') or
                key.endswith('/max') or
                key.endswith('/mean') or
                key.endswith('/std') or
                key.endswith('/absmax') or
                key.endswith('/has_nan') or
                key.endswith('/has_inf')
            ):
                lines.append(f'  {key}: {status[key]}')
        raise RuntimeError('\n'.join(lines))

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.

        This matches PbrVaeTrainer so existing sparse voxel batching behavior is
        unchanged.
        """
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

    def _inverse_triangle_area_weight(self, x: sp.SparseTensor) -> torch.Tensor:
        """
        Compute one reconstruction-loss weight per sparse voxel from the three
        stored vertex offsets. The offsets share the projected surface point as
        origin, so offset differences recover triangle edge vectors.
        """
        if x.feats.shape[1] < 11:
            raise ValueError(
                f'inverse_triangle_area weighting requires x features through channel 10, got {x.feats.shape[1]}'
            )

        cfg = self.voxel_loss_weight or {}
        eps = float(cfg.get('eps', 1e-8))
        clamp_max = cfg.get('clamp_max', None)
        normalize = cfg.get('normalize', 'mean')

        with torch.autocast(device_type='cuda', enabled=False):
            offset0 = x.feats[:, 2:5].float()
            offset1 = x.feats[:, 5:8].float()
            offset2 = x.feats[:, 8:11].float()
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
        target: torch.Tensor,
        pred: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.loss_type == 'l1':
            err = torch.abs(target - pred)
        elif self.loss_type == 'l2':
            err = (target - pred) ** 2
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

        if weight is not None:
            err = err * weight.reshape(-1, 1).to(device=err.device, dtype=err.dtype)
        return err.mean()

    @staticmethod
    def _module_attr(model, name: str, default=None):
        module = model.module if hasattr(model, 'module') else model
        return getattr(module, name, default)

    @staticmethod
    def _subdivision_metrics(sub_gt: torch.Tensor, sub: sp.SparseTensor, prefix: str) -> Dict[str, float]:
        gt = sub_gt.bool()
        pred = sub.feats > 0
        correct = pred == gt
        tp = (pred & gt).float().sum()
        fp = (pred & ~gt).float().sum()
        fn = (~pred & gt).float().sum()
        eps = torch.tensor(1e-8, device=pred.device)
        return {
            f'{prefix}/acc': correct.float().mean().item(),
            f'{prefix}/precision': (tp / (tp + fp + eps)).item(),
            f'{prefix}/recall': (tp / (tp + fn + eps)).item(),
            f'{prefix}/iou': (tp / (tp + fp + fn + eps)).item(),
            f'{prefix}/perfect_parent': correct.all(dim=1).float().mean().item(),
            f'{prefix}/gt_active_frac': gt.float().mean().item(),
            f'{prefix}/pred_active_frac': pred.float().mean().item(),
        }

    def training_losses(
        self,
        x: sp.SparseTensor,
        target: sp.SparseTensor,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        Compute losses with asymmetric input and target tensors.

        Args:
            x: Full encoder input features.
            target: Two-channel reconstruction target containing d_tri, d_vert.
        """
        status = {}
        if self.debug_nans:
            status.update(self._debug_sparse_stats('x', x))
            status.update(self._debug_sparse_stats('target', target))
            self._debug_abort_if_nonfinite('input', status)

        x, aux_dropout_mode, aux_dropout_groups = self._apply_aux_feature_dropout(x)
        if self.aux_feature_dropout is not None and self.aux_feature_dropout.get('enabled', True):
            status['aux_dropout/mode_full'] = float(aux_dropout_mode == 'full')
            status['aux_dropout/mode_drop_all'] = float(aux_dropout_mode == 'drop_all')
            status['aux_dropout/mode_drop_groups'] = float(aux_dropout_mode == 'drop_groups')
            status['aux_dropout/decoder_frozen'] = float(self._zero_decoder_grad_for_step)
            status['aux_dropout/num_groups_dropped'] = float(len(aux_dropout_groups))

        z, mean, logvar = self.training_models['encoder'](x, sample_posterior=True, return_raw=True)
        if self.debug_nans:
            with torch.autocast(device_type='cuda', enabled=False):
                mean_f = mean.detach().float()
                logvar_f = logvar.detach().float()
                std_f = torch.exp(0.5 * logvar_f)
            status.update(self._debug_tensor_stats('posterior/mean', mean))
            status.update(self._debug_tensor_stats('posterior/logvar', logvar))
            status.update(self._debug_tensor_stats('posterior/std', std_f))
            status.update(self._debug_sparse_stats('posterior/z', z))
            self._debug_abort_if_nonfinite('posterior', status)

        decoder_predicts_subdiv = bool(self._module_attr(self.training_models['decoder'], 'pred_subdiv', False))
        if decoder_predicts_subdiv:
            y, subs_gt, subs = self.training_models['decoder'](z)
        else:
            y = self.training_models['decoder'](z)
            subs_gt, subs = [], []
        if self.debug_nans:
            status.update(self._debug_sparse_stats('decoder/y', y))
            self._debug_abort_if_nonfinite('decoder', status)

        if y.feats.shape != target.feats.shape:
            raise ValueError(
                f'Decoder output shape must match target shape, got '
                f'{tuple(y.feats.shape)} vs {tuple(target.feats.shape)}. '
                f'Check decoder out_channels and dataset target channels.'
            )

        terms = edict(loss=0.0)
        loss_weight = None
        if self.voxel_loss_weight is not None:
            loss_weight = self._inverse_triangle_area_weight(x)
            if self.debug_nans:
                status.update(self._debug_tensor_stats('loss_weight', loss_weight))
                self._debug_abort_if_nonfinite('loss_weight', status)

        if self.loss_type == 'l1':
            terms['l1'] = self._reconstruction_loss(target.feats, y.feats, loss_weight)
            if self.debug_nans:
                status.update(self._debug_tensor_stats('loss/l1', terms['l1'].reshape(1)))
            terms['loss'] = terms['loss'] + terms['l1']
        elif self.loss_type == 'l2':
            terms['l2'] = self._reconstruction_loss(target.feats, y.feats, loss_weight)
            if self.debug_nans:
                status.update(self._debug_tensor_stats('loss/l2', terms['l2'].reshape(1)))
            terms['loss'] = terms['loss'] + terms['l2']
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

        for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
            terms[f'bce_sub{i}'] = F.binary_cross_entropy_with_logits(sub.feats, sub_gt.float())
            terms['loss'] = terms['loss'] + self.lambda_subdiv * terms[f'bce_sub{i}']
            status.update(self._subdivision_metrics(sub_gt, sub, f'subdiv/sub{i}'))
        if subs:
            status['subdiv/num_stages'] = float(len(subs))

        terms['kl'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms['loss'] = terms['loss'] + self.lambda_kl * terms['kl']
        if self.debug_nans:
            status.update(self._debug_tensor_stats('loss/kl', terms['kl'].reshape(1)))
            status.update(self._debug_tensor_stats('loss/total', terms['loss'].reshape(1)))
            self._debug_abort_if_nonfinite('loss', status)

        return terms, status

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        snapshot_dataset = copy.deepcopy(self.dataset)
        dataloader = DataLoader(
            snapshot_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        gt_images = {}
        pred_images = {}
        self.models['encoder'].eval()
        self.models['decoder'].eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            z = self.models['encoder'](args['x'])
            y = self.models['decoder'](z)
            decoder_predicts_subdiv = bool(self._module_attr(self.models['decoder'], 'pred_subdiv', False))
            if not decoder_predicts_subdiv and y.feats.shape != args['target'].feats.shape:
                raise ValueError(
                    f'Decoder output shape must match target shape, got '
                    f'{tuple(y.feats.shape)} vs {tuple(args["target"].feats.shape)}.'
                )

            gt_vis = self.dataset.visualize_sample({'target': args['target']})
            pred_vis = self.dataset.visualize_sample({'target': y})
            for k, v in gt_vis.items():
                gt_images.setdefault(k, []).append(v[:batch])
            for k, v in pred_vis.items():
                pred_images.setdefault(k, []).append(v[:batch])
        self.models['encoder'].train()
        self.models['decoder'].train()

        sample_dict = {}
        for k in gt_images:
            sample_dict[f'gt_{k}'] = {'value': torch.cat(gt_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'pred_{k}'] = {'value': torch.cat(pred_images[k], dim=0)[:num_samples], 'type': 'image'}
        return sample_dict
