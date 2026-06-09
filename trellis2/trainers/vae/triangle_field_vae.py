from typing import Dict, Tuple

import copy
import functools
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from .pbr_vae import PbrVaeTrainer
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from ...utils.loss_utils import l1_loss, l2_loss


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
        debug_nans: bool = False,
        **kwargs,
    ):
        self.debug_nans = debug_nans
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

        y = self.training_models['decoder'](z)
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

        if self.loss_type == 'l1':
            terms['l1'] = l1_loss(target.feats, y.feats)
            if self.debug_nans:
                status.update(self._debug_tensor_stats('loss/l1', terms['l1'].reshape(1)))
            terms['loss'] = terms['loss'] + terms['l1']
        elif self.loss_type == 'l2':
            terms['l2'] = l2_loss(target.feats, y.feats)
            if self.debug_nans:
                status.update(self._debug_tensor_stats('loss/l2', terms['l2'].reshape(1)))
            terms['loss'] = terms['loss'] + terms['l2']
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

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
            if y.feats.shape != args['target'].feats.shape:
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
