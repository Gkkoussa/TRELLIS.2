from typing import Dict, Tuple

import copy
import functools
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from .pbr_vae import PbrVaeTrainer
from ...modules import sparse as sp
from ...utils.data_utils import (
    recursive_to_device,
    cycle,
    BalancedResumableSampler,
)
from ...utils.dist_utils import read_file_dist


class TriangleFieldVaeTrainer(PbrVaeTrainer):
    """
    VAE trainer for triangle-field voxels.

    The encoder consumes the full triangle-field input tensor `x`, while the
    decoder reconstructs the channels selected by the dataset target layout.
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
        partial_load_expanded_io: bool = False,
        sample_balanced_loss: bool = False,
        batch_size_per_gpu_by_resolution: dict = None,
        resolution_sampling_weights: dict = None,
        validation_dataset=None,
        lambda_vertex: float = 0.0,
        vertex_training_resolutions: list = None,
        lambda_edge: float = 0.0,
        edge_training_resolutions: list = None,
        edge_negative_pairs: int = 50,
        edge_nearby_negative_pairs: int = 25,
        edge_knn_k: int = 32,
        edge_threshold: float = 0.5,
        vertex_child_pos_weight_by_stage: dict = None,
        vertex_finest_loss: str = 'weighted_bce',
        vertex_asymmetric_gamma_negative: float = 4.0,
        vertex_asymmetric_gamma_positive: float = 0.0,
        vertex_threshold: float = 0.5,
        vertex_snapshot_samples: int = 4,
        vertex_snapshot_stage_samples: int = 1,
        vertex_snapshot_image_resolution: int = 256,
        vertex_snapshot_ssaa: int = 2,
        **kwargs,
    ):
        self.debug_nans = debug_nans
        self.voxel_loss_weight = voxel_loss_weight
        self.aux_feature_dropout = aux_feature_dropout
        self.partial_load_expanded_io = partial_load_expanded_io
        self.lambda_subdiv = lambda_subdiv
        self.sample_balanced_loss = bool(sample_balanced_loss)
        self.batch_size_per_gpu_by_resolution = (
            {int(key): int(value) for key, value in batch_size_per_gpu_by_resolution.items()}
            if batch_size_per_gpu_by_resolution is not None else None
        )
        self.resolution_sampling_weights = (
            {int(key): float(value) for key, value in resolution_sampling_weights.items()}
            if resolution_sampling_weights is not None else None
        )
        self.validation_dataset = validation_dataset
        self.lambda_vertex = float(lambda_vertex)
        self.vertex_training_resolutions = (
            None
            if vertex_training_resolutions is None
            else tuple(sorted({int(value) for value in vertex_training_resolutions}))
        )
        self.lambda_edge = float(lambda_edge)
        self.edge_training_resolutions = (
            None
            if edge_training_resolutions is None
            else tuple(sorted({int(value) for value in edge_training_resolutions}))
        )
        self.edge_negative_pairs = int(edge_negative_pairs)
        self.edge_nearby_negative_pairs = int(edge_nearby_negative_pairs)
        self.edge_knn_k = int(edge_knn_k)
        self.edge_threshold = float(edge_threshold)
        self.vertex_child_pos_weight_by_stage = {
            int(stage): float(weight)
            for stage, weight in (vertex_child_pos_weight_by_stage or {}).items()
        }
        self.vertex_finest_loss = str(vertex_finest_loss).lower()
        self.vertex_asymmetric_gamma_negative = float(vertex_asymmetric_gamma_negative)
        self.vertex_asymmetric_gamma_positive = float(vertex_asymmetric_gamma_positive)
        self.vertex_threshold = float(vertex_threshold)
        self.vertex_snapshot_samples = int(vertex_snapshot_samples)
        self.vertex_snapshot_stage_samples = int(vertex_snapshot_stage_samples)
        self.vertex_snapshot_image_resolution = int(vertex_snapshot_image_resolution)
        self.vertex_snapshot_ssaa = int(vertex_snapshot_ssaa)
        if self.lambda_vertex < 0:
            raise ValueError('lambda_vertex must be non-negative.')
        if self.lambda_edge < 0:
            raise ValueError('lambda_edge must be non-negative.')
        if (
            self.vertex_training_resolutions is not None
            and (
                len(self.vertex_training_resolutions) == 0
                or any(value <= 0 for value in self.vertex_training_resolutions)
            )
        ):
            raise ValueError(
                'vertex_training_resolutions must contain positive resolutions, '
                'or be omitted to supervise every resolution.'
            )
        if (
            self.edge_training_resolutions is not None
            and (
                len(self.edge_training_resolutions) == 0
                or any(value <= 0 for value in self.edge_training_resolutions)
            )
        ):
            raise ValueError(
                'edge_training_resolutions must contain positive resolutions, '
                'or be omitted to supervise every resolution.'
            )
        if self.edge_negative_pairs <= 0:
            raise ValueError('edge_negative_pairs must be positive.')
        if not (0 <= self.edge_nearby_negative_pairs <= self.edge_negative_pairs):
            raise ValueError(
                'edge_nearby_negative_pairs must be between zero and '
                'edge_negative_pairs.'
            )
        if self.edge_knn_k <= 0:
            raise ValueError('edge_knn_k must be positive.')
        if self.edge_nearby_negative_pairs > self.edge_knn_k:
            raise ValueError(
                'edge_nearby_negative_pairs cannot exceed edge_knn_k.'
            )
        if not (0.0 < self.edge_threshold < 1.0):
            raise ValueError('edge_threshold must be in (0, 1).')
        if not (0.0 < self.vertex_threshold < 1.0):
            raise ValueError('vertex_threshold must be in (0, 1).')
        if any(weight <= 0 for weight in self.vertex_child_pos_weight_by_stage.values()):
            raise ValueError('All vertex child positive weights must be positive.')
        if any(stage < 0 for stage in self.vertex_child_pos_weight_by_stage):
            raise ValueError('Vertex child positive-weight stage indices must be non-negative.')
        if self.vertex_finest_loss not in ('weighted_bce', 'asymmetric'):
            raise ValueError(
                "vertex_finest_loss must be either 'weighted_bce' or 'asymmetric'."
            )
        if (
            self.vertex_asymmetric_gamma_negative < 0 or
            self.vertex_asymmetric_gamma_positive < 0
        ):
            raise ValueError('Asymmetric-loss gamma values must be non-negative.')
        if self.vertex_snapshot_samples < 0 or self.vertex_snapshot_stage_samples < 0:
            raise ValueError('Vertex snapshot sample counts must be non-negative.')
        if self.vertex_snapshot_image_resolution <= 0 or self.vertex_snapshot_ssaa <= 0:
            raise ValueError('Vertex snapshot image resolution and SSAA must be positive.')
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
        decoder_predicts_vertex = bool(
            self._module_attr(self.models.get('decoder'), 'pred_vertex_subdiv', False)
        )
        decoder_predicts_edge = bool(
            self._module_attr(self.models.get('decoder'), 'pred_edge', False)
        )
        dataset_has_vertex_targets = bool(getattr(self.dataset, 'vertex_prediction', False))
        dataset_has_edge_targets = bool(getattr(self.dataset, 'edge_prediction', False))
        if decoder_predicts_vertex != dataset_has_vertex_targets:
            raise ValueError(
                'Decoder pred_vertex_subdiv and dataset vertex_prediction must match.'
            )
        if decoder_predicts_vertex and self.lambda_vertex <= 0:
            raise ValueError(
                'A vertex-enabled decoder requires lambda_vertex > 0 so its head is trained.'
            )
        if decoder_predicts_edge != dataset_has_edge_targets:
            raise ValueError(
                'Decoder pred_edge and dataset edge_prediction must match.'
            )
        if decoder_predicts_edge and not decoder_predicts_vertex:
            raise ValueError('Edge prediction requires the hierarchical vertex decoder.')
        if decoder_predicts_edge and self.lambda_edge <= 0:
            raise ValueError(
                'An edge-enabled decoder requires lambda_edge > 0 so its head is trained.'
            )
        if decoder_predicts_edge and self.edge_training_resolutions is None:
            raise ValueError(
                'An edge-enabled decoder requires edge_training_resolutions.'
            )
        dataset_edge_resolutions = tuple(sorted(
            int(value)
            for value in getattr(self.dataset, 'edge_training_resolutions', tuple())
        ))
        if decoder_predicts_edge and dataset_edge_resolutions != self.edge_training_resolutions:
            raise ValueError(
                'Trainer and dataset edge_training_resolutions must match, got '
                f'{self.edge_training_resolutions} and {dataset_edge_resolutions}.'
            )
        self._register_decoder_freeze_hooks()

    def finetune_from(self, finetune_ckpt):
        """Load a base VAE while preserving newly added heads or expanded I/O."""
        decoder_predicts_vertex = bool(
            self._module_attr(self.models.get('decoder'), 'pred_vertex_subdiv', False)
        )
        if not decoder_predicts_vertex and not self.partial_load_expanded_io:
            return super().finetune_from(finetune_ckpt)

        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')

        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name not in finetune_ckpt:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
                continue

            model_ckpt = torch.load(
                read_file_dist(finetune_ckpt[name]),
                map_location=self.device,
                weights_only=True,
            )
            allowed_missing_prefixes = (
                ('vertex_subdiv_heads.',)
                if name == 'decoder' else tuple()
            )
            allowed_missing_prefixes += tuple(
                self._module_attr(model, 'finetune_allowed_missing_prefixes', tuple())
            )
            if self.partial_load_expanded_io:
                allowed_missing = set(model_state_dict)
            else:
                allowed_missing = {
                    key for key in model_state_dict
                    if key.startswith(allowed_missing_prefixes)
                }
            missing = set(model_state_dict) - set(model_ckpt)
            disallowed_missing = missing - allowed_missing
            if disallowed_missing:
                raise RuntimeError(
                    f'{name} finetune checkpoint is missing unexpected parameters: '
                    f'{sorted(disallowed_missing)}'
                )
            for key in sorted(missing):
                model_ckpt[key] = model_state_dict[key]
                if self.is_master:
                    print(f'  - Initialized new parameter: {name}.{key}')

            allowed_unexpected_prefixes = tuple(
                self._module_attr(model, 'finetune_allowed_unexpected_prefixes', tuple())
            )
            unexpected = set(model_ckpt) - set(model_state_dict)
            disallowed_unexpected = (
                set()
                if self.partial_load_expanded_io
                else {
                    key for key in unexpected
                    if not key.startswith(allowed_unexpected_prefixes)
                }
            )
            if disallowed_unexpected:
                raise RuntimeError(
                    f'{name} finetune checkpoint has unexpected parameters: '
                    f'{sorted(disallowed_unexpected)}'
                )
            for key in unexpected - disallowed_unexpected:
                del model_ckpt[key]
                if self.is_master:
                    print(f'  - Ignored legacy parameter: {name}.{key}')
            mismatched = {}
            for key in model_ckpt:
                source = model_ckpt[key]
                target = model_state_dict[key]
                if source.shape == target.shape:
                    continue

                expanded = None
                if (
                    self.partial_load_expanded_io
                    and name == 'encoder'
                    and key == 'input_layer.weight'
                    and source.ndim == target.ndim == 2
                    and source.shape[0] == target.shape[0]
                    and source.shape[1] < target.shape[1]
                ):
                    expanded = torch.zeros_like(target)
                    expanded[:, :source.shape[1]] = source
                elif (
                    self.partial_load_expanded_io
                    and name == 'decoder'
                    and key in ('output_layer.weight', 'output_layer.bias')
                    and source.ndim == target.ndim
                    and source.shape[0] < target.shape[0]
                    and source.shape[1:] == target.shape[1:]
                ):
                    expanded = torch.zeros_like(target)
                    expanded[:source.shape[0]] = source

                if expanded is None:
                    if self.partial_load_expanded_io:
                        model_ckpt[key] = target
                        if self.is_master:
                            print(
                                f'Warning: {name}.{key} shape mismatch '
                                f'{tuple(source.shape)} vs {tuple(target.shape)}; '
                                'left initialized.'
                            )
                    else:
                        mismatched[key] = (tuple(source.shape), tuple(target.shape))
                else:
                    model_ckpt[key] = expanded
                    if self.is_master:
                        print(
                            f'  - Expanded {name}.{key} from {tuple(source.shape)} '
                            f'to {tuple(target.shape)} with zero-initialized new channels.'
                        )
            if mismatched:
                raise RuntimeError(
                    f'{name} finetune checkpoint has shape mismatches: {mismatched}'
                )

            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt)

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
        if layout is None:
            target_channels = getattr(self.dataset, 'num_target_channels', 2)
            return [('aux', slice(target_channels, None))]

        configured = None
        if self.aux_feature_dropout is not None:
            configured = self.aux_feature_dropout.get('groups', None)

        groups = []
        for name, slc in layout.items():
            if name in getattr(self.dataset, 'target_layout', {}):
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
        if self.batch_size_per_gpu_by_resolution is not None:
            from ...utils.data_utils import GroupedBalancedResumableBatchSampler

            if self.batch_split != 1:
                raise ValueError('Resolution-specific batches require batch_split=1.')
            groups = getattr(self.dataset, 'resolution_groups', None)
            if groups is None:
                raise ValueError(
                    'Resolution-specific batches require dataset.resolution_groups.'
                )
            if set(groups) != set(self.batch_size_per_gpu_by_resolution):
                raise ValueError(
                    'batch_size_per_gpu_by_resolution must define every resolution: '
                    f'{sorted(groups)}; got {sorted(self.batch_size_per_gpu_by_resolution)}'
                )
            if (
                self.resolution_sampling_weights is not None
                and set(groups) != set(self.resolution_sampling_weights)
            ):
                raise ValueError(
                    'resolution_sampling_weights must define every resolution: '
                    f'{sorted(groups)}; got {sorted(self.resolution_sampling_weights)}'
                )
            self.data_sampler = GroupedBalancedResumableBatchSampler(
                self.dataset,
                groups=groups,
                batch_sizes=self.batch_size_per_gpu_by_resolution,
                group_weights=self.resolution_sampling_weights,
                shuffle=True,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_sampler=self.data_sampler,
                num_workers=self.num_workers if self.num_workers is not None else int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
                pin_memory=True,
                persistent_workers=(self.num_workers or 0) > 0,
                collate_fn=self.dataset.collate_fn,
            )
            self.data_iterator = self._cycle_resolution_dataloader()
            return

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

    def _cycle_resolution_dataloader(self):
        while True:
            for data in self.dataloader:
                self.data_sampler.idx += 1
                yield data
            self.data_sampler.epoch += 1
            self.data_sampler.idx = 0

    def __str__(self):
        lines = [super().__str__(), f'  - Sample-balanced loss: {self.sample_balanced_loss}']
        if self.batch_size_per_gpu_by_resolution is not None:
            lines.append(
                '  - Batch size per GPU by resolution: '
                f'{self.batch_size_per_gpu_by_resolution}'
            )
        if self.resolution_sampling_weights is not None:
            lines.append(
                f'  - Resolution sampling weights: {self.resolution_sampling_weights}'
            )
        return '\n'.join(lines)

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
        layout=None,
    ) -> torch.Tensor:
        if self.loss_type == 'l1':
            err = torch.abs(target - pred)
        elif self.loss_type == 'l2':
            err = (target - pred) ** 2
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

        if weight is not None:
            err = err * weight.reshape(-1, 1).to(device=err.device, dtype=err.dtype)
        if self.sample_balanced_loss:
            if layout is None:
                raise ValueError('Sample-balanced reconstruction loss requires a sparse layout.')
            return torch.stack([err[sample_slice].mean() for sample_slice in layout]).mean()
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

    @staticmethod
    def _voxel_coordinate_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
        """Encode xyz voxel coordinates as unique int64 values for exact matching."""
        coords = coords.to(dtype=torch.long)
        return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]

    def _vertex_training_sample_mask(
        self,
        final_resolutions: torch.Tensor,
    ) -> torch.Tensor:
        """Select samples whose final resolution may supervise vertex heads."""
        final_resolutions = final_resolutions.reshape(-1).long()
        if self.vertex_training_resolutions is None:
            return torch.ones_like(final_resolutions, dtype=torch.bool)
        allowed = torch.tensor(
            self.vertex_training_resolutions,
            device=final_resolutions.device,
            dtype=final_resolutions.dtype,
        )
        return torch.isin(final_resolutions, allowed)

    def _vertex_resolution_is_enabled(self, resolution: int) -> bool:
        return (
            self.vertex_training_resolutions is None
            or int(resolution) in self.vertex_training_resolutions
        )

    def _edge_resolution_is_enabled(self, resolution: int) -> bool:
        return (
            self.edge_training_resolutions is None
            or int(resolution) in self.edge_training_resolutions
        )

    @staticmethod
    def _edge_pair_keys(pairs: torch.Tensor, num_vertices: int) -> torch.Tensor:
        pairs = pairs.long()
        return pairs[:, 0] * num_vertices + pairs[:, 1]

    def _sample_random_non_edges(
        self,
        num_vertices: int,
        count: int,
        blocked_keys: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample unique unordered non-edges without constructing all N^2 pairs."""
        if count <= 0 or num_vertices < 2:
            return torch.empty((0, 2), device=device, dtype=torch.long)
        blocked_keys = torch.unique(blocked_keys.long())
        selected_keys = torch.empty((0,), device=device, dtype=torch.long)
        for _ in range(128):
            remaining = count - len(selected_keys)
            if remaining <= 0:
                break
            draw_count = max(remaining * 16, 256)
            endpoints = torch.randint(
                num_vertices,
                (draw_count, 2),
                device=device,
                dtype=torch.long,
            )
            endpoints, _ = torch.sort(endpoints, dim=1)
            endpoints = endpoints[endpoints[:, 0] != endpoints[:, 1]]
            if len(endpoints) == 0:
                continue
            candidate_keys = torch.unique(
                self._edge_pair_keys(endpoints, num_vertices)
            )
            all_blocked = torch.cat([blocked_keys, selected_keys])
            candidate_keys = candidate_keys[
                ~torch.isin(candidate_keys, all_blocked)
            ]
            if len(candidate_keys) == 0:
                continue
            selected_keys = torch.cat(
                [selected_keys, candidate_keys[:remaining]]
            )
        selected_keys = selected_keys[:count]
        return torch.stack(
            [selected_keys // num_vertices, selected_keys % num_vertices], dim=1
        )

    def _sample_edge_training_pairs(
        self,
        qem_vertex_coords: list,
        qem_edges: list,
        final_resolutions: torch.Tensor,
    ):
        """Keep all QEM edges and sample negatives separately for every vertex.

        Each QEM vertex is an anchor. For that anchor, nearby negatives are the
        closest non-neighbors among its K nearest vertices, and random negatives
        are sampled from its remaining non-neighbors. Negative pairs are kept as
        anchored pairs, so the same undirected non-edge may occur once for each
        endpoint; this is intentional because every vertex receives supervision.
        """
        final_resolutions = final_resolutions.reshape(-1).long()
        if len(qem_vertex_coords) != len(final_resolutions):
            raise ValueError('qem_vertex_coords must contain one tensor per sample.')
        if len(qem_edges) != len(final_resolutions):
            raise ValueError('qem_edges must contain one tensor per sample.')

        all_pairs = []
        all_labels = []
        positive_counts = []
        anchor_counts = []
        nearby_counts = []
        random_counts = []
        for batch_index, resolution_value in enumerate(final_resolutions):
            resolution = int(resolution_value.item())
            coords = qem_vertex_coords[batch_index].to(
                device=final_resolutions.device, dtype=torch.long
            )
            positive = qem_edges[batch_index].to(
                device=final_resolutions.device, dtype=torch.long
            )
            if not self._edge_resolution_is_enabled(resolution):
                all_pairs.append(torch.empty(
                    (0, 2), device=final_resolutions.device, dtype=torch.long
                ))
                all_labels.append(torch.empty(
                    (0,), device=final_resolutions.device, dtype=torch.float32
                ))
                positive_counts.append(0)
                anchor_counts.append(0)
                nearby_counts.append(0)
                random_counts.append(0)
                continue

            if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 2:
                raise ValueError(
                    f'R{resolution} edge sample {batch_index} needs at least two '
                    'QEM vertex coordinates.'
                )
            if positive.ndim != 2 or positive.shape[1] != 2 or len(positive) == 0:
                raise ValueError(
                    f'R{resolution} edge sample {batch_index} has no valid GT edges.'
                )
            positive, _ = torch.sort(positive, dim=1)
            positive = torch.unique(positive, dim=0)
            if (
                (positive[:, 0] == positive[:, 1]).any()
                or positive.min() < 0
                or positive.max() >= len(coords)
            ):
                raise ValueError(
                    f'R{resolution} edge sample {batch_index} has invalid GT endpoints.'
                )

            num_vertices = len(coords)
            positive_neighbors = [set() for _ in range(num_vertices)]
            for endpoint_u, endpoint_v in positive.detach().cpu().tolist():
                positive_neighbors[endpoint_u].add(endpoint_v)
                positive_neighbors[endpoint_v].add(endpoint_u)

            max_positive_degree = max(map(len, positive_neighbors))
            # Query enough geometric neighbors that removing every possible GT
            # neighbor still leaves a pool of K non-edges whenever the mesh has
            # that many non-neighbors.
            query_k = min(
                num_vertices,
                1 + self.edge_knn_k + max_positive_degree,
            )
            try:
                from scipy.spatial import cKDTree
            except ImportError as exc:
                raise ImportError(
                    'Edge KNN negative sampling requires scipy.'
                ) from exc
            coordinate_tree = cKDTree(coords.detach().cpu().numpy())
            _, neighbors_np = coordinate_tree.query(
                coords.detach().cpu().numpy(),
                k=query_k,
                workers=1,
            )
            nearby_pairs = []
            random_pairs = []
            random_per_vertex = (
                self.edge_negative_pairs - self.edge_nearby_negative_pairs
            )
            all_vertex_indices = set(range(num_vertices))
            for anchor in range(num_vertices):
                true_neighbors = positive_neighbors[anchor]
                nearby_candidates = [
                    int(candidate)
                    for candidate in neighbors_np[anchor, 1:]
                    if int(candidate) != anchor
                    and int(candidate) not in true_neighbors
                ][:self.edge_knn_k]
                nearby_for_anchor = nearby_candidates[
                    :self.edge_nearby_negative_pairs
                ]
                nearby_pairs.extend(
                    (anchor, candidate) for candidate in nearby_for_anchor
                )

                random_candidates = list(
                    all_vertex_indices
                    - {anchor}
                    - true_neighbors
                    - set(nearby_for_anchor)
                )
                random_take = min(random_per_vertex, len(random_candidates))
                if random_take:
                    random_order = torch.randperm(
                        len(random_candidates)
                    )[:random_take].tolist()
                    random_pairs.extend(
                        (anchor, random_candidates[index])
                        for index in random_order
                    )

            nearby = torch.tensor(
                nearby_pairs, device=coords.device, dtype=torch.long
            ).reshape(-1, 2)
            random_negative = torch.tensor(
                random_pairs, device=coords.device, dtype=torch.long
            ).reshape(-1, 2)
            negative = torch.cat([nearby, random_negative], dim=0)

            pairs = torch.cat([positive, negative], dim=0)
            labels = torch.cat([
                torch.ones(len(positive), device=coords.device),
                torch.zeros(len(negative), device=coords.device),
            ]).float()
            all_pairs.append(pairs)
            all_labels.append(labels)
            positive_counts.append(len(positive))
            anchor_counts.append(num_vertices)
            nearby_counts.append(len(nearby))
            random_counts.append(len(random_negative))

        return (
            all_pairs,
            all_labels,
            positive_counts,
            anchor_counts,
            nearby_counts,
            random_counts,
        )

    def _build_vertex_child_stage_target(
        self,
        vertex_logits: sp.SparseTensor,
        final_vertex_occupancy: sp.SparseTensor,
        final_resolutions: torch.Tensor,
        stage_index: int,
        num_stages: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build eight child-vertex labels from one same-resolution QEM target.

        Stage zero supervises every active latent parent.  Later stages are
        teacher-forced and supervise only parents in the downsampled GT vertex
        hierarchy.  The hierarchy is always derived from the sample's final
        QEM occupancy; independently voxelized lower-resolution QEM files are
        never consulted.
        """
        if vertex_logits.feats.ndim != 2 or vertex_logits.feats.shape[1] != 8:
            raise ValueError(
                'Each vertex-subdivision head must emit [num_parents, 8] logits, '
                f'got {tuple(vertex_logits.feats.shape)}.'
            )
        final_resolutions = final_resolutions.reshape(-1)
        if len(final_resolutions) == 0:
            raise ValueError('resolution must contain at least one sample.')
        occupancy = final_vertex_occupancy.feats.reshape(-1)
        if not torch.logical_or(occupancy == 0, occupancy == 1).all():
            raise ValueError('vertex_occupancy must be binary.')

        target = torch.zeros_like(vertex_logits.feats, dtype=torch.float32)
        supervised_parent = torch.zeros(
            vertex_logits.feats.shape[0],
            device=vertex_logits.feats.device,
            dtype=torch.bool,
        )
        child_resolutions = torch.full(
            (vertex_logits.feats.shape[0],),
            -1,
            device=vertex_logits.feats.device,
            dtype=torch.long,
        )
        head_batch_ids = vertex_logits.coords[:, 0].to(
            device=vertex_logits.feats.device, dtype=torch.long
        )
        occupancy_batch_ids = final_vertex_occupancy.coords[:, 0].to(
            device=vertex_logits.feats.device, dtype=torch.long
        )
        child_offsets = torch.tensor(
            [
                [child_index % 2, (child_index // 2) % 2, child_index // 4]
                for child_index in range(8)
            ],
            device=vertex_logits.feats.device,
            dtype=torch.long,
        )

        for batch_index, final_resolution_value in enumerate(final_resolutions):
            final_resolution = int(final_resolution_value.item())
            if final_resolution <= 0 or final_resolution % (2 ** num_stages) != 0:
                raise ValueError(
                    f'Final resolution {final_resolution} must be divisible by '
                    f'2 ** num_stages ({2 ** num_stages}).'
                )
            parent_resolution = final_resolution // (2 ** (num_stages - stage_index))
            child_resolution = parent_resolution * 2

            head_mask = head_batch_ids.eq(batch_index)
            if not head_mask.any():
                raise ValueError(
                    f'Vertex stage {stage_index} emitted no parent features for '
                    f'batch sample {batch_index}.'
                )
            head_xyz = vertex_logits.coords[head_mask, 1:4].to(
                device=vertex_logits.feats.device, dtype=torch.long
            )
            if (head_xyz < 0).any() or (head_xyz >= parent_resolution).any():
                raise ValueError(
                    f'Vertex stage {stage_index} parent coordinates are outside '
                    f'[0, {parent_resolution}) for batch sample {batch_index}.'
                )

            final_positive_mask = occupancy_batch_ids.eq(batch_index) & occupancy.bool()
            final_positive_xyz = final_vertex_occupancy.coords[
                final_positive_mask, 1:4
            ].to(device=vertex_logits.feats.device, dtype=torch.long)
            if len(final_positive_xyz) == 0:
                raise ValueError(
                    f'Batch sample {batch_index} has no positive final-resolution '
                    'QEM vertex voxels.'
                )
            if (
                (final_positive_xyz < 0).any()
                or (final_positive_xyz >= final_resolution).any()
            ):
                raise ValueError(
                    f'Batch sample {batch_index} has QEM coordinates outside '
                    f'[0, {final_resolution}).'
                )

            downsample_factor = final_resolution // child_resolution
            positive_child_xyz = torch.unique(
                final_positive_xyz // downsample_factor, dim=0
            )
            positive_parent_xyz = torch.unique(positive_child_xyz // 2, dim=0)
            head_keys = self._voxel_coordinate_keys(head_xyz, parent_resolution)
            positive_parent_keys = self._voxel_coordinate_keys(
                positive_parent_xyz, parent_resolution
            )
            parent_in_head = torch.isin(positive_parent_keys, head_keys)
            if not parent_in_head.all():
                raise ValueError(
                    f'Batch sample {batch_index}, vertex stage {stage_index}: '
                    f'{int((~parent_in_head).sum().item())} GT vertex parents are '
                    'absent from the decoder parent support.'
                )

            candidate_child_xyz = head_xyz[:, None, :] * 2 + child_offsets[None]
            candidate_child_keys = self._voxel_coordinate_keys(
                candidate_child_xyz.reshape(-1, 3), child_resolution
            ).reshape(-1, 8)
            positive_child_keys = self._voxel_coordinate_keys(
                positive_child_xyz, child_resolution
            )
            sample_target = torch.isin(
                candidate_child_keys, positive_child_keys
            ).float()
            represented_children = torch.isin(
                positive_child_keys, candidate_child_keys.reshape(-1)
            )
            if not represented_children.all():
                raise ValueError(
                    f'Batch sample {batch_index}, vertex stage {stage_index}: '
                    f'{int((~represented_children).sum().item())} GT vertex children '
                    'cannot be represented by the decoder parent support.'
                )

            sample_supervised = (
                torch.ones_like(head_keys, dtype=torch.bool)
                if stage_index == 0
                else torch.isin(head_keys, positive_parent_keys)
            )
            if stage_index > 0 and not sample_target[sample_supervised].any(dim=1).all():
                raise AssertionError(
                    f'Batch sample {batch_index}, vertex stage {stage_index}: a '
                    'teacher-forced vertex parent has no positive child.'
                )

            target[head_mask] = sample_target
            supervised_parent[head_mask] = sample_supervised
            child_resolutions[head_mask] = child_resolution

        return target, supervised_parent, child_resolutions

    def _build_vertex_token_stage_target(
        self,
        vertex_logits: sp.SparseTensor,
        final_vertex_occupancy: sp.SparseTensor,
        final_resolutions: torch.Tensor,
        stage_index: int,
        num_stages: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build scalar labels for explicit recursively generated child tokens."""
        if vertex_logits.feats.ndim != 2 or vertex_logits.feats.shape[1] != 1:
            raise ValueError(
                'Each hierarchical vertex stage must emit [num_children, 1] '
                f'logits, got {tuple(vertex_logits.feats.shape)}.'
            )
        final_resolutions = final_resolutions.reshape(-1).long()
        occupancy = final_vertex_occupancy.feats.reshape(-1)
        if not torch.logical_or(occupancy == 0, occupancy == 1).all():
            raise ValueError('vertex_occupancy must be binary.')

        target = torch.zeros_like(vertex_logits.feats, dtype=torch.float32)
        supervised = torch.ones(
            len(vertex_logits.feats),
            device=vertex_logits.device,
            dtype=torch.bool,
        )
        child_resolutions = torch.full(
            (len(vertex_logits.feats),),
            -1,
            device=vertex_logits.device,
            dtype=torch.long,
        )
        logit_batch_ids = vertex_logits.coords[:, 0].long()
        occupancy_batch_ids = final_vertex_occupancy.coords[:, 0].long()

        for batch_index, final_resolution_value in enumerate(final_resolutions):
            final_resolution = int(final_resolution_value.item())
            if final_resolution <= 0 or final_resolution % (2 ** num_stages) != 0:
                raise ValueError(
                    f'Final resolution {final_resolution} must be divisible by '
                    f'2 ** num_stages ({2 ** num_stages}).'
                )
            child_resolution = final_resolution // (
                2 ** (num_stages - stage_index - 1)
            )
            rows = torch.nonzero(
                logit_batch_ids.eq(batch_index), as_tuple=False
            ).flatten()
            if len(rows) == 0:
                raise ValueError(
                    f'Hierarchical vertex stage {stage_index} emitted no child '
                    f'tokens for batch sample {batch_index}.'
                )
            candidate_xyz = vertex_logits.coords[rows, 1:4].long()
            if (candidate_xyz < 0).any() or (candidate_xyz >= child_resolution).any():
                raise ValueError(
                    f'Vertex stage {stage_index} coordinates are outside '
                    f'[0, {child_resolution}) for batch sample {batch_index}.'
                )
            final_positive_xyz = final_vertex_occupancy.coords[
                occupancy_batch_ids.eq(batch_index) & occupancy.bool(), 1:4
            ].long()
            if len(final_positive_xyz) == 0:
                raise ValueError(f'Batch sample {batch_index} has no QEM vertices.')
            positive_child_xyz = torch.unique(
                final_positive_xyz // (final_resolution // child_resolution),
                dim=0,
            )
            candidate_keys = self._voxel_coordinate_keys(
                candidate_xyz, child_resolution
            )
            positive_keys = self._voxel_coordinate_keys(
                positive_child_xyz, child_resolution
            )
            represented = torch.isin(positive_keys, candidate_keys)
            if not represented.all():
                raise ValueError(
                    f'Batch sample {batch_index}, hierarchical vertex stage '
                    f'{stage_index}: {int((~represented).sum().item())} GT child '
                    'tokens are absent from the recursively supplied support.'
                )
            target[rows, 0] = torch.isin(candidate_keys, positive_keys).float()
            child_resolutions[rows] = child_resolution

        if (child_resolutions < 0).any():
            raise AssertionError('Some hierarchical vertex tokens were not assigned a resolution.')
        return target, supervised, child_resolutions

    @staticmethod
    def _asymmetric_binary_loss_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
        gamma_negative: float,
        gamma_positive: float,
    ) -> torch.Tensor:
        """Asymmetric binary loss without probability shifting or clipping.

        Positive: -(1 - p)^gamma_positive * log(p)
        Negative: -p^gamma_negative * log(1 - p)

        ``logsigmoid`` keeps both logarithms numerically stable for large logits.
        The focusing weights remain in the autograd graph.
        """
        probability = torch.sigmoid(logits)
        positive_loss = -(
            target *
            (1.0 - probability).pow(gamma_positive) *
            F.logsigmoid(logits)
        )
        negative_loss = -(
            (1.0 - target) *
            probability.pow(gamma_negative) *
            F.logsigmoid(-logits)
        )
        return (positive_loss + negative_loss).mean()

    def _vertex_child_loss_and_metrics(
        self,
        vertex_logits: sp.SparseTensor,
        target: torch.Tensor,
        supervised_parent: torch.Tensor,
        child_resolutions: torch.Tensor,
        stage_index: int,
        use_asymmetric_loss: bool,
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        all_logits = vertex_logits.feats.float()
        if all_logits.shape != target.shape:
            raise ValueError(
                'Vertex-child logit shape must match target shape, got '
                f'{tuple(all_logits.shape)} vs {tuple(target.shape)}.'
            )
        if supervised_parent.shape != (all_logits.shape[0],):
            raise ValueError('supervised_parent must contain one mask value per parent.')

        logits = all_logits[supervised_parent]
        target = target[supervised_parent]
        child_resolutions = child_resolutions[supervised_parent]
        supervised_coords = vertex_logits.coords[supervised_parent]
        target_bool = target > 0.5
        local_positive_count = int(target_bool.sum().item())
        local_negative_count = int((~target_bool).sum().item())
        class_counts = torch.tensor(
            [local_positive_count, local_negative_count],
            device=logits.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(class_counts, op=dist.ReduceOp.SUM)
        positive_count = int(class_counts[0].item())
        negative_count = int(class_counts[1].item())
        if positive_count == 0:
            if supervised_parent.any():
                raise ValueError(f'{prefix} has no positive vertex children.')
            # No selected-resolution sample exists on any rank for this step.
            # Keep a differentiable zero dependency on the complete stage logit
            # tensor so DDP observes every vertex parameter on non-R512 steps.
            zero_loss = all_logits.sum() * 0.0
            return zero_loss, {
                f'{prefix}/precision': 0.0,
                f'{prefix}/recall': 0.0,
                f'{prefix}/iou': 0.0,
                f'{prefix}/bit_accuracy': 0.0,
                f'{prefix}/perfect_parent': 0.0,
                f'{prefix}/gt_positive_frac': 0.0,
                f'{prefix}/pred_positive_frac': 0.0,
                f'{prefix}/gt_children_per_parent': 0.0,
                f'{prefix}/pred_children_per_parent': 0.0,
                f'{prefix}/positive_weight': 1.0,
                f'{prefix}/loss_is_asymmetric': float(use_asymmetric_loss),
                f'{prefix}/supervised_child_tokens': 0.0,
                f'{prefix}/supervised_parents': 0.0,
                f'{prefix}/positive_children_local': 0.0,
                f'{prefix}/negative_children_local': 0.0,
            }

        if len(logits) == 0:
            # Another rank owns the selected-resolution sample. Its nonzero
            # vertex gradient will be synchronized by DDP; this rank contributes
            # a graph-connected zero.
            zero_loss = all_logits.sum() * 0.0
            return zero_loss, {
                f'{prefix}/precision': 0.0,
                f'{prefix}/recall': 0.0,
                f'{prefix}/iou': 0.0,
                f'{prefix}/bit_accuracy': 0.0,
                f'{prefix}/perfect_parent': 0.0,
                f'{prefix}/gt_positive_frac': 0.0,
                f'{prefix}/pred_positive_frac': 0.0,
                f'{prefix}/gt_children_per_parent': 0.0,
                f'{prefix}/pred_children_per_parent': 0.0,
                f'{prefix}/positive_weight': float(
                    max(negative_count / positive_count, 1.0)
                ),
                f'{prefix}/loss_is_asymmetric': float(use_asymmetric_loss),
                f'{prefix}/supervised_child_tokens': 0.0,
                f'{prefix}/supervised_parents': 0.0,
                f'{prefix}/positive_children_local': 0.0,
                f'{prefix}/negative_children_local': 0.0,
            }

        if use_asymmetric_loss:
            positive_weight_value = 1.0
            vertex_loss = self._asymmetric_binary_loss_with_logits(
                logits,
                target,
                gamma_negative=self.vertex_asymmetric_gamma_negative,
                gamma_positive=self.vertex_asymmetric_gamma_positive,
            )
        else:
            if stage_index in self.vertex_child_pos_weight_by_stage:
                positive_weight_value = self.vertex_child_pos_weight_by_stage[stage_index]
            else:
                # Exact class balancing for the supervised parents in this batch.
                positive_weight_value = max(negative_count / positive_count, 1.0)
            positive_weight = torch.tensor(
                positive_weight_value, device=logits.device, dtype=logits.dtype
            )
            vertex_loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=positive_weight,
            )

        probability = torch.sigmoid(logits)
        pred = probability >= self.vertex_threshold
        eps = torch.tensor(1e-8, device=logits.device)

        def metrics_for(parent_mask: torch.Tensor, metric_prefix: str) -> Dict[str, float]:
            masked_pred = pred[parent_mask]
            masked_gt = target_bool[parent_mask]
            tp = (masked_pred & masked_gt).float().sum()
            fp = (masked_pred & ~masked_gt).float().sum()
            fn = (~masked_pred & masked_gt).float().sum()
            correct = masked_pred == masked_gt
            if logits.shape[1] == 1:
                # Explicit child-token logits are grouped by floor(child / 2)
                # so perfect_parent retains its original all-children-correct
                # meaning even though unsupported children are not materialized.
                masked_coords = supervised_coords[parent_mask]
                parent_coords = torch.cat(
                    [masked_coords[:, :1].long(), masked_coords[:, 1:4].long() // 2],
                    dim=1,
                )
                _, inverse = torch.unique(parent_coords, dim=0, return_inverse=True)
                num_parents = int(inverse.max().item()) + 1
                parent_correct = torch.ones(
                    num_parents, device=correct.device, dtype=torch.float32
                )
                parent_correct.scatter_reduce_(
                    0,
                    inverse,
                    correct.reshape(-1).float(),
                    reduce='amin',
                    include_self=True,
                )
                gt_per_parent = torch.zeros(
                    num_parents, device=correct.device, dtype=torch.float32
                )
                pred_per_parent = torch.zeros_like(gt_per_parent)
                gt_per_parent.scatter_add_(0, inverse, masked_gt.reshape(-1).float())
                pred_per_parent.scatter_add_(0, inverse, masked_pred.reshape(-1).float())
                perfect_parent = parent_correct.mean()
                gt_children_per_parent = gt_per_parent.mean()
                pred_children_per_parent = pred_per_parent.mean()
            else:
                perfect_parent = correct.all(dim=1).float().mean()
                gt_children_per_parent = masked_gt.float().sum(dim=1).mean()
                pred_children_per_parent = masked_pred.float().sum(dim=1).mean()
            return {
                f'{metric_prefix}/precision': (tp / (tp + fp + eps)).item(),
                f'{metric_prefix}/recall': (tp / (tp + fn + eps)).item(),
                f'{metric_prefix}/iou': (tp / (tp + fp + fn + eps)).item(),
                f'{metric_prefix}/bit_accuracy': correct.float().mean().item(),
                f'{metric_prefix}/perfect_parent': perfect_parent.item(),
                f'{metric_prefix}/gt_positive_frac': masked_gt.float().mean().item(),
                f'{metric_prefix}/pred_positive_frac': masked_pred.float().mean().item(),
                f'{metric_prefix}/gt_children_per_parent': gt_children_per_parent.item(),
                f'{metric_prefix}/pred_children_per_parent': pred_children_per_parent.item(),
            }

        status = metrics_for(
            torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool),
            prefix,
        )
        for child_resolution in torch.unique(child_resolutions).tolist():
            child_resolution = int(child_resolution)
            status.update(
                metrics_for(
                    child_resolutions.eq(child_resolution),
                    f'{prefix}/r{child_resolution}',
                )
            )
        status[f'{prefix}/positive_weight'] = float(positive_weight_value)
        status[f'{prefix}/loss_is_asymmetric'] = float(use_asymmetric_loss)
        if use_asymmetric_loss:
            status[f'{prefix}/asymmetric_gamma_negative'] = (
                self.vertex_asymmetric_gamma_negative
            )
            status[f'{prefix}/asymmetric_gamma_positive'] = (
                self.vertex_asymmetric_gamma_positive
            )
        if logits.shape[1] == 1:
            supervised_parent_coords = torch.cat(
                [supervised_coords[:, :1].long(), supervised_coords[:, 1:4].long() // 2],
                dim=1,
            )
            supervised_parent_count = len(
                torch.unique(supervised_parent_coords, dim=0)
            )
            status[f'{prefix}/supervised_child_tokens'] = float(logits.shape[0])
        else:
            supervised_parent_count = logits.shape[0]
            status[f'{prefix}/supervised_child_tokens'] = float(logits.numel())
        status[f'{prefix}/supervised_parents'] = float(supervised_parent_count)
        status[f'{prefix}/positive_children_local'] = float(local_positive_count)
        status[f'{prefix}/negative_children_local'] = float(local_negative_count)
        return vertex_loss, status

    def training_losses(
        self,
        x: sp.SparseTensor,
        target: sp.SparseTensor,
        vertex_occupancy: sp.SparseTensor = None,
        resolution: torch.Tensor = None,
        qem_vertex_coords: list = None,
        qem_edges: list = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        Compute losses with asymmetric input and target tensors.

        Args:
            x: Full encoder input features.
            target: Reconstruction channels selected by the dataset target layout.
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
        decoder_predicts_vertex = bool(
            self._module_attr(self.training_models['decoder'], 'pred_vertex_subdiv', False)
        )
        decoder_vertex_child_tokens = bool(
            self._module_attr(
                self.training_models['decoder'],
                'vertex_logits_are_child_tokens',
                False,
            )
        )
        decoder_predicts_edge = bool(
            self._module_attr(self.training_models['decoder'], 'pred_edge', False)
        )
        vertex_logits = None
        edge_logits = None
        edge_pairs = None
        edge_labels = None
        edge_positive_counts = None
        edge_anchor_counts = None
        edge_nearby_counts = None
        edge_random_counts = None
        if decoder_predicts_edge:
            if resolution is None or qem_vertex_coords is None or qem_edges is None:
                raise ValueError(
                    'Edge-enabled decoding requires resolution, qem_vertex_coords, '
                    'and qem_edges in the batch.'
                )
            (
                edge_pairs,
                edge_labels,
                edge_positive_counts,
                edge_anchor_counts,
                edge_nearby_counts,
                edge_random_counts,
            ) = self._sample_edge_training_pairs(
                qem_vertex_coords,
                qem_edges,
                resolution,
            )
        if decoder_predicts_subdiv and decoder_predicts_vertex:
            if decoder_predicts_edge:
                raise ValueError('Edge prediction does not support pred_subdiv=True.')
            y, vertex_logits, subs_gt, subs = self.training_models['decoder'](
                z, return_vertex=True
            )
        elif decoder_predicts_subdiv:
            y, subs_gt, subs = self.training_models['decoder'](z)
        elif decoder_predicts_vertex:
            if decoder_vertex_child_tokens:
                if decoder_predicts_edge:
                    y, vertex_logits, edge_logits = self.training_models['decoder'](
                        z,
                        return_vertex=True,
                        return_edge=True,
                        resolutions=resolution,
                        vertex_guide=vertex_occupancy,
                        vertex_threshold=self.vertex_threshold,
                        edge_vertex_coords=qem_vertex_coords,
                        edge_pairs=edge_pairs,
                    )
                else:
                    y, vertex_logits = self.training_models['decoder'](
                        z,
                        return_vertex=True,
                        resolutions=resolution,
                        vertex_guide=vertex_occupancy,
                        vertex_threshold=self.vertex_threshold,
                    )
            else:
                y, vertex_logits = self.training_models['decoder'](
                    z, return_vertex=True
                )
            subs_gt, subs = [], []
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
            terms['l1'] = self._reconstruction_loss(
                target.feats, y.feats, loss_weight, target.layout
            )
            if self.debug_nans:
                status.update(self._debug_tensor_stats('loss/l1', terms['l1'].reshape(1)))
            terms['loss'] = terms['loss'] + terms['l1']
        elif self.loss_type == 'l2':
            terms['l2'] = self._reconstruction_loss(
                target.feats, y.feats, loss_weight, target.layout
            )
            if self.debug_nans:
                status.update(self._debug_tensor_stats('loss/l2', terms['l2'].reshape(1)))
            terms['loss'] = terms['loss'] + terms['l2']
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

        if decoder_predicts_vertex:
            if vertex_occupancy is None or resolution is None:
                raise ValueError(
                    'Decoder has pred_vertex_subdiv=True, but the batch has no '
                    'vertex_occupancy or resolution.'
                )
            if not isinstance(vertex_logits, list):
                raise TypeError(
                    'Vertex-subdivision decoder must return one logit tensor per stage.'
                )
            if not torch.equal(vertex_occupancy.coords, target.coords):
                raise ValueError(
                    'Final vertex_occupancy must have exactly the same sparse '
                    'coordinates and ordering as the reconstruction target.'
                )

            stage_losses = []
            final_resolutions = resolution.reshape(-1)
            vertex_sample_mask = self._vertex_training_sample_mask(
                final_resolutions
            )
            local_vertex_sample_count = int(vertex_sample_mask.sum().item())
            vertex_sample_counts = torch.tensor(
                [local_vertex_sample_count],
                device=final_resolutions.device,
                dtype=torch.float64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(vertex_sample_counts, op=dist.ReduceOp.SUM)
            global_vertex_sample_count = int(vertex_sample_counts[0].item())
            vertex_loss_ddp_scale = (
                self.world_size
                * local_vertex_sample_count
                / global_vertex_sample_count
                if global_vertex_sample_count > 0
                else 0.0
            )
            status['vertex/supervised_samples_local'] = float(
                local_vertex_sample_count
            )
            status['vertex/supervised_samples_global'] = float(
                global_vertex_sample_count
            )
            status['vertex/skipped_samples_local'] = float(
                len(final_resolutions) - local_vertex_sample_count
            )
            status['vertex/loss_ddp_scale'] = float(vertex_loss_ddp_scale)
            for stage_index, stage_logits in enumerate(vertex_logits):
                is_finest_stage = stage_index == len(vertex_logits) - 1
                use_asymmetric_loss = (
                    is_finest_stage and self.vertex_finest_loss == 'asymmetric'
                )
                target_builder = (
                    self._build_vertex_token_stage_target
                    if decoder_vertex_child_tokens
                    else self._build_vertex_child_stage_target
                )
                stage_target, supervised_parent, child_resolutions = (
                    target_builder(
                        stage_logits,
                        vertex_occupancy,
                        final_resolutions,
                        stage_index,
                        len(vertex_logits),
                    )
                )
                stage_batch_ids = stage_logits.coords[:, 0].long()
                supervised_parent = supervised_parent & vertex_sample_mask[
                    stage_batch_ids
                ]
                stage_loss, vertex_status = self._vertex_child_loss_and_metrics(
                    stage_logits,
                    stage_target,
                    supervised_parent,
                    child_resolutions,
                    stage_index,
                    use_asymmetric_loss,
                    f'vertex/stage{stage_index}',
                )
                stage_loss = stage_loss * vertex_loss_ddp_scale
                stage_loss_name = (
                    f'asymmetric_vertex_stage{stage_index}'
                    if use_asymmetric_loss else f'bce_vertex_stage{stage_index}'
                )
                terms[stage_loss_name] = stage_loss
                stage_losses.append(stage_loss)
                status.update(vertex_status)
                if self.debug_nans:
                    status.update(self._debug_sparse_stats(
                        f'decoder/vertex_logits_stage{stage_index}', stage_logits
                    ))
                    status.update(self._debug_tensor_stats(
                        f'target/vertex_stage{stage_index}',
                        stage_target[supervised_parent],
                    ))

            terms['vertex'] = torch.stack(stage_losses).mean()
            terms['loss'] = terms['loss'] + self.lambda_vertex * terms['vertex']
            if self.debug_nans:
                status.update(
                    self._debug_tensor_stats('loss/vertex', terms['vertex'].reshape(1))
                )

        if decoder_predicts_edge:
            if not isinstance(edge_logits, list) or not isinstance(edge_labels, list):
                raise TypeError('Edge-enabled decoder must return one logit tensor per sample.')
            if len(edge_logits) != len(edge_labels):
                raise ValueError('Edge logits and labels have different batch lengths.')

            local_supervised_edge_samples = sum(
                int(positive_count > 0)
                for positive_count in edge_positive_counts
            )
            edge_sample_counts = torch.tensor(
                [local_supervised_edge_samples],
                device=x.device,
                dtype=torch.float64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(edge_sample_counts, op=dist.ReduceOp.SUM)
            global_supervised_edge_samples = int(edge_sample_counts[0].item())
            edge_loss_ddp_scale = (
                self.world_size
                * local_supervised_edge_samples
                / global_supervised_edge_samples
                if global_supervised_edge_samples > 0
                else 0.0
            )

            per_mesh_edge_losses = []
            local_positive_weight_sum = 0.0
            local_positive_only_edge_samples = 0
            local_tp = local_fp = local_fn = local_tn = 0
            local_positive_pairs = local_negative_pairs = 0
            zero_edge_dependency = sum(
                (logits.sum() * 0.0 for logits in edge_logits),
                torch.zeros((), device=x.device),
            )
            for batch_index, (logits, labels) in enumerate(
                zip(edge_logits, edge_labels)
            ):
                labels = labels.to(device=logits.device, dtype=torch.float32)
                logits = logits.reshape(-1)
                if len(logits) != len(labels):
                    raise ValueError(
                        f'Edge sample {batch_index} has {len(logits)} logits but '
                        f'{len(labels)} labels.'
                    )
                if edge_positive_counts[batch_index] == 0:
                    if len(logits) != 0:
                        raise ValueError(
                            f'Unsupervised edge sample {batch_index} emitted logits.'
                        )
                    continue
                positive_count = int(labels.sum().item())
                negative_count = len(labels) - positive_count
                if positive_count <= 0:
                    raise ValueError(
                        f'Edge sample {batch_index} requires a positive class, got '
                        f'{positive_count} positive and {negative_count} negative pairs.'
                    )
                if negative_count > 0:
                    positive_weight = negative_count / positive_count
                    pair_loss = F.binary_cross_entropy_with_logits(
                        logits.float(),
                        labels,
                        pos_weight=torch.tensor(
                            positive_weight,
                            device=logits.device,
                            dtype=torch.float32,
                        ),
                        reduction='sum',
                    )
                    # The effective positive weight sums to N and the negative
                    # weight also sums to N, so 2N gives a class-balanced mean.
                    per_mesh_edge_losses.append(
                        pair_loss / (2.0 * negative_count)
                    )
                else:
                    # A complete GT graph has no valid negative pair. Preserve
                    # all positive supervision with ordinary positive-only BCE;
                    # Nneg/Npos would be zero and would erase the positive loss.
                    positive_weight = 1.0
                    local_positive_only_edge_samples += 1
                    per_mesh_edge_losses.append(
                        F.binary_cross_entropy_with_logits(
                            logits.float(), labels, reduction='mean'
                        )
                    )
                local_positive_weight_sum += positive_weight

                prediction = torch.sigmoid(logits.float()) >= self.edge_threshold
                truth = labels.bool()
                local_tp += int((prediction & truth).sum().item())
                local_fp += int((prediction & ~truth).sum().item())
                local_fn += int((~prediction & truth).sum().item())
                local_tn += int((~prediction & ~truth).sum().item())
                local_positive_pairs += positive_count
                local_negative_pairs += negative_count

            if per_mesh_edge_losses:
                edge_loss = torch.stack(per_mesh_edge_losses).mean()
            else:
                edge_loss = zero_edge_dependency
            edge_loss = edge_loss * edge_loss_ddp_scale
            terms['edge'] = edge_loss
            terms['loss'] = terms['loss'] + self.lambda_edge * edge_loss

            edge_metric_counts = torch.tensor(
                [
                    local_tp,
                    local_fp,
                    local_fn,
                    local_tn,
                    local_positive_pairs,
                    local_negative_pairs,
                    local_positive_weight_sum,
                    local_positive_only_edge_samples,
                    sum(edge_anchor_counts),
                    sum(edge_nearby_counts),
                    sum(edge_random_counts),
                ],
                device=x.device,
                dtype=torch.float64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(edge_metric_counts, op=dist.ReduceOp.SUM)
            (
                tp,
                fp,
                fn,
                tn,
                positive_pairs,
                negative_pairs,
                positive_weight_sum,
                positive_only_samples,
                anchor_vertices,
                nearby_pairs,
                random_pairs,
            ) = edge_metric_counts.tolist()
            eps = 1e-8
            status.update({
                'edge/supervised_samples_local': float(local_supervised_edge_samples),
                'edge/supervised_samples_global': float(global_supervised_edge_samples),
                'edge/loss_ddp_scale': float(edge_loss_ddp_scale),
                'edge/positive_pairs': float(positive_pairs),
                'edge/negative_pairs': float(negative_pairs),
                'edge/positive_only_samples': float(positive_only_samples),
                'edge/anchor_vertices': float(anchor_vertices),
                'edge/negative_pairs_per_anchor': float(
                    negative_pairs / (anchor_vertices + eps)
                ),
                'edge/nearby_negative_pairs': float(nearby_pairs),
                'edge/random_negative_pairs': float(random_pairs),
                'edge/accuracy': float((tp + tn) / (tp + fp + fn + tn + eps)),
                'edge/precision': float(tp / (tp + fp + eps)),
                'edge/recall': float(tp / (tp + fn + eps)),
                'edge/f1': float(2 * tp / (2 * tp + fp + fn + eps)),
                'edge/positive_weight_mean': float(
                    positive_weight_sum / (global_supervised_edge_samples + eps)
                ),
            })
            if self.debug_nans:
                status.update(
                    self._debug_tensor_stats('loss/edge', terms['edge'].reshape(1))
                )

        for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
            terms[f'bce_sub{i}'] = F.binary_cross_entropy_with_logits(sub.feats, sub_gt.float())
            terms['loss'] = terms['loss'] + self.lambda_subdiv * terms[f'bce_sub{i}']
            status.update(self._subdivision_metrics(sub_gt, sub, f'subdiv/sub{i}'))
        if subs:
            status['subdiv/num_stages'] = float(len(subs))

        kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        if self.sample_balanced_loss:
            terms['kl'] = torch.stack([
                kl[sample_slice].mean() for sample_slice in z.layout
            ]).mean()
        else:
            terms['kl'] = kl.mean()
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
        snapshot_dataset = copy.deepcopy(self.validation_dataset or self.dataset)
        dataloader = DataLoader(
            snapshot_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        gt_images = {}
        pred_images = {}
        vertex_images = {}
        decoder_predicts_vertex = bool(
            self._module_attr(self.models['decoder'], 'pred_vertex_subdiv', False)
        )
        decoder_vertex_child_tokens = bool(
            self._module_attr(
                self.models['decoder'],
                'vertex_logits_are_child_tokens',
                False,
            )
        )
        render_vertex_snapshots = (
            decoder_predicts_vertex
            and self.vertex_snapshot_samples > 0
        )
        local_vertex_limit = min(
            num_samples,
            max(
                1,
                (self.vertex_snapshot_samples + self.world_size - 1) // self.world_size,
            ),
        ) if render_vertex_snapshots else 0
        local_stage_limit = min(
            local_vertex_limit,
            max(
                1,
                (
                    self.vertex_snapshot_stage_samples
                    + self.world_size
                    - 1
                ) // self.world_size,
            ),
        ) if self.vertex_snapshot_stage_samples > 0 else 0
        vertex_samples_rendered = 0
        stage_samples_rendered = 0
        if render_vertex_snapshots:
            from ...utils.vertex_subdivision import compute_vertex_hierarchy_predictions
            from ...utils.vertex_visualization import (
                PRED_COLOR,
                VertexVoxelRenderer,
                add_image_label,
                build_child_mask_renderings,
                build_vertex_renderings,
            )

            vertex_renderer = VertexVoxelRenderer(
                image_resolution=self.vertex_snapshot_image_resolution,
                ssaa=self.vertex_snapshot_ssaa,
            )

            def append_vertex_image(
                name: str,
                image: torch.Tensor,
                label: str,
            ) -> None:
                image = add_image_label(image, label)
                vertex_images.setdefault(name, []).append(image.unsqueeze(0))

        self.models['encoder'].eval()
        self.models['decoder'].eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            z = self.models['encoder'](args['x'])
            if render_vertex_snapshots:
                if decoder_vertex_child_tokens:
                    y, vertex_logits = self.models['decoder'](
                        z,
                        return_vertex=True,
                        resolutions=args['resolution'],
                        vertex_threshold=self.vertex_threshold,
                    )
                else:
                    y, vertex_logits = self.models['decoder'](
                        z, return_vertex=True
                    )
            else:
                y = self.models['decoder'](z)
                vertex_logits = None
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

            if render_vertex_snapshots and vertex_samples_rendered < local_vertex_limit:
                if 'vertex_occupancy' not in args or 'resolution' not in args:
                    raise ValueError(
                        'Vertex snapshot requested without vertex_occupancy or resolution.'
                    )
                if not torch.equal(args['vertex_occupancy'].coords, y.coords):
                    raise ValueError(
                        'Vertex snapshot occupancy must align with decoder output support.'
                    )
                output_batch_ids = y.coords[:, 0].long()
                occupancy = args['vertex_occupancy'].feats.reshape(-1).bool()
                for batch_index in range(batch):
                    if vertex_samples_rendered >= local_vertex_limit:
                        break
                    sample_mask = output_batch_ids.eq(batch_index)
                    support_xyz = y.coords[sample_mask, 1:4].long()
                    final_target = occupancy[sample_mask]
                    final_resolution = int(args['resolution'][batch_index].item())
                    if not self._vertex_resolution_is_enabled(final_resolution):
                        continue
                    stages = compute_vertex_hierarchy_predictions(
                        vertex_logits,
                        support_xyz,
                        final_target,
                        batch_index,
                        final_resolution,
                        threshold=self.vertex_threshold,
                    )
                    final_stage = stages[-1]
                    final_images = build_vertex_renderings(
                        vertex_renderer,
                        support_xyz,
                        final_target,
                        final_stage['recursive_prediction'],
                        final_resolution,
                        confidence=final_stage['recursive_score'],
                    )
                    final_labels = {
                        'gt': f'GT QEM VERTICES R{final_resolution}',
                        'prediction': f'RECURSIVE PREDICTION R{final_resolution}',
                        'error': 'ERROR: TP GREEN / FP RED / FN BLUE',
                        'support_context': f'ERRORS ON TRIANGLE SUPPORT R{final_resolution}',
                        'confidence': 'RECURSIVE CONFIDENCE: WEAKEST PATH PROBABILITY',
                    }
                    for name, image in final_images.items():
                        append_vertex_image(
                            f'vertex_final_{name}',
                            image,
                            final_labels[name],
                        )

                    if stage_samples_rendered < local_stage_limit:
                        for stage_index, stage in enumerate(stages):
                            child_resolution = int(stage['child_resolution'].item())
                            stage_images = build_vertex_renderings(
                                vertex_renderer,
                                stage['child_support_xyz'],
                                stage['gt_child'],
                                stage['recursive_prediction'],
                                child_resolution,
                                confidence=stage['recursive_score'],
                            )
                            append_vertex_image(
                                f'vertex_stage{stage_index}_gt',
                                stage_images['gt'],
                                f'STAGE {stage_index} GT CHILDREN R{child_resolution}',
                            )
                            teacher_xyz = stage['child_support_xyz'][
                                stage['teacher_prediction']
                            ]
                            teacher_colors = torch.tensor(
                                PRED_COLOR,
                                device=teacher_xyz.device,
                                dtype=torch.float32,
                            ).expand(len(teacher_xyz), -1)
                            append_vertex_image(
                                f'vertex_stage{stage_index}_teacher_prediction',
                                vertex_renderer.render(
                                    teacher_xyz,
                                    teacher_colors,
                                    child_resolution,
                                ),
                                f'STAGE {stage_index} TEACHER-FORCED R{child_resolution}',
                            )
                            stage_labels = {
                                'prediction': f'STAGE {stage_index} RECURSIVE R{child_resolution}',
                                'error': f'STAGE {stage_index} RECURSIVE ERRORS',
                                'support_context': f'STAGE {stage_index} ERRORS ON TRIANGLE SUPPORT',
                                'confidence': f'STAGE {stage_index} WEAKEST-PATH CONFIDENCE',
                            }
                            for name, label in stage_labels.items():
                                append_vertex_image(
                                    f'vertex_stage{stage_index}_recursive_{name}',
                                    stage_images[name],
                                    label,
                                )

                            child_images = build_child_mask_renderings(
                                vertex_renderer,
                                stage['example_gt'],
                                stage['example_teacher_prediction'],
                                stage['example_local_probability'],
                            )
                            append_vertex_image(
                                f'vertex_stage{stage_index}_child_mask_error',
                                child_images['error'],
                                f'HEAD {stage_index} ONE-PARENT LOCAL CHILD ERRORS',
                            )
                            append_vertex_image(
                                f'vertex_stage{stage_index}_child_mask_confidence',
                                child_images['confidence'],
                                f'HEAD {stage_index} ONE-PARENT LOCAL PROBABILITY',
                            )
                        stage_samples_rendered += 1
                    vertex_samples_rendered += 1
        self.models['encoder'].train()
        self.models['decoder'].train()

        sample_dict = {}
        for k in gt_images:
            sample_dict[f'gt_{k}'] = {'value': torch.cat(gt_images[k], dim=0)[:num_samples], 'type': 'image'}
            sample_dict[f'pred_{k}'] = {'value': torch.cat(pred_images[k], dim=0)[:num_samples], 'type': 'image'}
        for k, images in vertex_images.items():
            sample_dict[k] = {
                'value': torch.cat(images, dim=0),
                'type': 'image',
            }
        return sample_dict
