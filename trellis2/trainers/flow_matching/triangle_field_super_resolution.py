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
from ..utils import make_master_params, master_params_to_model_params
from ..vae.triangle_field_vae import TriangleFieldVaeTrainer
from ... import models
from ...modules import sparse as sp
from ...utils.data_utils import (
    recursive_to_device,
    cycle,
    BalancedResumableSampler,
    GroupedBalancedResumableBatchSampler,
)


DENSITY_STATISTIC_CONDITION_NAMES = (
    'density_minimum',
    'density_median',
    'density_maximum',
)


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
        cond_partial_drop_prob: float = 0.0,
        cfg_drop_field_condition: bool = True,
        condition_drop_values: dict = None,
        loss_type: str = 'l1',
        sample_posterior: bool = False,
        snapshot_t: float = 0.5,
        voxel_loss_weight: dict = None,
        conditioning_augmentation: dict = None,
        zero_init_input_layer_weight: bool = False,
        partial_load_input_layer_weight: bool = True,
        input_layer_channel_map: list = None,
        pretrained_freeze_steps: int = 0,
        post_unfreeze_warmup_steps: int = None,
        fp16_after_step: int = None,
        fp16_initial_log_scale: float = 20.0,
        finetune_ckpt: dict = None,
        **kwargs,
    ):
        self.decoder_model_config = decoder_model
        self.decoder_ckpt = decoder_ckpt
        self.num_workers = num_workers
        self.cond_drop_prob = float(cond_drop_prob)
        self.cond_partial_drop_prob = float(cond_partial_drop_prob)
        self.cfg_drop_field_condition = bool(cfg_drop_field_condition)
        self.condition_drop_values = {
            'field': 0.0,
            'density': 0.0,
            'elongation': 0.0,
            **{name: 0.0 for name in DENSITY_STATISTIC_CONDITION_NAMES},
        }
        if condition_drop_values is not None:
            unknown = set(condition_drop_values) - set(self.condition_drop_values)
            if unknown:
                raise ValueError(f'Unknown condition_drop_values keys: {sorted(unknown)}')
            self.condition_drop_values.update({
                name: float(value)
                for name, value in condition_drop_values.items()
            })
        if not all(np.isfinite(value) for value in self.condition_drop_values.values()):
            raise ValueError(f'condition_drop_values must be finite, got {self.condition_drop_values}')
        self.loss_type = loss_type
        self.sample_posterior = sample_posterior
        self.snapshot_t = float(snapshot_t)
        self.voxel_loss_weight = voxel_loss_weight
        self.conditioning_augmentation = conditioning_augmentation
        self.zero_init_input_layer_weight = bool(zero_init_input_layer_weight)
        self.partial_load_input_layer_weight = bool(partial_load_input_layer_weight)
        self.input_layer_channel_map = (
            None
            if input_layer_channel_map is None
            else [
                None if source is None else int(source)
                for source in input_layer_channel_map
            ]
        )
        if self.input_layer_channel_map is not None:
            invalid = [
                source
                for source in self.input_layer_channel_map
                if source is not None and source < 0
            ]
            if invalid:
                raise ValueError(
                    'input_layer_channel_map entries must be non-negative source indices or null, '
                    f'got {invalid}'
                )
        self.pretrained_freeze_steps = int(pretrained_freeze_steps)
        self.post_unfreeze_warmup_steps = (
            self.pretrained_freeze_steps
            if post_unfreeze_warmup_steps is None
            else int(post_unfreeze_warmup_steps)
        )
        self.fp16_after_step = None if fp16_after_step is None else int(fp16_after_step)
        self.fp16_initial_log_scale = float(fp16_initial_log_scale)
        if self.pretrained_freeze_steps < 0:
            raise ValueError(f'pretrained_freeze_steps must be non-negative, got {self.pretrained_freeze_steps}')
        if self.post_unfreeze_warmup_steps < 0:
            raise ValueError(
                f'post_unfreeze_warmup_steps must be non-negative, got {self.post_unfreeze_warmup_steps}'
            )
        if self.fp16_after_step is not None and self.fp16_after_step < 0:
            raise ValueError(f'fp16_after_step must be non-negative, got {self.fp16_after_step}')
        self._pretrained_frozen_param_names = {}
        resume_step = kwargs.get('step')
        should_start_frozen = (
            self.pretrained_freeze_steps > 0 and
            (resume_step is None or int(resume_step) <= self.pretrained_freeze_steps)
        )
        if should_start_frozen:
            if finetune_ckpt is None:
                raise ValueError('pretrained_freeze_steps requires finetune_ckpt')
            models = args[0]
            for model_name, ckpt_path in finetune_ckpt.items():
                if model_name not in models:
                    continue
                model = models[model_name]
                model_state = model.state_dict()
                raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                frozen_names = {
                    name
                    for name, param in model.named_parameters()
                    if name in raw and name in model_state and raw[name].shape == param.shape
                }
                self._pretrained_frozen_param_names[model_name] = frozen_names
        self._cond_blur_cache = {}
        if not (0.0 <= self.cond_drop_prob <= 1.0):
            raise ValueError(f'cond_drop_prob must be in [0, 1], got {self.cond_drop_prob}')
        if not (0.0 <= self.cond_partial_drop_prob <= 1.0):
            raise ValueError(
                f'cond_partial_drop_prob must be in [0, 1], got {self.cond_partial_drop_prob}'
            )
        if self.cond_drop_prob + self.cond_partial_drop_prob > 1.0:
            raise ValueError(
                'cond_drop_prob + cond_partial_drop_prob must be <= 1, got '
                f'{self.cond_drop_prob + self.cond_partial_drop_prob}'
            )
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
        super().__init__(
            *args,
            fp16_initial_log_scale=self.fp16_initial_log_scale,
            finetune_ckpt=finetune_ckpt,
            **kwargs,
        )
        self._pretrained_unfrozen = not should_start_frozen
        if should_start_frozen:
            self._build_pretrained_freeze_phase()
        if self.fp16_after_step is not None and self.mix_precision_mode != 'inflat_all':
            raise ValueError('fp16_after_step currently requires mix_precision_mode="inflat_all"')
        if self.fp16_after_step is not None and self.step < self.fp16_after_step:
            if self.mix_precision_dtype != torch.bfloat16:
                raise ValueError('fp16_after_step requires bfloat16 as the initial mixed-precision dtype')
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
            f'  - Cond partial drop prob: {self.cond_partial_drop_prob}',
            f'  - CFG drops field condition: {self.cfg_drop_field_condition}',
            f'  - Condition drop values: {self.condition_drop_values}',
            f'  - Loss type: {self.loss_type}',
            f'  - Sample posterior: {self.sample_posterior}',
            f'  - Snapshot t: {self.snapshot_t}',
            f'  - Voxel loss weight: {self.voxel_loss_weight}',
            f'  - Conditioning augmentation: {self.conditioning_augmentation}',
            f'  - Zero-init input layer weight: {self.zero_init_input_layer_weight}',
            f'  - Partial-load input layer weight: {self.partial_load_input_layer_weight}',
            f'  - Input layer channel map: {self.input_layer_channel_map}',
            f'  - Freeze pretrained parameters for steps: {self.pretrained_freeze_steps}',
            f'  - Post-unfreeze LR warmup steps: {self.post_unfreeze_warmup_steps}',
            f'  - Switch BF16 to FP16 after step: {self.fp16_after_step}',
        ]
        return '\n'.join(lines)

    def _reset_optimizer_and_scheduler(self, warmup_steps: int) -> None:
        if self.mix_precision_mode != 'inflat_all':
            raise ValueError('pretrained staged unfreezing currently requires mix_precision_mode="inflat_all"')
        self.master_params = make_master_params(self.model_params)
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        optimizer_cls = getattr(torch.optim, self.optimizer_config['name'])
        self.optimizer = optimizer_cls(self.master_params, **self.optimizer_config['args'])
        if self.lr_scheduler_config is not None:
            scheduler_args = copy.deepcopy(self.lr_scheduler_config['args'])
            scheduler_args['warmup_steps'] = warmup_steps
            self.lr_scheduler = type(self.lr_scheduler)(self.optimizer, **scheduler_args)
        if self.mix_precision_dtype == torch.float16:
            self.log_scale = self.fp16_initial_log_scale

    def _build_pretrained_freeze_phase(self) -> None:
        self.model_params = [
            param
            for model_name, model in self.models.items()
            for name, param in model.named_parameters()
            if name not in self._pretrained_frozen_param_names.get(model_name, set())
        ]
        self._reset_optimizer_and_scheduler(self.lr_scheduler_config['args']['warmup_steps'])
        if self.is_master:
            frozen_count = sum(len(names) for names in self._pretrained_frozen_param_names.values())
            print(
                f'Frozen {frozen_count} pretrained parameter tensors by excluding them from '
                'the optimizer; autograd remains enabled for flex_gemm compatibility.'
            )

    def _clear_pretrained_frozen_grads(self) -> None:
        for model_name, frozen_names in self._pretrained_frozen_param_names.items():
            for name, param in self.models[model_name].named_parameters():
                if name in frozen_names:
                    param.grad = None

    def _maybe_unfreeze_pretrained(self) -> None:
        if self._pretrained_unfrozen or self.step < self.pretrained_freeze_steps:
            return
        if self.world_size > 1:
            dist.barrier()

        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()],
            [],
        )
        self._reset_optimizer_and_scheduler(self.post_unfreeze_warmup_steps)
        self._pretrained_unfrozen = True

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(
                f'\nUnfroze pretrained parameters after {self.pretrained_freeze_steps} completed steps; '
                f'rebuilt the optimizer and started a {self.post_unfreeze_warmup_steps}-step LR warmup.'
            )

    def _maybe_switch_to_fp16(self) -> None:
        if (
            self.fp16_after_step is None or
            self.step < self.fp16_after_step or
            self.mix_precision_dtype == torch.float16
        ):
            return
        if self.mix_precision_dtype != torch.bfloat16:
            raise ValueError(f'Expected bfloat16 before FP16 switch, got {self.mix_precision_dtype}')
        self.training_models = self.models
        for param in self.model_params:
            param.grad = None
        for name, model in self.models.items():
            if not hasattr(model, 'convert_to_fp16'):
                raise ValueError(f'Model {name} does not support convert_to_fp16()')
            model.convert_to_fp16()
        master_params_to_model_params(self.model_params, self.master_params)
        if self.parallel_mode == 'ddp' and self.world_size > 1:
            self.training_models = {
                name: torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
                for name, model in self.models.items()
            }
        self.mix_precision_dtype = torch.float16
        if not hasattr(self, 'log_scale'):
            self.log_scale = self.fp16_initial_log_scale
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(
                f'\nSwitched training compute from bfloat16 to float16 after '
                f'{self.fp16_after_step} completed steps (log_scale={self.log_scale}).'
            )

    def run_step(self, data_list):
        self._maybe_unfreeze_pretrained()
        self._maybe_switch_to_fp16()
        if not self._pretrained_unfrozen:
            self._clear_pretrained_frozen_grads()
        return super().run_step(data_list)

    def load(self, load_dir, step=0):
        super().load(load_dir, step)
        self._maybe_switch_to_fp16()
        if self.fp16_after_step is not None and self.mix_precision_dtype == torch.float16:
            misc_path = os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')
            misc_ckpt = torch.load(misc_path, map_location='cpu', weights_only=False)
            self.log_scale = float(misc_ckpt.get('log_scale', self.fp16_initial_log_scale))

    @staticmethod
    def _validate_conditioning_augmentation(cfg: dict) -> None:
        if cfg.get('type') != 'sparse_blur_noise':
            raise ValueError(
                f"Unsupported conditioning_augmentation type {cfg.get('type')}. "
                "Expected 'sparse_blur_noise'."
            )
        disable_blur = bool(cfg.get('disable_blur', False))
        blur_sigma = float(cfg.get('blur_sigma', 1.0))
        noise_level = float(cfg.get('noise_level', 0.0))
        apply_prob = float(cfg.get('apply_prob', 1.0))
        noise_level_sampling = cfg.get('noise_level_sampling', 'fixed')
        noise_level_min = float(cfg.get('noise_level_min', 0.0))
        noise_level_max = float(cfg.get('noise_level_max', 1.0))
        noise_interpolation = cfg.get('noise_interpolation', 'legacy')
        training_overrides = cfg.get('training_noise_level_by_high_resolution', {})
        if not disable_blur and blur_sigma <= 0:
            raise ValueError(f'conditioning_augmentation.blur_sigma must be positive, got {blur_sigma}')
        if not (0.0 <= noise_level <= 1.0):
            raise ValueError(f'conditioning_augmentation.noise_level must be in [0, 1], got {noise_level}')
        if not (0.0 <= apply_prob <= 1.0):
            raise ValueError(f'conditioning_augmentation.apply_prob must be in [0, 1], got {apply_prob}')
        if noise_level_sampling not in ('fixed', 'uniform'):
            raise ValueError(
                "conditioning_augmentation.noise_level_sampling must be 'fixed' or 'uniform', "
                f'got {noise_level_sampling}'
            )
        if not (0.0 <= noise_level_min <= noise_level_max <= 1.0):
            raise ValueError(
                'conditioning_augmentation noise level range must satisfy '
                f'0 <= min <= max <= 1, got [{noise_level_min}, {noise_level_max}]'
            )
        if noise_interpolation not in ('legacy', 'flow'):
            raise ValueError(
                "conditioning_augmentation.noise_interpolation must be 'legacy' or 'flow', "
                f'got {noise_interpolation}'
            )
        if not isinstance(training_overrides, dict):
            raise ValueError(
                'conditioning_augmentation.training_noise_level_by_high_resolution '
                'must be a mapping'
            )
        invalid_overrides = {
            key: value
            for key, value in training_overrides.items()
            if not (0.0 <= float(value) <= 1.0)
        }
        if invalid_overrides:
            raise ValueError(
                'conditioning augmentation resolution overrides must be in [0, 1], '
                f'got {invalid_overrides}'
            )

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
    def _augment_conditioning(
        self,
        cond: sp.SparseTensor,
        *,
        sample_noise_level: bool = False,
        noise_level: Optional[torch.Tensor] = None,
        high_resolution: Optional[torch.Tensor] = None,
        return_noise_level: bool = False,
    ) -> Union[sp.SparseTensor, Tuple[sp.SparseTensor, torch.Tensor]]:
        cfg = self.conditioning_augmentation
        if cfg is None:
            levels = torch.zeros(cond.shape[0], device=cond.feats.device, dtype=torch.float32)
            return (cond, levels) if return_noise_level else cond

        apply_prob = float(cfg.get('apply_prob', 1.0))
        if apply_prob <= 0.0:
            levels = torch.zeros(cond.shape[0], device=cond.feats.device, dtype=torch.float32)
            return (cond, levels) if return_noise_level else cond
        if apply_prob < 1.0:
            apply = torch.rand(cond.shape[0], device=cond.feats.device) < apply_prob
            if not apply.any():
                levels = torch.zeros(cond.shape[0], device=cond.feats.device, dtype=torch.float32)
                return (cond, levels) if return_noise_level else cond
        else:
            apply = torch.ones(cond.shape[0], device=cond.feats.device, dtype=torch.bool)

        disable_blur = bool(cfg.get('disable_blur', False))
        blur_sigma = float(cfg.get('blur_sigma', 1.0))
        if noise_level is None:
            if sample_noise_level and cfg.get('noise_level_sampling', 'fixed') == 'uniform':
                noise_level_min = float(cfg.get('noise_level_min', 0.0))
                noise_level_max = float(cfg.get('noise_level_max', 1.0))
                levels = torch.rand(
                    cond.shape[0],
                    device=cond.feats.device,
                    dtype=torch.float32,
                )
                levels = noise_level_min + (noise_level_max - noise_level_min) * levels
            else:
                levels = torch.full(
                    (cond.shape[0],),
                    float(cfg.get('noise_level', 0.0)),
                    device=cond.feats.device,
                    dtype=torch.float32,
                )
        else:
            levels = torch.as_tensor(
                noise_level,
                device=cond.feats.device,
                dtype=torch.float32,
            ).reshape(-1)
            if levels.numel() == 1 and cond.shape[0] > 1:
                levels = levels.expand(cond.shape[0]).clone()
            if levels.numel() != cond.shape[0]:
                raise ValueError(
                    f'noise_level must have {cond.shape[0]} entries, got {levels.numel()}'
                )
        if torch.any((levels < 0) | (levels > 1)):
            raise ValueError('conditioning noise levels must be in [0, 1]')

        if sample_noise_level and high_resolution is not None:
            resolution = torch.as_tensor(
                high_resolution,
                device=cond.feats.device,
                dtype=torch.long,
            ).reshape(-1)
            if resolution.numel() == 1 and cond.shape[0] > 1:
                resolution = resolution.expand(cond.shape[0])
            if resolution.numel() != cond.shape[0]:
                raise ValueError(
                    f'high_resolution must have {cond.shape[0]} entries, got {resolution.numel()}'
                )
            for key, value in cfg.get(
                'training_noise_level_by_high_resolution',
                {},
            ).items():
                levels[resolution == int(key)] = float(value)
        levels = torch.where(apply, levels, torch.zeros_like(levels))

        if disable_blur:
            aug_feats = cond.feats
        else:
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
        voxel_level = levels[cond.coords[:, 0].long()].reshape(-1, 1).to(aug_feats.dtype)
        if torch.any(levels > 0):
            if cfg.get('noise_interpolation', 'legacy') == 'flow':
                noise_scale = self.sigma_min + (1.0 - self.sigma_min) * voxel_level
            else:
                noise_scale = voxel_level
            aug_feats = (
                (1.0 - voxel_level) * aug_feats
                + noise_scale * torch.randn_like(aug_feats)
            )
            if bool(cfg.get('clamp', True)):
                aug_feats = aug_feats.clamp(-1.0, 1.0)

        feats = cond.feats.clone()
        feats[apply[cond.coords[:, 0].long()]] = aug_feats[apply[cond.coords[:, 0].long()]]
        augmented = cond.replace(feats)
        return (augmented, levels) if return_noise_level else augmented

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
        Partially load pretrained VAE encoder weights. An explicit input channel
        map copies semantically matching columns and zero-initializes unmapped
        columns. Otherwise, mismatched input layers use prefix loading.
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
            partially_loaded = []
            for k, v in raw.items():
                if k not in model_state:
                    skipped.append((k, 'missing_in_model'))
                elif (
                    name == 'encoder' and
                    k.endswith('input_layer.weight') and
                    self.input_layer_channel_map is not None
                ):
                    if v.ndim != 2 or model_state[k].ndim != 2 or v.shape[0] != model_state[k].shape[0]:
                        raise ValueError(
                            f'Cannot apply input_layer_channel_map to checkpoint shape {tuple(v.shape)} '
                            f'and model shape {tuple(model_state[k].shape)}'
                        )
                    if len(self.input_layer_channel_map) != model_state[k].shape[1]:
                        raise ValueError(
                            f'input_layer_channel_map has {len(self.input_layer_channel_map)} entries, '
                            f'but the model input layer has {model_state[k].shape[1]} channels'
                        )
                    invalid = [
                        source
                        for source in self.input_layer_channel_map
                        if source is not None and source >= v.shape[1]
                    ]
                    if invalid:
                        raise ValueError(
                            f'input_layer_channel_map references checkpoint channels {invalid}, '
                            f'but the checkpoint input layer has only {v.shape[1]} channels'
                        )
                    merged = torch.zeros_like(model_state[k])
                    mapped = []
                    for target, source in enumerate(self.input_layer_channel_map):
                        if source is None:
                            continue
                        merged[:, target] = v[:, source]
                        mapped.append(f'{target}<-{source}')
                    loadable[k] = merged
                    partially_loaded.append((
                        k,
                        f'semantically mapped {len(mapped)}/{model_state[k].shape[1]} columns '
                        f'({", ".join(mapped)}); unmapped columns zero-initialized',
                    ))
                elif (
                    k.endswith('input_layer.weight') and
                    self.partial_load_input_layer_weight and
                    v.ndim == 2 and
                    model_state[k].ndim == 2 and
                    v.shape[0] == model_state[k].shape[0] and
                    v.shape[1] != model_state[k].shape[1]
                ):
                    merged = model_state[k].clone()
                    copied_channels = min(v.shape[1], model_state[k].shape[1])
                    merged[:, :copied_channels] = v[:, :copied_channels]
                    loadable[k] = merged
                    partially_loaded.append((
                        k,
                        f'copied {copied_channels} prefix columns '
                        f'from checkpoint {v.shape[1]} -> model {model_state[k].shape[1]}',
                    ))
                elif v.shape != model_state[k].shape:
                    skipped.append((k, f'shape {tuple(v.shape)} != {tuple(model_state[k].shape)}'))
                else:
                    loadable[k] = v
            missing, unexpected = model.load_state_dict(loadable, strict=False)
            if self.is_master:
                for k, reason in partially_loaded:
                    print(f'Info: partially loaded {name}.{k}: {reason}')
                for k, reason in skipped:
                    print(f'Warning: skipped {name}.{k}: {reason}')
                for k in missing:
                    print(f'Warning: left initialized {name}.{k}')
                for k in unexpected:
                    print(f'Warning: unexpected {name}.{k}')
            if self.zero_init_input_layer_weight and name == 'encoder':
                if not hasattr(model, 'input_layer') or not hasattr(model.input_layer, 'weight'):
                    raise ValueError('zero_init_input_layer_weight requires encoder.input_layer.weight')
                with torch.no_grad():
                    model.input_layer.weight.zero_()
                if self.is_master:
                    print('Info: zero-initialized encoder.input_layer.weight after finetune loading')
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

    def _drop_conditions(
        self,
        cond: sp.SparseTensor,
        density_cond: Optional[sp.SparseTensor] = None,
        elongation_cond: Optional[sp.SparseTensor] = None,
        force_field_drop: Optional[torch.Tensor] = None,
        density_statistics: Optional[torch.Tensor] = None,
    ) -> Tuple[
        sp.SparseTensor,
        Optional[sp.SparseTensor],
        Optional[sp.SparseTensor],
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        batch_size = cond.shape[0]
        condition_names = ['field']
        conditions = [cond]
        for name, tensor in (
            ('density', density_cond),
            ('elongation', elongation_cond),
        ):
            if tensor is None:
                continue
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f'cond and {name}_cond batch sizes must match, got {batch_size} and {tensor.shape[0]}'
                )
            condition_names.append(name)
            conditions.append(tensor)
        if density_statistics is not None:
            density_statistics = torch.as_tensor(
                density_statistics,
                device=cond.feats.device,
                dtype=torch.float32,
            )
            if density_statistics.shape != (batch_size, 3):
                raise ValueError(
                    f'density_statistics must have shape ({batch_size}, 3), '
                    f'got {tuple(density_statistics.shape)}'
                )
            if not torch.isfinite(density_statistics).all():
                raise ValueError('density_statistics contains non-finite values')
            for index, name in enumerate(DENSITY_STATISTIC_CONDITION_NAMES):
                condition_names.append(name)
                conditions.append(density_statistics[:, index:index + 1])

        event = torch.rand(batch_size, device=cond.feats.device)
        drop_all = event < self.cond_drop_prob
        drop_partial = (
            (event >= self.cond_drop_prob) &
            (event < self.cond_drop_prob + self.cond_partial_drop_prob)
        )
        if drop_partial.any() and len(conditions) < 2:
            raise ValueError('Partial condition dropout requires at least two condition groups.')

        drop_masks = torch.zeros(
            (batch_size, len(conditions)),
            device=cond.feats.device,
            dtype=torch.bool,
        )
        droppable_indices = [
            index
            for index, name in enumerate(condition_names)
            if name != 'field' or self.cfg_drop_field_condition
        ]
        if not droppable_indices and (drop_all.any() or drop_partial.any()):
            raise ValueError('CFG dropout is enabled but no condition groups are droppable.')
        droppable_mask = torch.zeros(
            len(conditions),
            device=cond.feats.device,
            dtype=torch.bool,
        )
        droppable_mask[droppable_indices] = True
        drop_masks[drop_all] = droppable_mask
        num_partial = int(drop_partial.sum().item())
        if num_partial > 0:
            num_conditions = len(droppable_indices)
            if num_conditions < 2:
                raise ValueError(
                    'Partial condition dropout requires at least two droppable condition groups.'
                )
            subset_sizes = torch.randint(
                1,
                num_conditions,
                (num_partial,),
                device=cond.feats.device,
            )
            ranks = torch.rand(
                (num_partial, num_conditions),
                device=cond.feats.device,
            ).argsort(dim=1).argsort(dim=1)
            partial_rows = drop_partial.nonzero(as_tuple=False).reshape(-1)
            partial_masks = ranks < subset_sizes[:, None]
            for local_index, condition_index in enumerate(droppable_indices):
                drop_masks[partial_rows, condition_index] = partial_masks[:, local_index]
        if force_field_drop is not None:
            force_field_drop = force_field_drop.to(device=cond.feats.device, dtype=torch.bool).reshape(-1)
            if force_field_drop.shape[0] != batch_size:
                raise ValueError(
                    f'force_field_drop must have {batch_size} entries, got {force_field_drop.shape[0]}'
                )
            drop_masks[:, 0] |= force_field_drop

        dropped_conditions = []
        for name, tensor, sample_drop in zip(
            condition_names,
            conditions,
            drop_masks.unbind(dim=1),
        ):
            if sample_drop.any():
                use_presence = (
                    name in DENSITY_STATISTIC_CONDITION_NAMES or
                    (
                        getattr(self, 'scalar_condition_presence', False) and
                        name in ('density', 'elongation')
                    )
                )
                drop_value = 0.0 if use_presence else self.condition_drop_values[name]
                if isinstance(tensor, sp.SparseTensor):
                    feats = tensor.feats.clone()
                    feats[sample_drop[tensor.coords[:, 0].long()]] = drop_value
                    tensor = tensor.replace(feats)
                else:
                    tensor = tensor.clone()
                    tensor[sample_drop] = drop_value
            dropped_conditions.append(tensor)

        masks = {
            'all': drop_all,
            'partial': drop_partial,
        }
        masks.update({
            name: drop_masks[:, index]
            for index, name in enumerate(condition_names)
        })
        dropped_by_name = dict(zip(condition_names, dropped_conditions))
        dropped_density_statistics = None
        if density_statistics is not None:
            dropped_density_statistics = torch.cat([
                dropped_by_name[name]
                for name in DENSITY_STATISTIC_CONDITION_NAMES
            ], dim=-1)
        return (
            dropped_by_name['field'],
            dropped_by_name.get('density'),
            dropped_by_name.get('elongation'),
            dropped_density_statistics,
            masks,
        )

    @staticmethod
    def _make_encoder_input(
        x_t: sp.SparseTensor,
        cond: sp.SparseTensor,
        extra_cond: Optional[Union[sp.SparseTensor, List[sp.SparseTensor]]] = None,
    ) -> sp.SparseTensor:
        if not torch.equal(x_t.coords, cond.coords):
            raise ValueError(f'x_t and cond coords must match, got {x_t.coords.shape} vs {cond.coords.shape}')
        feats = [x_t.feats, cond.feats]
        if extra_cond is not None:
            if isinstance(extra_cond, sp.SparseTensor):
                extra_cond = [extra_cond]
            for i, tensor in enumerate(extra_cond):
                if tensor is None:
                    continue
                if not torch.equal(x_t.coords, tensor.coords):
                    raise ValueError(
                        f'extra_cond[{i}] coords must match x_t coords, '
                        f'got {tensor.coords.shape} vs {x_t.coords.shape}'
                    )
                feats.append(tensor.feats)
        return x_t.replace(torch.cat(feats, dim=-1))

    def _predict_x0(
        self,
        x_t: sp.SparseTensor,
        cond: sp.SparseTensor,
        t: torch.Tensor,
        extra_cond: Optional[Union[sp.SparseTensor, List[sp.SparseTensor]]] = None,
        conditioning_noise_level: Optional[torch.Tensor] = None,
    ) -> sp.SparseTensor:
        enc_in = self._make_encoder_input(x_t, cond, extra_cond)
        encoder_kwargs = {}
        if getattr(
            self.models['encoder'],
            'conditioning_noise_conditioning',
            False,
        ):
            encoder_kwargs['conditioning_noise_level'] = conditioning_noise_level
        z = self.training_models['encoder'](
            enc_in,
            t * 1000.0,
            sample_posterior=self.sample_posterior,
            **encoder_kwargs,
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
        density_cond: sp.SparseTensor = None,
        density_missing_parent_frac: torch.Tensor = None,
        elongation_cond: sp.SparseTensor = None,
        elongation_missing_parent_frac: torch.Tensor = None,
        area_offsets: sp.SparseTensor = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        self.decoder.zero_grad(set_to_none=True)
        t = self.sample_t(x_0.shape[0]).to(x_0.feats.device).float()
        x_t, _ = self._diffuse_sparse(x_0, t)
        cond, conditioning_noise_level = self._augment_conditioning(
            cond,
            sample_noise_level=True,
            high_resolution=kwargs.get('high_resolution', None),
            return_noise_level=True,
        )
        cond, density_cond, elongation_cond, _, cond_drop = self._drop_conditions(
            cond,
            density_cond,
            elongation_cond,
        )
        pred_x0 = self._predict_x0(
            x_t,
            cond,
            t,
            extra_cond=[density_cond, elongation_cond],
            conditioning_noise_level=conditioning_noise_level,
        )
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
            'cond/drop_frac': cond_drop['all'].float().mean(),
            'cond/drop_all_frac': cond_drop['all'].float().mean(),
            'cond/drop_partial_frac': cond_drop['partial'].float().mean(),
            'cond/field_drop_frac': cond_drop['field'].float().mean(),
            'cond/noise_level_mean': conditioning_noise_level.mean(),
            'cond/noise_level_min': conditioning_noise_level.min(),
            'cond/noise_level_max': conditioning_noise_level.max(),
        }
        if 'density' in cond_drop:
            status['cond/density_drop_frac'] = cond_drop['density'].float().mean()
        if 'elongation' in cond_drop:
            status['cond/elongation_drop_frac'] = cond_drop['elongation'].float().mean()
        if missing_low_parent_frac is not None:
            missing_low_parent_frac = missing_low_parent_frac.float()
            status['cond/missing_low_parent_frac'] = missing_low_parent_frac.mean()
            status['cond/missing_low_parent_frac_max'] = missing_low_parent_frac.max()
        if density_missing_parent_frac is not None:
            density_missing_parent_frac = density_missing_parent_frac.float()
            status['density_cond/missing_parent_frac'] = density_missing_parent_frac.mean()
            status['density_cond/missing_parent_frac_max'] = density_missing_parent_frac.max()
        if elongation_missing_parent_frac is not None:
            elongation_missing_parent_frac = elongation_missing_parent_frac.float()
            status['elongation_cond/missing_parent_frac'] = elongation_missing_parent_frac.mean()
            status['elongation_cond/missing_parent_frac_max'] = elongation_missing_parent_frac.max()

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
            snapshot_cond, conditioning_noise_level = self._augment_conditioning(
                args['cond'],
                sample_noise_level=True,
                high_resolution=args.get('high_resolution', None),
                return_noise_level=True,
            )
            enc_in = self._make_encoder_input(x_t, snapshot_cond, args.get('density_cond', None))
            encoder_kwargs = {}
            if getattr(
                self.models['encoder'],
                'conditioning_noise_conditioning',
                False,
            ):
                encoder_kwargs['conditioning_noise_level'] = conditioning_noise_level
            z = self.models['encoder'](
                enc_in,
                t * 1000.0,
                sample_posterior=False,
                **encoder_kwargs,
            )
            y = self.decoder(z)

            gt_vis = self.dataset.visualize_sample({'target': args['x_0']})
            cond_vis = self.dataset.visualize_sample({'target': snapshot_cond})
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
        decoded_density_mode: str = 'none',
        scalar_condition_presence: bool = False,
        density_statistics_relax_probability: float = 0.0,
        density_statistics_relax_std: float = 1.5,
        batch_size_per_gpu_by_high_resolution: dict = None,
        resolution_sampling_weights: dict = None,
        **kwargs,
    ):
        self.latent_loss_parameterization = latent_loss_parameterization
        self.latent_loss_t_min = float(latent_loss_t_min)
        self.batched_cache_decode = bool(batched_cache_decode)
        self.latent_self_conditioning = latent_self_conditioning or {'mode': 'none'}
        self.latent_self_conditioning_mode = self.latent_self_conditioning.get('mode', 'none')
        self.latent_self_conditioning_upsample_factor = self.latent_self_conditioning.get('upsample_factor', None)
        self.decoded_density_mode = str(decoded_density_mode)
        self.scalar_condition_presence = bool(scalar_condition_presence)
        self.density_statistics_relax_probability = float(
            density_statistics_relax_probability
        )
        self.density_statistics_relax_std = float(density_statistics_relax_std)
        self.batch_size_per_gpu_by_high_resolution = (
            {int(key): int(value) for key, value in batch_size_per_gpu_by_high_resolution.items()}
            if batch_size_per_gpu_by_high_resolution is not None else None
        )
        self.resolution_sampling_weights = (
            {int(key): float(value) for key, value in resolution_sampling_weights.items()}
            if resolution_sampling_weights is not None else None
        )
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
        if self.decoded_density_mode not in ('none', 'condition', 'state'):
            raise ValueError(
                "decoded_density_mode must be 'none', 'condition', or 'state', "
                f'got {self.decoded_density_mode}'
            )
        if not (0.0 <= self.density_statistics_relax_probability <= 1.0):
            raise ValueError(
                'density_statistics_relax_probability must be in [0, 1], got '
                f'{self.density_statistics_relax_probability}'
            )
        if self.density_statistics_relax_std < 0:
            raise ValueError(
                'density_statistics_relax_std must be non-negative, got '
                f'{self.density_statistics_relax_std}'
            )
        super().__init__(*args, **kwargs)
        model_uses_statistics = bool(getattr(
            self.models['encoder'],
            'density_statistics_conditioning',
            False,
        ))
        dataset_has_statistics = getattr(self.dataset, 'density_statistics', None) is not None
        if model_uses_statistics != dataset_has_statistics:
            raise ValueError(
                'Encoder density_statistics_conditioning and dataset density_statistics_path '
                f'must be enabled together, got {model_uses_statistics} and '
                f'{dataset_has_statistics}'
            )
        if (
            self.decoded_density_mode == 'condition' and
            getattr(self.dataset, 'density_conditioning', False)
        ):
            raise ValueError(
                "decoded_density_mode='condition' uses decoded density as the density condition, "
                'so dataset density_conditioning must be false.'
            )

    def prepare_dataloader(self, **kwargs):
        if self.batch_size_per_gpu_by_high_resolution is None:
            return super().prepare_dataloader(**kwargs)
        if self.batch_split != 1:
            raise ValueError('Resolution-specific batches require batch_split=1.')
        if not hasattr(self.dataset, 'datasets') or not hasattr(self.dataset, '_cumulative_sizes'):
            raise ValueError(
                'Resolution-specific batches require a multi-resolution dataset with pair datasets.'
            )

        groups = {}
        previous_size = 0
        for dataset, cumulative_size in zip(self.dataset.datasets, self.dataset._cumulative_sizes):
            groups[int(dataset.high_resolution)] = range(previous_size, cumulative_size)
            previous_size = cumulative_size
        if set(groups) != set(self.batch_size_per_gpu_by_high_resolution):
            raise ValueError(
                'batch_size_per_gpu_by_high_resolution must define every high resolution: '
                f'{sorted(groups)}; got {sorted(self.batch_size_per_gpu_by_high_resolution)}'
            )
        if (
            self.resolution_sampling_weights is not None
            and set(groups) != set(self.resolution_sampling_weights)
        ):
            raise ValueError(
                'resolution_sampling_weights must define every high resolution: '
                f'{sorted(groups)}; got {sorted(self.resolution_sampling_weights)}'
            )

        self.data_sampler = GroupedBalancedResumableBatchSampler(
            self.dataset,
            groups=groups,
            batch_sizes=self.batch_size_per_gpu_by_high_resolution,
            group_weights=self.resolution_sampling_weights,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.data_sampler,
            num_workers=(
                self.num_workers
                if self.num_workers is not None
                else int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
            ),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn,
        )
        self.data_iterator = self._cycle_resolution_dataloader()

    def _cycle_resolution_dataloader(self):
        while True:
            for data in self.dataloader:
                self.data_sampler.idx += 1
                yield data
            self.data_sampler.epoch += 1
            self.data_sampler.idx = 0

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
            f'  - Decoded density mode: {self.decoded_density_mode}',
            f'  - Scalar condition presence: {self.scalar_condition_presence}',
            '  - Density statistics relax probability: '
            f'{self.density_statistics_relax_probability}',
            f'  - Density statistics relax std: {self.density_statistics_relax_std}',
        ]
        if self.batch_size_per_gpu_by_high_resolution is not None:
            lines.append(
                '  - Batch size per GPU by high resolution: '
                f'{self.batch_size_per_gpu_by_high_resolution}'
            )
        if self.resolution_sampling_weights is not None:
            lines.append(
                '  - Resolution sampling weights: '
                f'{self.resolution_sampling_weights}'
            )
        return '\n'.join(lines)

    def _relax_density_statistics(
        self,
        density_statistics: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        if density_statistics is None:
            return None, None
        density_statistics = density_statistics.float()
        if density_statistics.ndim != 2 or density_statistics.shape[1] != 3:
            raise ValueError(
                f'density_statistics must have shape (batch, 3), got '
                f'{tuple(density_statistics.shape)}'
            )
        if not torch.isfinite(density_statistics).all():
            raise ValueError('density_statistics contains non-finite values')
        if torch.any(density_statistics[:, 0] > density_statistics[:, 1]) or torch.any(
            density_statistics[:, 1] > density_statistics[:, 2]
        ):
            raise ValueError('density_statistics must be ordered as min <= median <= max')

        relax_mask = torch.rand(
            (density_statistics.shape[0], 2),
            device=density_statistics.device,
        ) < self.density_statistics_relax_probability
        slack = (
            torch.randn(
                (density_statistics.shape[0], 2),
                device=density_statistics.device,
                dtype=torch.float32,
            ).abs()
            * self.density_statistics_relax_std
            * relax_mask.float()
        )
        relaxed = density_statistics.clone()
        relaxed[:, 0] -= slack[:, 0]
        relaxed[:, 2] += slack[:, 1]
        return relaxed, {
            'minimum_mask': relax_mask[:, 0],
            'maximum_mask': relax_mask[:, 1],
            'minimum_slack': slack[:, 0],
            'maximum_slack': slack[:, 1],
        }

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

    def _resolve_decoded_density(
        self,
        decoded: sp.SparseTensor,
        density_cond: Optional[sp.SparseTensor],
    ) -> Tuple[sp.SparseTensor, Optional[sp.SparseTensor]]:
        if self.decoded_density_mode == 'none':
            return decoded, density_cond
        if decoded.feats.shape[1] != 3:
            raise ValueError(
                f"decoded_density_mode='{self.decoded_density_mode}' requires exactly "
                f'3 decoder output channels, got {decoded.feats.shape[1]}'
            )
        if self.decoded_density_mode == 'condition':
            if density_cond is not None:
                raise ValueError(
                    "decoded_density_mode='condition' cannot also receive an external density condition."
                )
            return (
                decoded.replace(decoded.feats[:, :2]),
                decoded.replace(decoded.feats[:, 2:3]),
            )
        return decoded, density_cond

    def _replace_dropped_scalar_condition(
        self,
        name: str,
        tensor: Optional[sp.SparseTensor],
        sample_drop: Optional[torch.Tensor],
    ) -> Optional[sp.SparseTensor]:
        if tensor is None or sample_drop is None or not sample_drop.any():
            return tensor
        feats = tensor.feats.clone()
        drop_value = (
            0.0
            if self.scalar_condition_presence and name in ('density', 'elongation')
            else self.condition_drop_values[name]
        )
        feats[sample_drop[tensor.coords[:, 0].long()]] = drop_value
        return tensor.replace(feats)

    def _append_scalar_presence(
        self,
        tensor: Optional[sp.SparseTensor],
        sample_drop: Optional[torch.Tensor],
    ) -> Optional[sp.SparseTensor]:
        if tensor is None or not self.scalar_condition_presence:
            return tensor
        presence = torch.ones(
            (tensor.feats.shape[0], 1),
            dtype=tensor.feats.dtype,
            device=tensor.feats.device,
        )
        if sample_drop is not None and sample_drop.any():
            presence[sample_drop[tensor.coords[:, 0].long()]] = 0
        return tensor.replace(torch.cat([tensor.feats, presence], dim=-1))

    def _build_latent_encoder_input(
        self,
        z_t: sp.SparseTensor,
        x_t: sp.SparseTensor,
        cond: sp.SparseTensor,
        density_cond: Optional[sp.SparseTensor],
        elongation_cond: Optional[sp.SparseTensor],
        condition_drop_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[sp.SparseTensor, Dict[str, sp.SparseTensor], Optional[torch.Tensor]]:
        condition_drop_masks = condition_drop_masks or {}
        density_input = self._append_scalar_presence(
            density_cond,
            condition_drop_masks.get('density'),
        )
        elongation_input = self._append_scalar_presence(
            elongation_cond,
            condition_drop_masks.get('elongation'),
        )

        latent_cond_missing = None
        if self.latent_self_conditioning_mode == 'input':
            latent_cond_input, latent_cond_missing = self._latent_to_field_support(z_t, x_t)
            enc_in = self._make_encoder_input(x_t, cond)
            enc_in = enc_in.replace(torch.cat([enc_in.feats, latent_cond_input.feats], dim=-1))
            extra_feats = []
            for name, tensor in (
                ('density', density_input),
                ('elongation', elongation_input),
            ):
                if tensor is None:
                    continue
                if not torch.equal(enc_in.coords, tensor.coords):
                    raise ValueError(f'{name}_cond coords must match encoder input coords')
                extra_feats.append(tensor.feats)
            if extra_feats:
                enc_in = enc_in.replace(torch.cat([enc_in.feats, *extra_feats], dim=-1))
            encoder_kwargs = {}
        elif self.latent_self_conditioning_mode == 'bottleneck':
            enc_in = self._make_encoder_input(x_t, cond, [density_input, elongation_input])
            encoder_kwargs = {'latent_cond': z_t}
        else:
            enc_in = self._make_encoder_input(x_t, cond, [density_input, elongation_input])
            encoder_kwargs = {}
        return enc_in, encoder_kwargs, latent_cond_missing

    def prepare_latent_encoder_input(
        self,
        z_t: sp.SparseTensor,
        decoded: sp.SparseTensor,
        cond: sp.SparseTensor,
        density_cond: Optional[sp.SparseTensor] = None,
        elongation_cond: Optional[sp.SparseTensor] = None,
        dropped_condition_names: Optional[Set[str]] = None,
        density_statistics: Optional[torch.Tensor] = None,
    ) -> Tuple[sp.SparseTensor, Dict[str, sp.SparseTensor], Optional[torch.Tensor]]:
        x_t, density_cond = self._resolve_decoded_density(decoded, density_cond)
        if not torch.equal(x_t.coords, cond.coords):
            raise ValueError(
                f'Decoded z_t coords must match cond coords, got {x_t.coords.shape} vs {cond.coords.shape}'
            )

        dropped_condition_names = set(dropped_condition_names or ())
        unknown = dropped_condition_names - {
            'density',
            'elongation',
            *DENSITY_STATISTIC_CONDITION_NAMES,
        }
        if unknown:
            raise ValueError(f'Unknown dropped scalar conditions: {sorted(unknown)}')
        drop_masks = {}
        for name, tensor in (
            ('density', density_cond),
            ('elongation', elongation_cond),
        ):
            if name not in dropped_condition_names:
                continue
            if tensor is None:
                raise ValueError(f'Cannot drop absent scalar condition {name}')
            sample_drop = torch.ones(
                tensor.shape[0],
                dtype=torch.bool,
                device=tensor.feats.device,
            )
            tensor = self._replace_dropped_scalar_condition(name, tensor, sample_drop)
            drop_masks[name] = sample_drop
            if name == 'density':
                density_cond = tensor
            else:
                elongation_cond = tensor
        enc_in, encoder_kwargs, latent_cond_missing = self._build_latent_encoder_input(
            z_t,
            x_t,
            cond,
            density_cond,
            elongation_cond,
            drop_masks,
        )
        model_uses_statistics = bool(getattr(
            self.models['encoder'],
            'density_statistics_conditioning',
            False,
        ))
        if model_uses_statistics:
            if density_statistics is None:
                raise ValueError('density_statistics must be provided by the dataset')
            density_statistics = density_statistics.float().clone()
            presence = torch.ones_like(density_statistics)
            for index, name in enumerate(DENSITY_STATISTIC_CONDITION_NAMES):
                if name in dropped_condition_names:
                    density_statistics[:, index] = 0
                    presence[:, index] = 0
            encoder_kwargs['density_statistics'] = density_statistics
            encoder_kwargs['density_statistics_presence'] = presence
        elif density_statistics is not None:
            raise ValueError(
                'Dataset provided density_statistics but encoder '
                'density_statistics_conditioning is disabled'
            )
        return enc_in, encoder_kwargs, latent_cond_missing

    def training_losses(
        self,
        z_0: sp.SparseTensor,
        cond: sp.SparseTensor,
        triangle_field_slat_cache: List[Dict[str, Any]] = None,
        triangle_field_slat_cache_path: List[str] = None,
        x_0: sp.SparseTensor = None,
        missing_low_parent_frac: torch.Tensor = None,
        density_cond: sp.SparseTensor = None,
        density_missing_parent_frac: torch.Tensor = None,
        elongation_cond: sp.SparseTensor = None,
        elongation_missing_parent_frac: torch.Tensor = None,
        density_statistics: torch.Tensor = None,
        force_field_drop: torch.Tensor = None,
        high_resolution: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        t = self.sample_t(z_0.shape[0]).to(z_0.feats.device).float()
        z_t, _ = self._diffuse_latent(z_0, t)
        decoded = self._decode_latents_with_cache(
            z_t,
            caches=triangle_field_slat_cache,
            cache_paths=triangle_field_slat_cache_path,
        )
        x_t, density_cond = self._resolve_decoded_density(decoded, density_cond)
        if not torch.equal(x_t.coords, cond.coords):
            raise ValueError(
                f'Decoded z_t coords must match cond coords, got {x_t.coords.shape} vs {cond.coords.shape}'
            )

        cond, conditioning_noise_level = self._augment_conditioning(
            cond,
            sample_noise_level=True,
            high_resolution=high_resolution,
            return_noise_level=True,
        )
        true_density_statistics = density_statistics
        density_statistics, density_statistics_relaxation = self._relax_density_statistics(
            density_statistics
        )
        relaxed_density_statistics = density_statistics
        cond, density_cond, elongation_cond, density_statistics, cond_drop = self._drop_conditions(
            cond,
            density_cond,
            elongation_cond,
            force_field_drop,
            density_statistics,
        )
        enc_in, encoder_kwargs, latent_cond_missing = self._build_latent_encoder_input(
            z_t,
            x_t,
            cond,
            density_cond,
            elongation_cond,
            cond_drop,
        )
        model_uses_statistics = bool(getattr(
            self.models['encoder'],
            'density_statistics_conditioning',
            False,
        ))
        if model_uses_statistics:
            if density_statistics is None:
                raise ValueError('density_statistics must be provided by the dataset')
            encoder_kwargs['density_statistics'] = density_statistics
            encoder_kwargs['density_statistics_presence'] = torch.stack([
                ~cond_drop[name]
                for name in DENSITY_STATISTIC_CONDITION_NAMES
            ], dim=-1)
        elif density_statistics is not None:
            raise ValueError(
                'Dataset provided density_statistics but encoder '
                'density_statistics_conditioning is disabled'
            )
        if getattr(
            self.models['encoder'],
            'conditioning_noise_conditioning',
            False,
        ):
            encoder_kwargs['conditioning_noise_level'] = conditioning_noise_level
        pred_z0 = self.training_models['encoder'](
            enc_in,
            t * 1000.0,
            sample_posterior=self.sample_posterior,
            resolution=high_resolution,
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
            'cond/drop_frac': cond_drop['all'].float().mean(),
            'cond/drop_all_frac': cond_drop['all'].float().mean(),
            'cond/drop_partial_frac': cond_drop['partial'].float().mean(),
            'cond/field_drop_frac': cond_drop['field'].float().mean(),
            'cond/noise_level_mean': conditioning_noise_level.mean(),
            'cond/noise_level_min': conditioning_noise_level.min(),
            'cond/noise_level_max': conditioning_noise_level.max(),
        }
        if 'density' in cond_drop:
            status['cond/density_drop_frac'] = cond_drop['density'].float().mean()
        if 'elongation' in cond_drop:
            status['cond/elongation_drop_frac'] = cond_drop['elongation'].float().mean()
        for name in DENSITY_STATISTIC_CONDITION_NAMES:
            if name in cond_drop:
                status[f'cond/{name}_drop_frac'] = cond_drop[name].float().mean()
        if true_density_statistics is not None:
            statistic_labels = ('minimum', 'median', 'maximum')
            for index, label in enumerate(statistic_labels):
                status[f'density_statistics/true_{label}_mean'] = (
                    true_density_statistics[:, index].float().mean()
                )
                status[f'density_statistics/conditioned_{label}_mean'] = (
                    relaxed_density_statistics[:, index].float().mean()
                )
        if density_statistics_relaxation is not None:
            for label in ('minimum', 'maximum'):
                status[f'density_statistics/{label}_relax_frac'] = (
                    density_statistics_relaxation[f'{label}_mask'].float().mean()
                )
                status[f'density_statistics/{label}_slack_mean'] = (
                    density_statistics_relaxation[f'{label}_slack'].mean()
                )
        if missing_low_parent_frac is not None:
            missing_low_parent_frac = missing_low_parent_frac.float()
            status['cond/missing_low_parent_frac'] = missing_low_parent_frac.mean()
            status['cond/missing_low_parent_frac_max'] = missing_low_parent_frac.max()
        if latent_cond_missing is not None:
            status['latent_cond/missing_parent_frac'] = latent_cond_missing.mean()
            status['latent_cond/missing_parent_frac_max'] = latent_cond_missing.max()
        if density_missing_parent_frac is not None:
            density_missing_parent_frac = density_missing_parent_frac.float()
            status['density_cond/missing_parent_frac'] = density_missing_parent_frac.mean()
            status['density_cond/missing_parent_frac_max'] = density_missing_parent_frac.max()
        if elongation_missing_parent_frac is not None:
            elongation_missing_parent_frac = elongation_missing_parent_frac.float()
            status['elongation_cond/missing_parent_frac'] = elongation_missing_parent_frac.mean()
            status['elongation_cond/missing_parent_frac_max'] = elongation_missing_parent_frac.max()

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
            decoded = self._decode_latents_with_cache(
                z_t,
                caches=args.get('triangle_field_slat_cache', None),
                cache_paths=args.get('triangle_field_slat_cache_path', None),
            )
            snapshot_cond, conditioning_noise_level = self._augment_conditioning(
                args['cond'],
                sample_noise_level=True,
                high_resolution=args.get('high_resolution', None),
                return_noise_level=True,
            )
            enc_in, encoder_kwargs, _ = self.prepare_latent_encoder_input(
                z_t,
                decoded,
                snapshot_cond,
                args.get('density_cond', None),
                args.get('elongation_cond', None),
                density_statistics=args.get('density_statistics', None),
            )
            if getattr(
                self.models['encoder'],
                'conditioning_noise_conditioning',
                False,
            ):
                encoder_kwargs['conditioning_noise_level'] = conditioning_noise_level
            pred_z0 = self.models['encoder'](
                enc_in,
                t * 1000.0,
                sample_posterior=False,
                resolution=args.get('high_resolution', None),
                **encoder_kwargs,
            )
            y = self._decode_latents_with_cache(
                pred_z0,
                caches=args.get('triangle_field_slat_cache', None),
                cache_paths=args.get('triangle_field_slat_cache_path', None),
            )

            resolution = args.get('high_resolution', None)
            gt_vis = self.dataset.visualize_sample({'target': gt, 'high_resolution': resolution})
            cond_vis = self.dataset.visualize_sample({'target': snapshot_cond, 'high_resolution': resolution})
            noisy_vis = self.dataset.visualize_sample({'target': decoded, 'high_resolution': resolution})
            pred_vis = self.dataset.visualize_sample({'target': y, 'high_resolution': resolution})
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
