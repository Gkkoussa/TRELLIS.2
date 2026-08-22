"""TRELLIS flow-matching trainer for Hunyuan point-density prediction."""

import copy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ...datasets.point_density_mesh import render_point_density
from ...pipelines.samplers.flow_euler import FlowEulerSampler
from ...utils.data_utils import ResumableSampler, cycle
from .flow_matching import FlowMatchingTrainer


class PointDensityFlowEulerSampler(FlowEulerSampler):
    """Euler sampler for a model whose native output is predicted clean density."""

    def __init__(
        self,
        sigma_min: float,
        *,
        x0_target_mean: Optional[float] = None,
        x0_target_std: Optional[float] = None,
        x0_clamp_max: Optional[float] = None,
    ):
        super().__init__(sigma_min)
        self.x0_target_mean = x0_target_mean
        self.x0_target_std = x0_target_std
        self.x0_clamp_max = x0_clamp_max
        self.constraint_stats = []

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        timestep = torch.full(
            (x_t.shape[0],),
            float(t),
            device=x_t.device,
            dtype=torch.float32,
        )
        output = model(x_t, timestep, **cond, **kwargs)
        return output['context']

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_x_0 = self._inference_model(model, x_t, t, cond, **kwargs)
        if any(value is not None for value in (
            self.x0_target_mean,
            self.x0_target_std,
            self.x0_clamp_max,
        )):
            pred_x_0 = pred_x_0.float()
            reduce_dims = tuple(range(1, pred_x_0.ndim))
            pre_shift_mean = pred_x_0.mean(dim=reduce_dims, keepdim=True)
            if self.x0_target_mean is not None:
                pred_x_0 = pred_x_0 + self.x0_target_mean - pre_shift_mean
            shifted_mean = pred_x_0.mean(dim=reduce_dims, keepdim=True)
            pre_scale_std = pred_x_0.std(
                dim=reduce_dims, correction=0, keepdim=True
            )
            if self.x0_target_std is not None:
                pred_x_0 = shifted_mean + (
                    (pred_x_0 - shifted_mean)
                    * (self.x0_target_std / pre_scale_std.clamp_min(1e-8))
                )
            scaled_mean = pred_x_0.mean(dim=reduce_dims, keepdim=True)
            scaled_std = pred_x_0.std(
                dim=reduce_dims, correction=0, keepdim=True
            )
            if self.x0_clamp_max is not None:
                clipped_fraction = (pred_x_0 > self.x0_clamp_max).float().mean(
                    dim=reduce_dims, keepdim=True
                )
                pred_x_0 = pred_x_0.clamp(max=self.x0_clamp_max)
            else:
                clipped_fraction = torch.zeros_like(shifted_mean)
            self.constraint_stats.append({
                'timestep': float(t),
                'pre_shift_mean': pre_shift_mean.detach(),
                'shifted_mean': shifted_mean.detach(),
                'pre_scale_std': pre_scale_std.detach(),
                'scaled_mean': scaled_mean.detach(),
                'scaled_std': scaled_std.detach(),
                'post_clamp_mean': pred_x_0.mean(
                    dim=reduce_dims, keepdim=True
                ).detach(),
                'post_clamp_std': pred_x_0.std(
                    dim=reduce_dims, correction=0, keepdim=True
                ).detach(),
                'clipped_fraction': clipped_fraction.detach(),
            })
        pred_eps = self._xstart_to_eps(x_t, t, pred_x_0)
        pred_v = self._xstart_to_pred(x_t, t, pred_x_0)
        return pred_x_0, pred_eps, pred_v


class PointDensityFlowTrainer(FlowMatchingTrainer):
    """Jointly optimize flow velocity on P and clean density reconstruction on Q."""

    def __init__(
        self,
        models,
        dataset,
        *,
        validation_dataset=None,
        query_loss_weight: float = 1.0,
        num_workers: int = 8,
        snapshot_num_samples: int = 8,
        snapshot_steps: int = 50,
        field_name: str = 'density',
        context_loss_type: str = 'velocity',
        **kwargs,
    ):
        self.validation_dataset = validation_dataset
        self.query_loss_weight = float(query_loss_weight)
        self.num_workers = min(int(num_workers), 8)
        self.snapshot_num_samples = int(snapshot_num_samples)
        self.snapshot_steps = int(snapshot_steps)
        self.field_name = str(field_name)
        if context_loss_type not in ('velocity', 'x0'):
            raise ValueError("context_loss_type must be 'velocity' or 'x0'")
        self.context_loss_type = context_loss_type
        super().__init__(models, dataset, **kwargs)

    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Query loss weight: {self.query_loss_weight}',
            f'  - Target field: {self.field_name}',
            f'  - Context loss type: {self.context_loss_type}',
            f'  - Snapshot samples: {self.snapshot_num_samples}',
            f'  - Snapshot Euler steps: {self.snapshot_steps}',
            f'  - Validation dataset: {self.validation_dataset is not None}',
        ]
        return '\n'.join(lines)

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def get_sampler(self, **kwargs):
        return PointDensityFlowEulerSampler(self.sigma_min)

    def training_losses(
        self,
        context_points: torch.Tensor,
        context_normals: torch.Tensor,
        context_density: torch.Tensor,
        query_points: torch.Tensor,
        query_density: torch.Tensor,
        query_normals: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        noise = torch.randn_like(context_density)
        timestep = self.sample_t(context_density.shape[0]).to(context_density.device).float()
        density_t = self.diffuse(context_density, timestep, noise=noise)
        output = self.training_models['denoiser'](
            density_t,
            timestep,
            context_points,
            context_normals,
            query_points=query_points,
            query_normals=query_normals,
        )

        if self.context_loss_type == 'velocity':
            scale = self.sigma_min + (1.0 - self.sigma_min) * timestep
            scale = scale[:, None, None]
            pred_velocity = (
                (1.0 - self.sigma_min) * density_t - output['context']
            ) / scale
            target_velocity = self.get_v(context_density, noise, timestep)
            context_loss = F.mse_loss(pred_velocity, target_velocity)
            context_loss_name = 'context_velocity_l2'
        else:
            context_loss = F.mse_loss(output['context'], context_density)
            context_loss_name = 'context_x0_l2'
        query_loss = F.mse_loss(output['query'], query_density)
        total_loss = context_loss + self.query_loss_weight * query_loss
        losses = {
            'loss': total_loss,
            context_loss_name: context_loss,
            'query_x0_l2': query_loss,
        }
        status = {
            'timestep_mean': timestep.mean(),
            'pred_context_mean': output['context'].mean(),
            'pred_query_mean': output['query'].mean(),
        }
        return losses, status

    def snapshot(self, *args, num_samples=None, **kwargs):
        if num_samples is None:
            num_samples = self.snapshot_num_samples
        return super().snapshot(*args, num_samples=num_samples, **kwargs)

    def snapshot_dataset(self, num_samples=None, batch_size=1):
        if num_samples is None:
            num_samples = self.snapshot_num_samples
        return super().snapshot_dataset(num_samples=num_samples, batch_size=batch_size)

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        steps: int = None,
        **kwargs,
    ) -> Dict:
        snapshot_dataset = self.validation_dataset or self.dataset
        snapshot_indices = [
            (self.rank * num_samples + index) % len(snapshot_dataset)
            for index in range(num_samples)
        ]
        dataloader = DataLoader(
            Subset(copy.deepcopy(snapshot_dataset), snapshot_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        iterator = iter(dataloader)
        sampler = self.get_sampler()
        model = self.models['denoiser']
        steps = self.snapshot_steps if steps is None else int(steps)
        images = {
            f'query_{self.field_name}_gt': [],
            f'query_{self.field_name}_pred': [],
            f'context_{self.field_name}_sample': [],
        }

        for start in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - start)
            data = next(iterator)
            data = {
                key: value[:current_batch].to(self.device, non_blocking=True)
                for key, value in data.items()
            }
            anchor_indices = model.select_anchor_indices(
                data['context_points'],
                random_start=False,
            )
            condition = {
                'context_points': data['context_points'],
                'context_normals': data['context_normals'],
                'anchor_indices': anchor_indices,
            }
            result = sampler.sample(
                model,
                noise=torch.randn_like(data['context_density']),
                cond=condition,
                steps=steps,
                verbose=verbose,
                tqdm_desc='Sampling surface density',
            )

            # Re-encode the completed context once at t=0 before arbitrary Q queries.
            final_output = model(
                result.samples,
                torch.zeros(current_batch, device=self.device),
                data['context_points'],
                data['context_normals'],
                query_points=data['query_points'],
                query_normals=data.get('query_normals'),
                anchor_indices=anchor_indices,
            )
            images[f'query_{self.field_name}_gt'].append(render_point_density(
                data['query_points'], data['query_density']
            ))
            images[f'query_{self.field_name}_pred'].append(render_point_density(
                data['query_points'], final_output['query']
            ))
            images[f'context_{self.field_name}_sample'].append(render_point_density(
                data['context_points'], result.samples
            ))

        return {
            key: {'value': torch.cat(value, dim=0), 'type': 'image'}
            for key, value in images.items()
        }
