"""Hunyuan VecSet shape tokens with PyTorch3D farthest-point sampling."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .hunyuan_external.models.autoencoders.attention_blocks import (
    FourierEmbedder,
    PointCrossAttentionEncoder,
)


class HunyuanShapeTokenEncoder(nn.Module):
    """Compress surface points and normals into a fixed set of shape tokens."""

    def __init__(
        self,
        *,
        context_points: int = 16384,
        num_tokens: int = 1024,
        width: int = 1024,
        heads: int = 16,
        layers: int = 8,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_ln_post: bool = True,
    ):
        super().__init__()
        if context_points <= 0 or num_tokens <= 0:
            raise ValueError('context_points and num_tokens must be positive')
        if context_points < num_tokens:
            raise ValueError(
                f'context_points must be >= num_tokens, got {context_points} < {num_tokens}'
            )
        if context_points % num_tokens != 0:
            raise ValueError(
                'Hunyuan downsample_ratio requires context_points divisible by '
                f'num_tokens, got {context_points}/{num_tokens}'
            )
        if width % heads != 0:
            raise ValueError(f'width must be divisible by heads, got {width}/{heads}')

        self.context_points = int(context_points)
        self.num_tokens = int(num_tokens)
        self.width = int(width)
        self.fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs,
            include_pi=include_pi,
        )
        self.point_encoder = PointCrossAttentionEncoder(
            num_latents=self.num_tokens,
            downsample_ratio=self.context_points // self.num_tokens,
            pc_size=self.context_points,
            pc_sharpedge_size=0,
            fourier_embedder=self.fourier_embedder,
            point_feats=3,
            width=self.width,
            heads=heads,
            layers=layers,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            qk_norm=qk_norm,
        )

    @staticmethod
    def _fps(
        points: torch.Tensor,
        num_tokens: int,
        random_start_point: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from pytorch3d.ops import sample_farthest_points
        except ImportError as exc:
            raise ImportError(
                'Shape conditioning requires PyTorch3D in the trellis2 environment. '
                'Install a build compatible with the active PyTorch/CUDA versions.'
            ) from exc
        return sample_farthest_points(
            points.float().contiguous(),
            K=num_tokens,
            random_start_point=random_start_point,
        )

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_indices = indices[..., None].expand(-1, -1, values.shape[-1])
        return torch.gather(values, dim=1, index=gather_indices)

    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        *,
        random_start_point: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f'points must have shape [B, N, 3], got {tuple(points.shape)}')
        if normals.shape != points.shape:
            raise ValueError(
                f'normals must match points shape, got {tuple(normals.shape)} and '
                f'{tuple(points.shape)}'
            )
        if points.shape[1] != self.context_points:
            raise ValueError(
                f'Expected {self.context_points} shape points, got {points.shape[1]}'
            )
        if not torch.isfinite(points).all() or not torch.isfinite(normals).all():
            raise ValueError('Shape points and normals must be finite')

        random_start = self.training if random_start_point is None else bool(random_start_point)
        anchor_points, anchor_indices = self._fps(
            points,
            self.num_tokens,
            random_start,
        )
        anchor_normals = self._gather(normals, anchor_indices)

        data = torch.cat([self.fourier_embedder(points), normals], dim=-1)
        query = torch.cat([self.fourier_embedder(anchor_points), anchor_normals], dim=-1)
        projection_dtype = self.point_encoder.input_proj.weight.dtype
        data = data.to(dtype=projection_dtype)
        query = query.to(dtype=projection_dtype)
        data = self.point_encoder.input_proj(data)
        query = self.point_encoder.input_proj(query)
        tokens = self.point_encoder.cross_attn(query, data)
        if self.point_encoder.self_attn is not None:
            tokens = self.point_encoder.self_attn(tokens)
        if self.point_encoder.ln_post is not None:
            tokens = self.point_encoder.ln_post(tokens)
        return tokens, anchor_points
