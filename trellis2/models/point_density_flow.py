"""Hunyuan point-set architecture adapted for surface-density flow matching."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .hunyuan_external.models.autoencoders.attention_blocks import (
    CrossAttentionDecoder,
    FourierEmbedder,
    PointCrossAttentionEncoder,
    fps,
)
from .hunyuan_external.models.denoisers.hunyuan3ddit import (
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


class HunyuanPointDensityFlowModel(nn.Module):
    """Predict clean surface density from noisy point-associated density."""

    def __init__(
        self,
        *,
        context_points: int = 81920,
        num_latents: int = 4096,
        width: int = 1024,
        heads: int = 16,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        time_embed_dim: int = 256,
        time_factor: float = 1000.0,
        query_chunk_size: int = 4096,
        training_query_chunk_size: Optional[int] = None,
        use_query_normals: bool = True,
    ):
        super().__init__()
        if context_points % num_latents != 0:
            raise ValueError('context_points must be divisible by num_latents')
        if width % heads != 0:
            raise ValueError('width must be divisible by heads')

        self.context_points = int(context_points)
        self.num_latents = int(num_latents)
        self.time_embed_dim = int(time_embed_dim)
        self.time_factor = float(time_factor)
        self.query_chunk_size = int(query_chunk_size)
        self.use_query_normals = bool(use_query_normals)
        self.training_query_chunk_size = (
            None
            if training_query_chunk_size is None
            else int(training_query_chunk_size)
        )

        self.fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs,
            include_pi=include_pi,
        )
        # Hunyuan's cross-attention encoder is retained; its ordinary latent
        # Transformer is disabled and replaced by timestep-modulated DiT blocks.
        self.point_encoder = PointCrossAttentionEncoder(
            num_latents=self.num_latents,
            downsample_ratio=self.context_points // self.num_latents,
            pc_size=self.context_points,
            pc_sharpedge_size=0,
            fourier_embedder=self.fourier_embedder,
            point_feats=4,
            width=width,
            heads=heads,
            layers=0,
            qkv_bias=qkv_bias,
            use_ln_post=False,
            qk_norm=qk_norm,
        )
        self.time_in = MLPEmbedder(self.time_embed_dim, width)
        self.dit_blocks = nn.ModuleList([
            SingleStreamBlock(
                hidden_size=width,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        self.query_decoder = CrossAttentionDecoder(
            num_latents=self.num_latents,
            out_channels=1,
            fourier_embedder=self.fourier_embedder,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type='regression',
        )
        query_feature_dim = self.fourier_embedder.out_dim
        if self.use_query_normals:
            query_feature_dim += 3
        self.query_decoder.query_proj = nn.Linear(
            query_feature_dim,
            width,
        )

    def select_anchor_indices(
        self,
        points: torch.Tensor,
        *,
        random_start: bool,
    ) -> torch.Tensor:
        """Select Hunyuan FPS anchors and return per-example point indices."""
        batch_size, num_points, _ = points.shape
        if num_points != self.context_points:
            raise ValueError(
                f'Expected {self.context_points} context points, got {num_points}'
            )
        flat_points = points.reshape(batch_size * num_points, 3)
        batch = torch.arange(batch_size, device=points.device).repeat_interleave(num_points)
        flat_indices = fps(
            flat_points,
            batch=batch,
            ratio=self.num_latents / num_points,
            random_start=random_start,
            batch_size=batch_size,
        )
        if flat_indices.numel() != batch_size * self.num_latents:
            raise RuntimeError(
                'FPS returned an unexpected number of anchors: '
                f'{flat_indices.numel()} instead of {batch_size * self.num_latents}'
            )
        offsets = torch.arange(batch_size, device=points.device)[:, None] * num_points
        return flat_indices.reshape(batch_size, self.num_latents) - offsets

    @staticmethod
    def _gather_points(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_indices = indices[..., None].expand(-1, -1, values.shape[-1])
        return torch.gather(values, dim=1, index=gather_indices)

    def encode(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        density_t: torch.Tensor,
        timestep: torch.Tensor,
        anchor_indices: Optional[torch.Tensor] = None,
        anchor_random_start: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_indices is None:
            random_start = (
                self.training
                if anchor_random_start is None
                else bool(anchor_random_start)
            )
            anchor_indices = self.select_anchor_indices(
                points,
                random_start=random_start,
            )

        point_features = torch.cat([normals, density_t], dim=-1)
        anchor_points = self._gather_points(points, anchor_indices)
        anchor_features = self._gather_points(point_features, anchor_indices)

        data = torch.cat([self.fourier_embedder(points), point_features], dim=-1)
        query = torch.cat([self.fourier_embedder(anchor_points), anchor_features], dim=-1)
        data = self.point_encoder.input_proj(data)
        query = self.point_encoder.input_proj(query)
        latents = self.point_encoder.cross_attn(query, data)

        time_vec = self.time_in(
            timestep_embedding(
                timestep,
                self.time_embed_dim,
                time_factor=self.time_factor,
            ).to(dtype=latents.dtype)
        )
        for block in self.dit_blocks:
            latents = block(latents, vec=time_vec, pe=None)
        return latents, anchor_indices

    def decode(
        self,
        latents: torch.Tensor,
        query_points: torch.Tensor,
        query_normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_query_normals and query_normals is None:
            raise ValueError('query_normals are required when use_query_normals=True')
        chunk_size = (
            self.training_query_chunk_size if self.training else self.query_chunk_size
        )
        if chunk_size is None:
            chunk_size = query_points.shape[1]
        predictions = []
        for start in range(0, query_points.shape[1], chunk_size):
            end = min(start + chunk_size, query_points.shape[1])
            query_features = self.fourier_embedder(query_points[:, start:end])
            if self.use_query_normals:
                query_features = torch.cat([
                    query_features,
                    query_normals[:, start:end],
                ], dim=-1)
            query_embeddings = self.query_decoder.query_proj(query_features)
            predictions.append(self.query_decoder(
                query_embeddings=query_embeddings,
                latents=latents,
            ))
        return torch.cat(predictions, dim=1)

    def forward(
        self,
        density_t: torch.Tensor,
        timestep: torch.Tensor,
        context_points: torch.Tensor,
        context_normals: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        query_normals: Optional[torch.Tensor] = None,
        anchor_indices: Optional[torch.Tensor] = None,
        anchor_random_start: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        latents, anchor_indices = self.encode(
            context_points,
            context_normals,
            density_t,
            timestep,
            anchor_indices=anchor_indices,
            anchor_random_start=anchor_random_start,
        )
        output = {
            'context': self.decode(
                latents,
                context_points,
                context_normals if self.use_query_normals else None,
            ),
            'anchor_indices': anchor_indices,
        }
        if query_points is not None:
            if self.use_query_normals and query_normals is None:
                raise ValueError('query_normals are required with query_points')
            output['query'] = self.decode(
                latents,
                query_points,
                query_normals if self.use_query_normals else None,
            )
        return output
