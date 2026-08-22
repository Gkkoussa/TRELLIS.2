"""Hierarchical R512 vertex decoder with symmetric pairwise edge prediction."""

from typing import List, Optional

import torch
import torch.nn as nn

from ...modules import sparse as sp
from ...modules.utils import convert_module_to_f16
from .sparse_unet_vae_hierarchical_vertex import (
    SparseUnetVaeHierarchicalVertexDecoder,
)


class SparseUnetVaeHierarchicalVertexEdgeDecoder(
    SparseUnetVaeHierarchicalVertexDecoder
):
    """Score undirected QEM vertex pairs from final hierarchical token features."""

    def __init__(
        self,
        *args,
        pred_edge: bool = True,
        edge_hidden_channels: Optional[int] = None,
        **kwargs,
    ):
        if not pred_edge:
            raise ValueError(
                'SparseUnetVaeHierarchicalVertexEdgeDecoder requires pred_edge=True.'
            )
        super().__init__(*args, **kwargs)
        self.pred_edge = True
        hidden_channels = (
            self.vertex_feature_channels
            if edge_hidden_channels is None
            else int(edge_hidden_channels)
        )
        if hidden_channels <= 0:
            raise ValueError('edge_hidden_channels must be positive.')
        self.edge_hidden_channels = hidden_channels
        self.edge_connection_head = nn.Sequential(
            nn.Linear(2 * self.vertex_feature_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        for module in self.edge_connection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.use_fp16:
            self.edge_connection_head.apply(convert_module_to_f16)

        self.finetune_allowed_missing_prefixes = (
            *self.finetune_allowed_missing_prefixes,
            'edge_connection_head.',
        )

    def _convert_vertex_modules(self, converter) -> None:
        super()._convert_vertex_modules(converter)
        if hasattr(self, 'edge_connection_head'):
            self.edge_connection_head.apply(converter)

    def _score_edge_pairs(
        self,
        final_vertex_tokens: sp.SparseTensor,
        final_resolutions: torch.Tensor,
        edge_vertex_coords: List[torch.Tensor],
        edge_pairs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if len(edge_vertex_coords) != final_vertex_tokens.shape[0]:
            raise ValueError(
                'edge_vertex_coords must contain one coordinate tensor per sample.'
            )
        if len(edge_pairs) != final_vertex_tokens.shape[0]:
            raise ValueError('edge_pairs must contain one pair tensor per sample.')

        outputs = []
        for batch_index in range(final_vertex_tokens.shape[0]):
            vertex_coords = edge_vertex_coords[batch_index].to(
                device=final_vertex_tokens.device, dtype=torch.long
            )
            pairs = edge_pairs[batch_index].to(
                device=final_vertex_tokens.device, dtype=torch.long
            )
            if vertex_coords.ndim != 2 or vertex_coords.shape[1] != 3:
                raise ValueError(
                    f'edge_vertex_coords[{batch_index}] must have shape [N, 3].'
                )
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(f'edge_pairs[{batch_index}] must have shape [P, 2].')

            sample_slice = final_vertex_tokens.layout[batch_index]
            token_xyz = final_vertex_tokens.coords[sample_slice, 1:4].long()
            token_features = final_vertex_tokens.feats[sample_slice]
            if len(vertex_coords) == 0:
                if len(pairs) != 0:
                    raise ValueError(
                        f'edge_pairs[{batch_index}] is non-empty but its vertex set is empty.'
                    )
                empty_pair_features = token_features.new_empty(
                    (0, 2 * self.vertex_feature_channels)
                )
                # Executing the head on an empty tensor keeps a differentiable
                # zero dependency on its parameters for mixed-resolution DDP.
                directional = self.edge_connection_head(empty_pair_features)
                outputs.append((directional + directional).reshape(-1))
                continue

            resolution = int(final_resolutions[batch_index].item())
            if (vertex_coords < 0).any() or (vertex_coords >= resolution).any():
                raise ValueError(
                    f'edge_vertex_coords[{batch_index}] is outside R{resolution}.'
                )
            if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(vertex_coords)):
                raise ValueError(
                    f'edge_pairs[{batch_index}] contains an endpoint outside '
                    f'[0, {len(vertex_coords)}).'
                )

            token_keys = self._voxel_keys(token_xyz, resolution)
            vertex_keys = self._voxel_keys(vertex_coords, resolution)
            token_rows, valid = self._match_optional(vertex_keys, token_keys)
            if not valid.all():
                raise ValueError(
                    f'R{resolution} sample {batch_index}: '
                    f'{int((~valid).sum().item())} QEM vertices have no final token.'
                )
            vertex_features = token_features[token_rows]
            uv = torch.cat(
                [vertex_features[pairs[:, 0]], vertex_features[pairs[:, 1]]], dim=1
            )
            vu = torch.cat(
                [vertex_features[pairs[:, 1]], vertex_features[pairs[:, 0]]], dim=1
            )
            logits_uv = self.edge_connection_head(uv).reshape(-1)
            logits_vu = self.edge_connection_head(vu).reshape(-1)
            outputs.append(logits_uv + logits_vu)
        return outputs

    def forward(
        self,
        x: sp.SparseTensor,
        *args,
        return_vertex: bool = False,
        return_edge: bool = False,
        edge_vertex_coords: Optional[List[torch.Tensor]] = None,
        edge_pairs: Optional[List[torch.Tensor]] = None,
        resolutions: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not return_edge:
            return super().forward(
                x,
                *args,
                return_vertex=return_vertex,
                resolutions=resolutions,
                **kwargs,
            )
        if not return_vertex:
            raise ValueError('return_edge=True requires return_vertex=True.')
        if edge_vertex_coords is None or edge_pairs is None:
            raise ValueError(
                'return_edge=True requires edge_vertex_coords and edge_pairs.'
            )
        y, vertex_logits, final_vertex_tokens = super().forward(
            x,
            *args,
            return_vertex=True,
            return_vertex_features=True,
            resolutions=resolutions,
            **kwargs,
        )
        final_resolutions = self._validate_resolutions(
            resolutions, x.shape[0], x.device
        )
        edge_logits = self._score_edge_pairs(
            final_vertex_tokens,
            final_resolutions,
            edge_vertex_coords,
            edge_pairs,
        )
        return y, vertex_logits, edge_logits
