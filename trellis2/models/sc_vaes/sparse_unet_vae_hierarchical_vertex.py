"""Triangle-field VAE decoder with an explicit sparse vertex-token hierarchy.

The triangle-field reconstruction trunk is the normal SparseUnetVaeDecoder.
The additional branch starts from every active latent voxel and, at each x2
level, performs

    vertex self-attention -> vertex-to-latent cross-attention -> MLP

before producing eight child features.  Child features are intersected with
the supplied triangle-field support, fused with the corresponding VAE decoder
features, locally refined, classified with one scalar occupancy logit, and
pruned before the next hierarchy level.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from ...modules.sparse.transformer import SparseTransformerCrossBlock
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from .sparse_unet_vae import (
    SparseConvNeXtBlock3d,
    SparseUnetVaeDecoder,
)


class SparseUnetVaeHierarchicalVertexDecoder(SparseUnetVaeDecoder):
    """Supplied-support VAE decoder with hierarchical vertex child tokens."""

    def __init__(
        self,
        *args,
        pred_vertex_subdiv: bool = True,
        vertex_feature_channels: int = 128,
        vertex_num_heads: int = 4,
        vertex_mlp_ratio: float = 4.0,
        vertex_local_blocks: int = 1,
        vertex_use_checkpoint: bool = True,
        vertex_qk_rms_norm: bool = True,
        **kwargs,
    ):
        if not pred_vertex_subdiv:
            raise ValueError(
                'SparseUnetVaeHierarchicalVertexDecoder is a vertex decoder and '
                'requires pred_vertex_subdiv=True.'
            )
        # Do not construct the legacy parent->8-logit heads.  This class emits
        # explicit child tokens with one occupancy logit per child instead.
        super().__init__(*args, pred_vertex_subdiv=False, **kwargs)
        if self.pred_subdiv:
            raise ValueError(
                'Hierarchical vertex prediction requires pred_subdiv=False and '
                'the supplied triangle-field sparse support.'
            )
        if len(self.num_blocks) < 2:
            raise ValueError('The hierarchical vertex decoder requires an upsampling stage.')
        if vertex_feature_channels <= 0:
            raise ValueError('vertex_feature_channels must be positive.')
        if vertex_num_heads <= 0 or vertex_feature_channels % vertex_num_heads != 0:
            raise ValueError(
                'vertex_num_heads must be positive and divide vertex_feature_channels.'
            )
        if vertex_local_blocks < 0:
            raise ValueError('vertex_local_blocks must be non-negative.')

        self.pred_vertex_subdiv = True
        self.vertex_logits_are_child_tokens = True
        self.vertex_feature_channels = int(vertex_feature_channels)
        self.vertex_num_stages = len(self.num_blocks) - 1
        self.vertex_inference_threshold = 0.5

        latent_channels = self.from_latent.in_features
        vertex_channels = self.vertex_feature_channels
        self.vertex_latent_norm = LayerNorm32(latent_channels, eps=1e-6)
        self.vertex_latent_projection = sp.SparseLinear(latent_channels, vertex_channels)
        self.vertex_position_mlp = nn.Sequential(
            nn.Linear(3, vertex_channels),
            nn.SiLU(),
            nn.Linear(vertex_channels, vertex_channels),
        )
        self.vertex_initial_projection = sp.SparseLinear(
            self.model_channels[0], vertex_channels
        )
        self.vertex_attention_blocks = nn.ModuleList([
            SparseTransformerCrossBlock(
                vertex_channels,
                vertex_channels,
                num_heads=vertex_num_heads,
                mlp_ratio=vertex_mlp_ratio,
                attn_mode='full',
                use_checkpoint=vertex_use_checkpoint,
                qk_rms_norm=vertex_qk_rms_norm,
                qk_rms_norm_cross=vertex_qk_rms_norm,
                ln_affine=True,
            )
            for _ in range(self.vertex_num_stages)
        ])
        self.vertex_child_feature_projections = nn.ModuleList([
            sp.SparseLinear(vertex_channels, 8 * vertex_channels)
            for _ in range(self.vertex_num_stages)
        ])
        self.vertex_vae_feature_projections = nn.ModuleList([
            sp.SparseLinear(self.model_channels[i + 1], vertex_channels)
            for i in range(self.vertex_num_stages)
        ])
        self.vertex_child_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * vertex_channels, vertex_channels),
                nn.SiLU(),
                nn.Linear(vertex_channels, vertex_channels),
            )
            for _ in range(self.vertex_num_stages)
        ])
        self.vertex_local_refiners = nn.ModuleList([
            nn.ModuleList([
                SparseConvNeXtBlock3d(
                    vertex_channels,
                    use_checkpoint=vertex_use_checkpoint,
                )
                for _ in range(vertex_local_blocks)
            ])
            for _ in range(self.vertex_num_stages)
        ])
        self.vertex_occupancy_heads = nn.ModuleList([
            sp.SparseLinear(vertex_channels, 1)
            for _ in range(self.vertex_num_stages)
        ])

        # The base weighted-BCE checkpoint contains the legacy heads.  The
        # trainer uses these narrowly scoped prefixes when loading it as a
        # finetuning initialization.
        self.finetune_allowed_missing_prefixes = (
            'vertex_latent_norm.',
            'vertex_latent_projection.',
            'vertex_position_mlp.',
            'vertex_initial_projection.',
            'vertex_attention_blocks.',
            'vertex_child_feature_projections.',
            'vertex_vae_feature_projections.',
            'vertex_child_fusions.',
            'vertex_local_refiners.',
            'vertex_occupancy_heads.',
        )
        self.finetune_allowed_unexpected_prefixes = ('vertex_subdiv_heads.',)

        self._initialize_vertex_weights()
        if self.use_fp16:
            self._convert_vertex_modules(convert_module_to_f16)

    def _initialize_vertex_weights(self) -> None:
        """Initialize new projections while preserving zeroed local residuals."""
        explicit_modules = [
            self.vertex_latent_projection,
            self.vertex_position_mlp,
            self.vertex_initial_projection,
            self.vertex_attention_blocks,
            self.vertex_child_feature_projections,
            self.vertex_vae_feature_projections,
            self.vertex_child_fusions,
            self.vertex_occupancy_heads,
        ]

        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in explicit_modules:
            module.apply(init_linear)

    def _convert_vertex_modules(self, converter) -> None:
        for name in (
            'vertex_position_mlp',
            'vertex_initial_projection',
            'vertex_attention_blocks',
            'vertex_child_feature_projections',
            'vertex_vae_feature_projections',
            'vertex_child_fusions',
            'vertex_local_refiners',
            'vertex_occupancy_heads',
        ):
            if hasattr(self, name):
                getattr(self, name).apply(converter)

    def convert_to_fp16(self) -> None:
        # Avoid the legacy decoder's assumption that vertex_subdiv_heads exists.
        self.blocks.apply(convert_module_to_f16)
        self._convert_vertex_modules(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.blocks.apply(convert_module_to_f32)
        self._convert_vertex_modules(convert_module_to_f32)

    @staticmethod
    def _voxel_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
        coords = coords.long()
        return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]

    @staticmethod
    def _match_optional(
        query_keys: torch.Tensor,
        source_keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return source rows and a validity mask for exact key matches."""
        if len(source_keys) == 0:
            return torch.zeros_like(query_keys), torch.zeros_like(query_keys, dtype=torch.bool)
        sorted_keys, order = torch.sort(source_keys)
        if len(sorted_keys) > 1 and torch.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ValueError('Vertex token coordinate set contains duplicates.')
        positions = torch.searchsorted(sorted_keys, query_keys)
        valid = positions < len(sorted_keys)
        safe_positions = positions.clamp(max=len(sorted_keys) - 1)
        valid &= sorted_keys[safe_positions] == query_keys
        return order[safe_positions], valid

    @staticmethod
    def _row_select(x: sp.SparseTensor, mask: torch.Tensor) -> sp.SparseTensor:
        mask = mask.reshape(-1).bool()
        if len(mask) != len(x.feats):
            raise ValueError('Vertex row-selection mask has the wrong length.')
        if not mask.any():
            raise RuntimeError('Vertex row selection would remove every token.')
        return sp.SparseTensor(
            x.feats[mask],
            x.coords[mask],
            torch.Size([x.shape[0], x.feats.shape[1]]),
        )

    def _validate_resolutions(
        self,
        resolutions: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if resolutions is None:
            raise ValueError(
                'Hierarchical vertex decoding requires the final resolution of '
                'every batch sample.'
            )
        resolutions = torch.as_tensor(resolutions, device=device).reshape(-1).long()
        if len(resolutions) != batch_size:
            raise ValueError(
                f'Expected {batch_size} final resolutions, got {len(resolutions)}.'
            )
        divisor = 2 ** self.vertex_num_stages
        if torch.any(resolutions <= 0) or torch.any(resolutions % divisor != 0):
            raise ValueError(
                f'Every final resolution must be positive and divisible by {divisor}.'
            )
        return resolutions

    def _position_embedding(
        self,
        coords: torch.Tensor,
        per_sample_resolutions: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_ids = coords[:, 0].long()
        denom = per_sample_resolutions[batch_ids].float().unsqueeze(1)
        xyz = 2.0 * ((coords[:, 1:4].float() + 0.5) / denom) - 1.0
        return self.vertex_position_mlp(xyz.to(dtype=dtype))

    def _add_position(
        self,
        x: sp.SparseTensor,
        per_sample_resolutions: torch.Tensor,
    ) -> sp.SparseTensor:
        return x.replace(
            x.feats + self._position_embedding(
                x.coords, per_sample_resolutions, x.feats.dtype
            )
        )

    def _project_latent_context(
        self,
        latent: sp.SparseTensor,
        final_resolutions: torch.Tensor,
    ) -> sp.SparseTensor:
        latent_resolutions = final_resolutions // (2 ** self.vertex_num_stages)
        normalized = latent.replace(self.vertex_latent_norm(latent.feats))
        context = self.vertex_latent_projection(normalized)
        context = context.type(self.dtype)
        return self._add_position(context, latent_resolutions)

    def _make_child_candidates(
        self,
        stage_index: int,
        parent_tokens: sp.SparseTensor,
        child_vae_features: sp.SparseTensor,
        child_resolutions: torch.Tensor,
    ) -> sp.SparseTensor:
        """Expand parent features and retain children on supplied VAE support."""
        expanded = self.vertex_child_feature_projections[stage_index](
            parent_tokens
        ).feats.reshape(-1, 8, self.vertex_feature_channels)
        vae_projected = self.vertex_vae_feature_projections[stage_index](
            child_vae_features
        ).feats

        output_coords: List[torch.Tensor] = []
        propagated_features: List[torch.Tensor] = []
        matching_vae_features: List[torch.Tensor] = []
        for batch_index in range(parent_tokens.shape[0]):
            parent_slice = parent_tokens.layout[batch_index]
            child_slice = child_vae_features.layout[batch_index]
            parent_xyz = parent_tokens.coords[parent_slice, 1:4].long()
            child_xyz = child_vae_features.coords[child_slice, 1:4].long()
            if len(parent_xyz) == 0 or len(child_xyz) == 0:
                continue
            child_resolution = int(child_resolutions[batch_index].item())
            parent_resolution = child_resolution // 2
            parent_keys = self._voxel_keys(parent_xyz, parent_resolution)
            support_parent_keys = self._voxel_keys(
                child_xyz // 2, parent_resolution
            )
            local_parent_rows, valid = self._match_optional(
                support_parent_keys, parent_keys
            )
            if not valid.any():
                continue
            child_rows = torch.arange(
                child_slice.start,
                child_slice.stop,
                device=child_xyz.device,
            )[valid]
            local_child_xyz = child_xyz[valid]
            remainder = local_child_xyz % 2
            child_index = (
                remainder[:, 0] + 2 * remainder[:, 1] + 4 * remainder[:, 2]
            )
            global_parent_rows = local_parent_rows[valid] + parent_slice.start
            output_coords.append(child_vae_features.coords[child_rows])
            propagated_features.append(expanded[global_parent_rows, child_index])
            matching_vae_features.append(vae_projected[child_rows])

        if not output_coords:
            raise RuntimeError(
                f'Vertex hierarchy stage {stage_index} produced no supported child tokens.'
            )
        coords = torch.cat(output_coords, dim=0)
        propagated = torch.cat(propagated_features, dim=0)
        vae_features = torch.cat(matching_vae_features, dim=0)
        fused = self.vertex_child_fusions[stage_index](
            torch.cat([propagated, vae_features], dim=1)
        )
        tokens = sp.SparseTensor(
            fused,
            coords,
            torch.Size([parent_tokens.shape[0], self.vertex_feature_channels]),
        )
        tokens = self._add_position(tokens, child_resolutions)
        for block in self.vertex_local_refiners[stage_index]:
            tokens = block(tokens)
        return tokens

    def _guide_mask(
        self,
        child_tokens: sp.SparseTensor,
        final_vertex_guide: sp.SparseTensor,
        final_resolutions: torch.Tensor,
        stage_index: int,
    ) -> torch.Tensor:
        """Select GT-positive child tokens for teacher-forced continuation."""
        occupancy = final_vertex_guide.feats.reshape(-1)
        if not torch.logical_or(occupancy == 0, occupancy == 1).all():
            raise ValueError('vertex_guide occupancy must be binary.')
        mask = torch.zeros(len(child_tokens.feats), device=child_tokens.device, dtype=torch.bool)
        child_batch_ids = child_tokens.coords[:, 0].long()
        guide_batch_ids = final_vertex_guide.coords[:, 0].long()
        num_stages = self.vertex_num_stages
        for batch_index in range(child_tokens.shape[0]):
            final_resolution = int(final_resolutions[batch_index].item())
            child_resolution = final_resolution // (2 ** (num_stages - stage_index - 1))
            candidate_rows = torch.nonzero(
                child_batch_ids.eq(batch_index), as_tuple=False
            ).flatten()
            positive_final = final_vertex_guide.coords[
                guide_batch_ids.eq(batch_index) & occupancy.bool(), 1:4
            ].long()
            if len(positive_final) == 0:
                raise ValueError(f'Batch sample {batch_index} has no GT vertex voxels.')
            positive_child = torch.unique(
                positive_final // (final_resolution // child_resolution), dim=0
            )
            if len(candidate_rows) == 0:
                raise ValueError(
                    f'Vertex stage {stage_index} has no candidates for batch sample '
                    f'{batch_index}, but GT children exist.'
                )
            candidate_xyz = child_tokens.coords[candidate_rows, 1:4].long()
            candidate_keys = self._voxel_keys(candidate_xyz, child_resolution)
            positive_keys = self._voxel_keys(positive_child, child_resolution)
            represented = torch.isin(positive_keys, candidate_keys)
            if not represented.all():
                raise ValueError(
                    f'Vertex stage {stage_index}, batch sample {batch_index}: '
                    f'{int((~represented).sum().item())} GT children are outside '
                    'the recursively supplied triangle support.'
                )
            mask[candidate_rows] = torch.isin(candidate_keys, positive_keys)
        return mask

    @staticmethod
    def _threshold_with_fallback(
        logits: sp.SparseTensor,
        threshold: float,
    ) -> torch.Tensor:
        """Threshold tokens while retaining one continuation per batch sample.

        The fallback token keeps sparse execution defined when a stage predicts
        no positives.  Its true path score is still retained by evaluation, so
        it cannot become a positive at the requested threshold unless all later
        decisions also pass and its own probability passes.
        """
        probability = torch.sigmoid(logits.feats.float()).reshape(-1)
        keep = probability >= threshold
        for sample_slice in logits.layout:
            if sample_slice.stop == sample_slice.start:
                continue
            if not keep[sample_slice].any():
                local_row = torch.argmax(probability[sample_slice])
                keep[sample_slice.start + local_row] = True
        return keep

    def forward(
        self,
        x: sp.SparseTensor,
        guide_subs: Optional[List[sp.SparseTensor]] = None,
        return_subs: bool = False,
        return_vertex: bool = False,
        return_vertex_features: bool = False,
        resolutions: Optional[torch.Tensor] = None,
        vertex_guide: Optional[sp.SparseTensor] = None,
        vertex_threshold: Optional[float] = None,
    ):
        if return_vertex_features and not return_vertex:
            raise ValueError(
                'return_vertex_features=True requires return_vertex=True.'
            )
        if not return_vertex:
            return super().forward(
                x,
                guide_subs=guide_subs,
                return_subs=return_subs,
                return_vertex=False,
            )
        if guide_subs is not None or return_subs:
            raise ValueError(
                'Hierarchical supplied-support vertex decoding does not use '
                'guide_subs or return_subs.'
            )
        threshold = (
            self.vertex_inference_threshold
            if vertex_threshold is None else float(vertex_threshold)
        )
        if not (0.0 < threshold < 1.0):
            raise ValueError('vertex_threshold must be in (0, 1).')
        final_resolutions = self._validate_resolutions(
            resolutions, x.shape[0], x.device
        )
        if self.training and vertex_guide is None:
            raise ValueError(
                'Training the hierarchical vertex decoder requires vertex_guide '
                'for GT teacher-forced pruning.'
            )

        latent = x
        context = self._project_latent_context(latent, final_resolutions)
        h = self.from_latent(x)
        h = h.type(self.dtype)

        # Process the latent-resolution VAE blocks before creating the initial
        # vertex candidate features.
        for block in self.blocks[0][:-1]:
            h = block(h)
        latent_resolutions = final_resolutions // (2 ** self.vertex_num_stages)
        vertex_tokens = self.vertex_initial_projection(h)
        vertex_tokens = self._add_position(vertex_tokens, latent_resolutions)

        vertex_logits = []
        for stage_index in range(self.vertex_num_stages):
            # Exact requested order: self-attention -> cross-attention -> MLP.
            vertex_tokens = self.vertex_attention_blocks[stage_index](
                vertex_tokens, context
            )

            # Advance the unchanged VAE trunk by one supplied-support level and
            # finish its local processing at the child resolution.
            upsample_block = self.blocks[stage_index][-1]
            h = upsample_block(h)
            next_blocks = self.blocks[stage_index + 1]
            normal_blocks = (
                next_blocks[:-1]
                if stage_index + 1 < self.vertex_num_stages
                else next_blocks
            )
            for block in normal_blocks:
                h = block(h)

            child_resolutions = final_resolutions // (
                2 ** (self.vertex_num_stages - stage_index - 1)
            )
            child_tokens = self._make_child_candidates(
                stage_index,
                vertex_tokens,
                h,
                child_resolutions,
            )
            logits = self.vertex_occupancy_heads[stage_index](child_tokens)
            vertex_logits.append(logits)

            if stage_index + 1 < self.vertex_num_stages:
                if vertex_guide is not None:
                    keep = self._guide_mask(
                        child_tokens,
                        vertex_guide,
                        final_resolutions,
                        stage_index,
                    )
                else:
                    keep = self._threshold_with_fallback(logits, threshold)
                vertex_tokens = self._row_select(child_tokens, keep)

        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        y = self.output_layer(h)
        if return_vertex_features:
            return y, vertex_logits, child_tokens
        return y, vertex_logits
