"""R512 edge-field refinement for hierarchical vertex and edge prediction.

This experiment leaves the existing hierarchical vertex/edge decoder intact.
It adds one R512-only proposal/refinement pass:

    hierarchical R512 vertex proposal tokens (queries)
        -> cross-attend to all low predicted-d_tri R512 VAE voxels
        -> vertex self-attention
        -> vertex-to-latent cross-attention
        -> FFN
        -> refined vertex logits and the existing symmetric edge head

The edge-context keys/values are projected native VAE voxel features.  Vertex
queries remain the hierarchical child tokens, including their propagated
parent information and corresponding child-resolution VAE fusion.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from ...modules.sparse.attention import SparseMultiHeadAttention
from ...modules.sparse.transformer.blocks import SparseFeedForwardNet
from ...modules.utils import convert_module_to_f16
from .sparse_unet_vae_hierarchical_vertex import (
    SparseUnetVaeHierarchicalVertexDecoder,
)
from .sparse_unet_vae_hierarchical_vertex_edge import (
    SparseUnetVaeHierarchicalVertexEdgeDecoder,
)


class R512VertexEdgeLatentRefinementBlock(nn.Module):
    """Edge cross-attention -> vertex self-attention -> latent cross-attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = True,
        qk_rms_norm: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.edge_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.vertex_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.latent_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.mlp_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.edge_cross_attention = SparseMultiHeadAttention(
            channels,
            ctx_channels=channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qk_rms_norm=qk_rms_norm,
        )
        self.vertex_self_attention = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode="full",
            qk_rms_norm=qk_rms_norm,
        )
        self.latent_cross_attention = SparseMultiHeadAttention(
            channels,
            ctx_channels=channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def _forward(
        self,
        vertex_tokens: sp.SparseTensor,
        edge_tokens: sp.SparseTensor,
        latent_context: sp.SparseTensor,
    ) -> sp.SparseTensor:
        h = vertex_tokens.replace(self.edge_norm(vertex_tokens.feats))
        vertex_tokens = vertex_tokens + self.edge_cross_attention(h, edge_tokens)

        h = vertex_tokens.replace(self.vertex_norm(vertex_tokens.feats))
        vertex_tokens = vertex_tokens + self.vertex_self_attention(h)

        h = vertex_tokens.replace(self.latent_norm(vertex_tokens.feats))
        vertex_tokens = vertex_tokens + self.latent_cross_attention(
            h, latent_context
        )

        h = vertex_tokens.replace(self.mlp_norm(vertex_tokens.feats))
        return vertex_tokens + self.mlp(h)

    def forward(
        self,
        vertex_tokens: sp.SparseTensor,
        edge_tokens: sp.SparseTensor,
        latent_context: sp.SparseTensor,
    ) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                vertex_tokens,
                edge_tokens,
                latent_context,
                use_reentrant=False,
            )
        return self._forward(vertex_tokens, edge_tokens, latent_context)


class SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder(
    SparseUnetVaeHierarchicalVertexEdgeDecoder
):
    """Hierarchical vertex/edge decoder with one additional R512 refinement."""

    def __init__(
        self,
        *args,
        r512_edge_dtri_threshold: float = 0.175,
        r512_vertex_proposal_threshold: float = 0.1,
        r512_refine_num_heads: Optional[int] = None,
        r512_refine_mlp_ratio: float = 4.0,
        r512_refine_use_checkpoint: bool = True,
        r512_refine_qk_rms_norm: bool = True,
        **kwargs,
    ):
        inherited_vertex_heads = int(kwargs.get("vertex_num_heads", 4))
        super().__init__(*args, **kwargs)
        if not (0.0 <= r512_edge_dtri_threshold <= 1.0):
            raise ValueError("r512_edge_dtri_threshold must be in raw [0, 1] units.")
        if not (0.0 < r512_vertex_proposal_threshold < 1.0):
            raise ValueError("r512_vertex_proposal_threshold must be in (0, 1).")
        refine_heads = (
            inherited_vertex_heads
            if r512_refine_num_heads is None
            else int(r512_refine_num_heads)
        )
        if refine_heads <= 0 or self.vertex_feature_channels % refine_heads != 0:
            raise ValueError(
                "r512_refine_num_heads must divide vertex_feature_channels."
            )
        if r512_refine_mlp_ratio <= 0:
            raise ValueError("r512_refine_mlp_ratio must be positive.")

        self.r512_edge_dtri_threshold = float(r512_edge_dtri_threshold)
        self.r512_vertex_proposal_threshold = float(
            r512_vertex_proposal_threshold
        )
        self.r512_refine_num_heads = refine_heads
        self.has_r512_dtri_edge_refinement = True

        # Normalize every explicit 512-D child token independently before its
        # shared binary occupancy classifier.  These four norms correspond to
        # the R64, R128, R256, and preliminary R512 heads.
        self.vertex_occupancy_norms = nn.ModuleList([
            LayerNorm32(self.vertex_feature_channels, eps=1e-6)
            for _ in range(self.vertex_num_stages)
        ])

        # The native final VAE feature has model_channels[-1] channels (64 in
        # the current configuration).  Only low-d_tri rows are retained after
        # this projection and used as edge-context keys/values.
        self.r512_edge_feature_norm = LayerNorm32(
            self.model_channels[-1], eps=1e-6
        )
        self.r512_edge_feature_projection = sp.SparseLinear(
            self.model_channels[-1], self.vertex_feature_channels
        )
        self.r512_vertex_refinement = R512VertexEdgeLatentRefinementBlock(
            self.vertex_feature_channels,
            num_heads=refine_heads,
            mlp_ratio=float(r512_refine_mlp_ratio),
            use_checkpoint=bool(r512_refine_use_checkpoint),
            qk_rms_norm=bool(r512_refine_qk_rms_norm),
        )
        self.r512_refined_vertex_norm = LayerNorm32(
            self.vertex_feature_channels, eps=1e-6
        )
        self.r512_refined_vertex_head = sp.SparseLinear(
            self.vertex_feature_channels, 1
        )

        self._initialize_r512_refinement_weights()
        if self.use_fp16:
            self._convert_r512_refinement_modules(convert_module_to_f16)

        self.finetune_allowed_missing_prefixes = (
            *self.finetune_allowed_missing_prefixes,
            "vertex_occupancy_norms.",
            "r512_edge_feature_norm.",
            "r512_edge_feature_projection.",
            "r512_vertex_refinement.",
            "r512_refined_vertex_norm.",
            "r512_refined_vertex_head.",
        )
        self.last_r512_proposal_logits: Optional[sp.SparseTensor] = None
        self.last_r512_refinement_stats = {}

    def _initialize_r512_refinement_weights(self) -> None:
        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.r512_edge_feature_projection.apply(initialize_linear)
        self.r512_vertex_refinement.apply(initialize_linear)
        self.r512_refined_vertex_head.apply(initialize_linear)

    def initialize_refined_head_from_proposal(self) -> None:
        """Copy the loaded preliminary R512 classifier into the refined head."""
        self.r512_refined_vertex_head.load_state_dict(
            self.vertex_occupancy_heads[-1].state_dict()
        )

    def _convert_r512_refinement_modules(self, converter) -> None:
        for name in (
            "r512_edge_feature_projection",
            "r512_vertex_refinement",
            "r512_refined_vertex_head",
        ):
            if hasattr(self, name):
                getattr(self, name).apply(converter)

    def _convert_vertex_modules(self, converter) -> None:
        super()._convert_vertex_modules(converter)
        self._convert_r512_refinement_modules(converter)

    def _new_parameter_zero_dependency(self) -> torch.Tensor:
        modules = (
            self.r512_edge_feature_norm,
            self.r512_edge_feature_projection,
            self.r512_vertex_refinement,
            self.r512_refined_vertex_norm,
            self.r512_refined_vertex_head,
        )
        zero = next(self.parameters()).new_zeros(())
        for module in modules:
            for parameter in module.parameters():
                zero = zero + parameter.reshape(-1)[0] * 0.0
        return zero

    @staticmethod
    def _select_batch_samples(
        x: sp.SparseTensor,
        sample_mask: torch.Tensor,
    ) -> tuple[sp.SparseTensor, torch.Tensor]:
        """Extract selected samples and compact their sparse batch indices."""
        sample_mask = sample_mask.reshape(-1).bool().to(device=x.device)
        if len(sample_mask) != x.shape[0]:
            raise ValueError("Sparse sample-selection mask has the wrong length.")
        sample_indices = torch.nonzero(sample_mask, as_tuple=False).flatten()
        if len(sample_indices) == 0:
            raise RuntimeError("Sparse sample selection would remove every sample.")

        old_batch_ids = x.coords[:, 0].long()
        row_keep = sample_mask[old_batch_ids]
        remap = torch.full(
            (x.shape[0],), -1, device=x.device, dtype=torch.long
        )
        remap[sample_indices] = torch.arange(
            len(sample_indices), device=x.device, dtype=torch.long
        )
        coords = x.coords[row_keep].clone()
        coords[:, 0] = remap[old_batch_ids[row_keep]].to(dtype=coords.dtype)
        return (
            sp.SparseTensor(
                x.feats[row_keep],
                coords,
                torch.Size([len(sample_indices), x.feats.shape[1]]),
            ),
            sample_indices,
        )

    @staticmethod
    def _restore_and_merge_samples(
        selected: sp.SparseTensor,
        fallback: sp.SparseTensor,
        selected_sample_indices: torch.Tensor,
    ) -> sp.SparseTensor:
        """Restore compact batch IDs and replace those samples in fallback."""
        if selected.shape[0] != len(selected_sample_indices):
            raise ValueError("Selected sparse tensor has the wrong batch size.")
        if selected.feats.shape[1] != fallback.feats.shape[1]:
            raise ValueError("Selected and fallback sparse channels do not match.")

        selected_sample_indices = selected_sample_indices.to(
            device=fallback.device, dtype=torch.long
        )
        restored_coords = selected.coords.clone()
        restored_coords[:, 0] = selected_sample_indices[
            selected.coords[:, 0].long()
        ].to(dtype=restored_coords.dtype)
        selected_by_original_batch = {
            int(original_batch): compact_batch
            for compact_batch, original_batch in enumerate(
                selected_sample_indices.tolist()
            )
        }

        feature_parts = []
        coordinate_parts = []
        for batch_index in range(fallback.shape[0]):
            compact_batch = selected_by_original_batch.get(batch_index)
            if compact_batch is None:
                sample_slice = fallback.layout[batch_index]
                feature_parts.append(fallback.feats[sample_slice])
                coordinate_parts.append(fallback.coords[sample_slice])
            else:
                sample_slice = selected.layout[compact_batch]
                feature_parts.append(selected.feats[sample_slice])
                coordinate_parts.append(restored_coords[sample_slice])

        return sp.SparseTensor(
            torch.cat(feature_parts, dim=0),
            torch.cat(coordinate_parts, dim=0),
            torch.Size([fallback.shape[0], fallback.feats.shape[1]]),
        )

    @staticmethod
    def _mask_with_per_sample_fallback(
        values: torch.Tensor,
        layout: List[slice],
        keep: torch.Tensor,
        choose_lowest: bool,
    ) -> torch.Tensor:
        keep = keep.clone()
        for sample_slice in layout:
            if sample_slice.stop == sample_slice.start:
                continue
            if not keep[sample_slice].any():
                local_values = values[sample_slice]
                local_row = (
                    torch.argmin(local_values)
                    if choose_lowest
                    else torch.argmax(local_values)
                )
                keep[sample_slice.start + local_row] = True
        return keep

    def _build_r512_edge_tokens(
        self,
        normalized_h: sp.SparseTensor,
        y: sp.SparseTensor,
        final_resolutions: torch.Tensor,
    ) -> tuple[sp.SparseTensor, torch.Tensor]:
        # Decoder outputs use [-1, 1], while the public threshold is expressed
        # in the raw triangle-field [0, 1] convention.
        transformed_threshold = 2.0 * self.r512_edge_dtri_threshold - 1.0
        predicted_dtri = y.feats[:, 0].float()
        raw_keep = predicted_dtri <= transformed_threshold
        keep = self._mask_with_per_sample_fallback(
            predicted_dtri,
            y.layout,
            raw_keep,
            choose_lowest=True,
        )
        edge_input = normalized_h.replace(
            self.r512_edge_feature_norm(normalized_h.feats)
        )
        edge_tokens = self.r512_edge_feature_projection(edge_input)
        edge_tokens = self._row_select(edge_tokens, keep)
        edge_tokens = self._add_position(edge_tokens, final_resolutions)
        return edge_tokens, raw_keep

    def _refine_r512_vertices(
        self,
        child_tokens: sp.SparseTensor,
        proposal_logits: sp.SparseTensor,
        normalized_h: sp.SparseTensor,
        y: sp.SparseTensor,
        latent_context: sp.SparseTensor,
        final_resolutions: torch.Tensor,
        vertex_guide: Optional[sp.SparseTensor],
    ) -> tuple[sp.SparseTensor, sp.SparseTensor]:
        proposal_probability = torch.sigmoid(proposal_logits.feats.float()).reshape(-1)
        predicted_keep = proposal_probability >= self.r512_vertex_proposal_threshold
        query_keep = predicted_keep.clone()
        if self.training:
            if vertex_guide is None:
                raise ValueError("R512 refinement training requires vertex_guide.")
            gt_keep = self._guide_mask(
                child_tokens,
                vertex_guide,
                final_resolutions,
                self.vertex_num_stages - 1,
            )
            query_keep |= gt_keep
        else:
            query_keep = self._mask_with_per_sample_fallback(
                proposal_probability,
                proposal_logits.layout,
                query_keep,
                choose_lowest=False,
            )

        vertex_queries = self._row_select(child_tokens, query_keep)
        edge_tokens, raw_edge_keep = self._build_r512_edge_tokens(
            normalized_h, y, final_resolutions
        )
        refined_tokens = self.r512_vertex_refinement(
            vertex_queries,
            edge_tokens,
            latent_context,
        )
        normalized_refined_tokens = refined_tokens.replace(
            self.r512_refined_vertex_norm(refined_tokens.feats)
        )
        refined_logits = self.r512_refined_vertex_head(
            normalized_refined_tokens
        )

        # A numerical fallback keeps sparse attention defined when inference
        # proposes no vertices, but it must not become a real final prediction.
        if not self.training:
            query_predicted_keep = predicted_keep[query_keep]
            refined_logits = refined_logits.replace(
                torch.where(
                    query_predicted_keep[:, None],
                    refined_logits.feats,
                    torch.full_like(refined_logits.feats, -1.0e4),
                )
            )

        self.last_r512_refinement_stats = {
            "active_voxels": int(len(y.feats)),
            "low_dtri_voxels": int(raw_edge_keep.sum().item()),
            "edge_context_tokens": int(len(edge_tokens.feats)),
            "proposal_candidates": int(len(proposal_logits.feats)),
            "predicted_proposals": int(predicted_keep.sum().item()),
            "refinement_queries": int(len(refined_tokens.feats)),
        }
        return refined_logits, refined_tokens

    def forward(
        self,
        x: sp.SparseTensor,
        guide_subs: Optional[List[sp.SparseTensor]] = None,
        return_subs: bool = False,
        return_vertex: bool = False,
        return_vertex_features: bool = False,
        return_edge: bool = False,
        edge_vertex_coords: Optional[List[torch.Tensor]] = None,
        edge_pairs: Optional[List[torch.Tensor]] = None,
        resolutions: Optional[torch.Tensor] = None,
        vertex_guide: Optional[sp.SparseTensor] = None,
        vertex_threshold: Optional[float] = None,
    ):
        if return_vertex_features and not return_vertex:
            raise ValueError(
                "return_vertex_features=True requires return_vertex=True."
            )
        if return_edge and not return_vertex:
            raise ValueError("return_edge=True requires return_vertex=True.")
        if return_edge and return_vertex_features:
            raise ValueError(
                "return_edge and return_vertex_features cannot be requested together."
            )
        if not return_vertex:
            return SparseUnetVaeHierarchicalVertexDecoder.forward(
                self,
                x,
                guide_subs=guide_subs,
                return_subs=return_subs,
                return_vertex=False,
            )
        if guide_subs is not None or return_subs:
            raise ValueError(
                "R512 d_tri refinement uses supplied support, not subdivision guidance."
            )
        threshold = (
            self.vertex_inference_threshold
            if vertex_threshold is None
            else float(vertex_threshold)
        )
        if not (0.0 < threshold < 1.0):
            raise ValueError("vertex_threshold must be in (0, 1).")
        final_resolutions = self._validate_resolutions(
            resolutions, x.shape[0], x.device
        )
        if self.training and vertex_guide is None:
            raise ValueError(
                "Training the hierarchical vertex decoder requires vertex_guide."
            )
        is_r512 = final_resolutions.eq(512)

        latent_context = self._project_latent_context(x, final_resolutions)
        h = self.from_latent(x).type(self.dtype)
        for block in self.blocks[0][:-1]:
            h = block(h)

        latent_resolutions = final_resolutions // (2 ** self.vertex_num_stages)
        vertex_tokens = self.vertex_initial_projection(h)
        vertex_tokens = self._add_position(vertex_tokens, latent_resolutions)

        vertex_logits: List[sp.SparseTensor] = []
        child_tokens = None
        for stage_index in range(self.vertex_num_stages):
            vertex_tokens = self.vertex_attention_blocks[stage_index](
                vertex_tokens, latent_context
            )

            h = self.blocks[stage_index][-1](h)
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
            normalized_child_tokens = child_tokens.replace(
                self.vertex_occupancy_norms[stage_index](child_tokens.feats)
            )
            stage_logits = self.vertex_occupancy_heads[stage_index](
                normalized_child_tokens
            )
            vertex_logits.append(stage_logits)

            if stage_index + 1 < self.vertex_num_stages:
                keep = (
                    self._guide_mask(
                        child_tokens,
                        vertex_guide,
                        final_resolutions,
                        stage_index,
                    )
                    if vertex_guide is not None
                    else self._threshold_with_fallback(stage_logits, threshold)
                )
                vertex_tokens = self._row_select(child_tokens, keep)

        if child_tokens is None:
            raise RuntimeError("Hierarchical decoder produced no final child tokens.")

        # Predict final-resolution d_tri/d_vert before selecting the edge
        # context.  Both output scalars and edge tokens originate from this same
        # normalized R512 VAE feature field.
        h = h.type(x.dtype)
        normalized_h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        y = self.output_layer(normalized_h)

        proposal_logits = vertex_logits[-1]
        self.last_r512_proposal_logits = proposal_logits
        if bool(is_r512.all().item()):
            refined_logits, final_vertex_tokens = self._refine_r512_vertices(
                child_tokens,
                proposal_logits,
                normalized_h,
                y,
                latent_context,
                final_resolutions,
                vertex_guide,
            )
            vertex_logits[-1] = refined_logits
        elif bool(is_r512.any().item()):
            r512_child_tokens, selected_samples = self._select_batch_samples(
                child_tokens, is_r512
            )
            r512_proposal_logits, _ = self._select_batch_samples(
                proposal_logits, is_r512
            )
            r512_normalized_h, _ = self._select_batch_samples(
                normalized_h, is_r512
            )
            r512_y, _ = self._select_batch_samples(y, is_r512)
            r512_latent_context, _ = self._select_batch_samples(
                latent_context, is_r512
            )
            r512_vertex_guide = None
            if vertex_guide is not None:
                r512_vertex_guide, _ = self._select_batch_samples(
                    vertex_guide, is_r512
                )
            r512_resolutions = final_resolutions[selected_samples]
            refined_logits, refined_tokens = self._refine_r512_vertices(
                r512_child_tokens,
                r512_proposal_logits,
                r512_normalized_h,
                r512_y,
                r512_latent_context,
                r512_resolutions,
                r512_vertex_guide,
            )
            vertex_logits[-1] = self._restore_and_merge_samples(
                refined_logits,
                proposal_logits,
                selected_samples,
            )
            final_vertex_tokens = self._restore_and_merge_samples(
                refined_tokens,
                child_tokens,
                selected_samples,
            )
        else:
            # Preserve lower-resolution behavior and keep every new parameter
            # visible to DDP on ranks whose current mixed-resolution sample is
            # not R512.
            zero = self._new_parameter_zero_dependency()
            vertex_logits[-1] = proposal_logits.replace(
                proposal_logits.feats + zero
            )
            final_vertex_tokens = child_tokens.replace(child_tokens.feats + zero)
            self.last_r512_refinement_stats = {
                "active_voxels": 0,
                "low_dtri_voxels": 0,
                "edge_context_tokens": 0,
                "proposal_candidates": 0,
                "predicted_proposals": 0,
                "refinement_queries": 0,
            }

        if return_edge:
            if edge_vertex_coords is None or edge_pairs is None:
                raise ValueError(
                    "return_edge=True requires edge_vertex_coords and edge_pairs."
                )
            edge_logits = self._score_edge_pairs(
                final_vertex_tokens,
                final_resolutions,
                edge_vertex_coords,
                edge_pairs,
            )
            return y, vertex_logits, edge_logits
        if return_vertex_features:
            return y, vertex_logits, final_vertex_tokens
        return y, vertex_logits
