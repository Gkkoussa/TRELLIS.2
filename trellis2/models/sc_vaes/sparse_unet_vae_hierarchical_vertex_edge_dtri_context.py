"""R512 low-d_tri context for direct hierarchical vertex/edge prediction.

This is an isolated alternative to the d_tri refinement decoder.  The
hierarchical R512 vertex head is the final vertex classifier.  Its selected
512-D vertex tokens receive one residual cross-attention update from low-d_tri
R512 VAE tokens and then go directly to the existing symmetric edge head:

    final hierarchical vertex tokens
        -> cross-attend to projected low-d_tri R512 VAE tokens
        -> existing symmetric pairwise edge head

There is deliberately no second vertex classifier, vertex self-attention,
latent cross-attention, or refinement FFN after the edge-context attention.
"""

from typing import Optional

import torch
import torch.nn as nn

from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from ...modules.sparse.attention import SparseMultiHeadAttention
from ...modules.utils import convert_module_to_f16
from .sparse_unet_vae_hierarchical_vertex_edge_dtri_refine import (
    SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder,
)


class R512VertexEdgeCrossAttentionBlock(nn.Module):
    """One residual vertex-query to low-d_tri-voxel cross-attention layer."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        use_checkpoint: bool = True,
        qk_rms_norm: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        # Keep these names aligned with the corresponding part of the old
        # refinement block so its trained cross-attention weights load exactly.
        self.edge_norm = LayerNorm32(
            channels, elementwise_affine=True, eps=1e-6
        )
        self.edge_cross_attention = SparseMultiHeadAttention(
            channels,
            ctx_channels=channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qk_rms_norm=qk_rms_norm,
        )

    def _forward(
        self,
        vertex_tokens: sp.SparseTensor,
        edge_tokens: sp.SparseTensor,
    ) -> sp.SparseTensor:
        h = vertex_tokens.replace(self.edge_norm(vertex_tokens.feats))
        return vertex_tokens + self.edge_cross_attention(h, edge_tokens)

    def forward(
        self,
        vertex_tokens: sp.SparseTensor,
        edge_tokens: sp.SparseTensor,
    ) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                vertex_tokens,
                edge_tokens,
                use_reentrant=False,
            )
        return self._forward(vertex_tokens, edge_tokens)


class SparseUnetVaeHierarchicalVertexEdgeDtriContextDecoder(
    SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder
):
    """Final hierarchical vertices plus configurable R512 edge context."""

    def __init__(
        self,
        *args,
        r512_edge_context_architecture: str = "cross_only",
        r512_edge_context_num_heads: Optional[int] = None,
        r512_edge_context_use_checkpoint: bool = True,
        r512_edge_context_qk_rms_norm: bool = True,
        r512_vertex_query_threshold: float = 0.5,
        **kwargs,
    ):
        architecture = str(r512_edge_context_architecture).lower()
        if architecture != "cross_only":
            raise ValueError(
                "r512_edge_context_architecture currently supports only "
                "'cross_only'. Use the separate dtri-refine decoder for the "
                "old full refinement architecture."
            )
        if not (0.0 < r512_vertex_query_threshold < 1.0):
            raise ValueError("r512_vertex_query_threshold must be in (0, 1).")

        inherited_vertex_heads = int(kwargs.get("vertex_num_heads", 4))
        context_heads = (
            inherited_vertex_heads
            if r512_edge_context_num_heads is None
            else int(r512_edge_context_num_heads)
        )

        # Build the established hierarchy, R512 VAE edge projection, and
        # checkpoint-compatible module names, then replace only the old full
        # refinement block/head with the direct edge-context path.
        super().__init__(
            *args,
            r512_vertex_proposal_threshold=r512_vertex_query_threshold,
            r512_refine_num_heads=context_heads,
            r512_refine_use_checkpoint=r512_edge_context_use_checkpoint,
            r512_refine_qk_rms_norm=r512_edge_context_qk_rms_norm,
            **kwargs,
        )
        if context_heads <= 0 or self.vertex_feature_channels % context_heads != 0:
            raise ValueError(
                "r512_edge_context_num_heads must divide vertex_feature_channels."
            )

        self.r512_edge_context_architecture = architecture
        self.r512_vertex_query_threshold = float(r512_vertex_query_threshold)
        self.r512_edge_context_num_heads = context_heads
        self.has_r512_dtri_edge_refinement = False
        self.has_r512_dtri_edge_context = True

        self.r512_vertex_refinement = R512VertexEdgeCrossAttentionBlock(
            self.vertex_feature_channels,
            num_heads=context_heads,
            use_checkpoint=bool(r512_edge_context_use_checkpoint),
            qk_rms_norm=bool(r512_edge_context_qk_rms_norm),
        )
        # Parent forward/unused-parameter handling refers to these attributes.
        # Identities keep that code shared without adding a second vertex head.
        self.r512_refined_vertex_norm = nn.Identity()
        self.r512_refined_vertex_head = nn.Identity()

        self.r512_vertex_refinement.apply(self._initialize_linear)
        if self.use_fp16:
            self.r512_vertex_refinement.apply(convert_module_to_f16)

        # These are the only parameters present in an old dtri-refine
        # checkpoint but intentionally absent from this architecture.
        self.finetune_allowed_unexpected_prefixes = (
            *getattr(self, "finetune_allowed_unexpected_prefixes", tuple()),
            "r512_vertex_refinement.vertex_norm.",
            "r512_vertex_refinement.vertex_self_attention.",
            "r512_vertex_refinement.latent_norm.",
            "r512_vertex_refinement.latent_cross_attention.",
            "r512_vertex_refinement.mlp_norm.",
            "r512_vertex_refinement.mlp.",
            "r512_refined_vertex_norm.",
            "r512_refined_vertex_head.",
        )
        self.last_r512_edge_context_stats = {}

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        """Contextualize final vertex features without changing vertex logits."""
        del latent_context  # No post-vertex latent cross-attention in this version.

        vertex_probability = torch.sigmoid(
            proposal_logits.feats.float()
        ).reshape(-1)
        predicted_keep = vertex_probability >= self.r512_vertex_query_threshold
        query_keep = predicted_keep.clone()
        if self.training:
            if vertex_guide is None:
                raise ValueError("R512 edge-context training requires vertex_guide.")
            gt_keep = self._guide_mask(
                child_tokens,
                vertex_guide,
                final_resolutions,
                self.vertex_num_stages - 1,
            )
            query_keep |= gt_keep
        else:
            query_keep = self._mask_with_per_sample_fallback(
                vertex_probability,
                proposal_logits.layout,
                query_keep,
                choose_lowest=False,
            )

        vertex_queries = self._row_select(child_tokens, query_keep)
        edge_tokens, raw_edge_keep = self._build_r512_edge_tokens(
            normalized_h, y, final_resolutions
        )
        edge_aware_vertex_tokens = self.r512_vertex_refinement(
            vertex_queries, edge_tokens
        )

        stats = {
            "active_voxels": int(len(y.feats)),
            "low_dtri_voxels": int(raw_edge_keep.sum().item()),
            "edge_context_tokens": int(len(edge_tokens.feats)),
            "vertex_candidates": int(len(proposal_logits.feats)),
            "predicted_vertices": int(predicted_keep.sum().item()),
            "edge_queries": int(len(edge_aware_vertex_tokens.feats)),
        }
        self.last_r512_edge_context_stats = stats
        # Keep the inherited mixed-resolution/status hook populated too.
        self.last_r512_refinement_stats = stats

        # The hierarchical 512->1 output is final and is returned unchanged.
        return proposal_logits, edge_aware_vertex_tokens


class SparseUnetVaeHierarchicalVertexEdgeDtriContextNoVaeFusionDecoder(
    SparseUnetVaeHierarchicalVertexEdgeDtriContextDecoder
):
    """Cross-only edge context without vertex/VAE fusion at any stage.

    Every child stage still uses the sparse VAE field as coordinate support,
    but every child token feature contains only the expanded parent feature,
    its positional embedding, and local refinement.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The no-fusion architecture has no trainable VAE-to-vertex projections
        # or concatenation/fusion MLPs at any hierarchy resolution.
        self.vertex_vae_feature_projections = nn.ModuleList([
            nn.Identity() for _ in range(self.vertex_num_stages)
        ])
        self.vertex_child_fusions = nn.ModuleList([
            nn.Identity() for _ in range(self.vertex_num_stages)
        ])
        self.has_vertex_vae_fusion = False
        self.finetune_allowed_unexpected_prefixes = (
            *self.finetune_allowed_unexpected_prefixes,
            "vertex_vae_feature_projections.",
            "vertex_child_fusions.",
        )

    def _make_child_candidates(
        self,
        stage_index: int,
        parent_tokens: sp.SparseTensor,
        child_vae_features: sp.SparseTensor,
        child_resolutions: torch.Tensor,
    ) -> sp.SparseTensor:
        expanded = self.vertex_child_feature_projections[stage_index](
            parent_tokens
        ).feats.reshape(-1, 8, self.vertex_feature_channels)

        output_coords = []
        propagated_features = []
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
                remainder[:, 0]
                + 2 * remainder[:, 1]
                + 4 * remainder[:, 2]
            )
            global_parent_rows = local_parent_rows[valid] + parent_slice.start
            output_coords.append(child_vae_features.coords[child_rows])
            propagated_features.append(
                expanded[global_parent_rows, child_index]
            )

        if not output_coords:
            raise RuntimeError(
                f"Vertex hierarchy stage {stage_index} produced no supported "
                "child tokens."
            )

        tokens = sp.SparseTensor(
            torch.cat(propagated_features, dim=0),
            torch.cat(output_coords, dim=0),
            torch.Size([
                parent_tokens.shape[0],
                self.vertex_feature_channels,
            ]),
        )
        tokens = self._add_position(tokens, child_resolutions)
        for block in self.vertex_local_refiners[stage_index]:
            tokens = block(tokens)
        return tokens
