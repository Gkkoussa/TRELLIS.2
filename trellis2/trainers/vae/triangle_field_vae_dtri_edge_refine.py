"""Trainer for R512 low-d_tri edge-context vertex refinement."""

from typing import Dict, Tuple

import torch
import torch.distributed as dist

from ...modules import sparse as sp
from .triangle_field_vae import TriangleFieldVaeTrainer


class TriangleFieldVaeDtriEdgeRefineTrainer(TriangleFieldVaeTrainer):
    """Add preliminary R512 proposal supervision to the refined vertex loss."""

    def __init__(
        self,
        *args,
        lambda_vertex_proposal: float = 1.0,
        **kwargs,
    ):
        self.lambda_vertex_proposal = float(lambda_vertex_proposal)
        if self.lambda_vertex_proposal <= 0:
            raise ValueError("lambda_vertex_proposal must be positive.")
        super().__init__(*args, **kwargs)
        decoder = self.models.get("decoder")
        if not bool(
            self._module_attr(decoder, "has_r512_dtri_edge_refinement", False)
        ):
            raise ValueError(
                "TriangleFieldVaeDtriEdgeRefineTrainer requires the R512 "
                "d_tri edge-refinement decoder."
            )
        if self.vertex_training_resolutions != (512,):
            raise ValueError(
                "The d_tri edge refinement experiment requires "
                "vertex_training_resolutions=[512]."
            )

    def finetune_from(self, finetune_ckpt) -> None:
        # First load every existing VAE, hierarchy, and edge-head parameter.
        super().finetune_from(finetune_ckpt)

        # The refined head has the same shape and task as the loaded preliminary
        # R512 proposal head.  Copying it gives the residual refiner a calibrated
        # starting classifier instead of a second random binary head.
        decoder = self.models.get("decoder")
        initialize = getattr(decoder, "initialize_refined_head_from_proposal", None)
        if initialize is None:
            raise ValueError("Decoder cannot initialize its refined R512 head.")
        initialize()

        # BasicTrainer created optimizer/master/EMA tensors before calling
        # finetune_from.  Synchronize all of them with the copied classifier.
        state_dicts = {
            name: model.state_dict() for name, model in self.models.items()
        }
        self._state_dicts_to_master_params(self.master_params, state_dicts)
        if self.is_master:
            for ema_params in self.ema_params:
                self._state_dicts_to_master_params(ema_params, state_dicts)
            print(
                "Initialized decoder.r512_refined_vertex_head from the loaded "
                "preliminary R512 proposal head."
            )

    def training_losses(
        self,
        x: sp.SparseTensor,
        target: sp.SparseTensor,
        vertex_occupancy: sp.SparseTensor = None,
        resolution: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        terms, status = super().training_losses(
            x,
            target,
            vertex_occupancy=vertex_occupancy,
            resolution=resolution,
            **kwargs,
        )
        if vertex_occupancy is None or resolution is None:
            raise ValueError(
                "R512 proposal supervision requires vertex_occupancy and resolution."
            )

        decoder = self.models["decoder"]
        proposal_logits = getattr(decoder, "last_r512_proposal_logits", None)
        if proposal_logits is None:
            raise RuntimeError("Decoder did not expose preliminary R512 proposal logits.")

        final_resolutions = resolution.reshape(-1)
        stage_target, supervised, child_resolutions = (
            self._build_vertex_token_stage_target(
                proposal_logits,
                vertex_occupancy,
                final_resolutions,
                stage_index=3,
                num_stages=4,
            )
        )
        vertex_sample_mask = self._vertex_training_sample_mask(final_resolutions)
        sample_batch_ids = proposal_logits.coords[:, 0].long()
        supervised &= vertex_sample_mask[sample_batch_ids]

        local_sample_count = int(vertex_sample_mask.sum().item())
        sample_counts = torch.tensor(
            [local_sample_count],
            device=proposal_logits.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sample_counts, op=dist.ReduceOp.SUM)
        global_sample_count = int(sample_counts[0].item())
        ddp_scale = (
            self.world_size * local_sample_count / global_sample_count
            if global_sample_count > 0
            else 0.0
        )

        proposal_loss, proposal_status = self._vertex_child_loss_and_metrics(
            proposal_logits,
            stage_target,
            supervised,
            child_resolutions,
            stage_index=3,
            use_asymmetric_loss=False,
            prefix="vertex/r512_proposal",
        )
        proposal_loss = proposal_loss * ddp_scale
        terms["bce_vertex_r512_proposal"] = proposal_loss
        terms["loss"] = (
            terms["loss"] + self.lambda_vertex_proposal * proposal_loss
        )
        status.update(proposal_status)
        status["vertex/r512_proposal/loss_ddp_scale"] = float(ddp_scale)
        status["vertex/r512_proposal/lambda"] = self.lambda_vertex_proposal

        refinement_stats = getattr(
            decoder, "last_r512_refinement_stats", {}
        )
        for name, value in refinement_stats.items():
            status[f"vertex/r512_refine/{name}"] = float(value)

        if self.debug_nans:
            status.update(
                self._debug_tensor_stats(
                    "loss/r512_proposal", proposal_loss.reshape(1)
                )
            )
            self._debug_abort_if_nonfinite("r512_proposal_loss", status)
        return terms, status

