#!/usr/bin/env python3
"""Evaluate recursive eight-child QEM vertex prediction for a triangle-field VAE."""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as tv_utils
from tqdm import tqdm

from eval_metadata_filters import (
    add_eval_metadata_filter_args,
    attach_eval_metadata_filter,
    resolve_eval_metadata_filter_csv,
)
from trellis2 import datasets, models
from trellis2.utils.data_utils import recursive_to_device
from trellis2.utils.vertex_subdivision import (
    child_offsets,
    compute_vertex_hierarchy_predictions,
    voxel_keys,
)
from trellis2.utils.vertex_visualization import (
    GT_COLOR,
    PRED_COLOR,
    VertexVoxelRenderer,
    add_image_label,
    build_child_mask_renderings,
    build_vertex_renderings,
)


DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll out all eight-child vertex heads and evaluate final-resolution "
            "QEM vertex occupancy on the supplied triangle-field support."
        )
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True, choices=[32, 64, 128, 256, 512])
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="latest", help="Checkpoint step or 'latest'.")
    parser.add_argument(
        "--ema_rate",
        default="0.9999",
        help="EMA checkpoint rate. Use 'none' to load ordinary encoder/decoder checkpoints.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--primary_threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--final_children_per_parent",
        type=int,
        default=None,
        help=(
            "Inference-only R256-to-R512 constraint. For each recursively predicted "
            "R256 parent, retain at most this many threshold-passing R512 children, "
            "ranked by the final head's local occupancy probability. The option is "
            "only applied when --resolution=512."
        ),
    )
    parser.add_argument(
        "--child_limit_stages",
        type=int,
        nargs="+",
        default=[3],
        help=(
            "Zero-based decoder stages constrained by --final_children_per_parent. "
            "For an R512 evaluation, stages 0,1,2,3 produce R64,R128,R256,R512. "
            "Constraints are only applied when --resolution=512. Default: stage 3."
        ),
    )
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--num_visualizations", type=int, default=16)
    parser.add_argument("--num_stage_visualizations", type=int, default=4)
    parser.add_argument("--visualization_seed", type=int, default=0)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument(
        "--vertex_marker_pixels",
        type=int,
        default=6,
        help=(
            "Minimum rendered width of vertex/error markers. The enlargement is "
            "image-space only and does not move voxel centers."
        ),
    )
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def checkpoint_prefix(ema_rate: str) -> str:
    if str(ema_rate).strip().lower() in {"", "none", "false", "0"}:
        return ""
    return f"_ema{ema_rate}"


def checkpoint_path(run_dir: Path, model_name: str, step: int, ema_rate: str) -> Path:
    return run_dir / "ckpts" / f"{model_name}{checkpoint_prefix(ema_rate)}_step{step:07d}.pt"


def find_checkpoint_step(run_dir: Path, requested: str, ema_rate: str) -> int:
    if requested != "latest":
        step = int(requested)
        missing = [
            path
            for path in (
                checkpoint_path(run_dir, "encoder", step, ema_rate),
                checkpoint_path(run_dir, "decoder", step, ema_rate),
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint files: {[str(path) for path in missing]}")
        return step

    prefix = checkpoint_prefix(ema_rate)
    encoder_pattern = str(run_dir / "ckpts" / f"encoder{prefix}_step*.pt")
    encoder_steps = {
        int(Path(path).stem.split("step")[-1])
        for path in glob.glob(encoder_pattern)
    }
    decoder_steps = {
        int(Path(path).stem.split("step")[-1])
        for path in glob.glob(str(run_dir / "ckpts" / f"decoder{prefix}_step*.pt"))
    }
    common_steps = sorted(encoder_steps & decoder_steps)
    if not common_steps:
        raise RuntimeError(
            f"No matching encoder/decoder checkpoints found under {run_dir / 'ckpts'} "
            f"for ema_rate={ema_rate}."
        )
    return common_steps[-1]


def build_data_dir(root: Path, split: str, resolution: int, dataset_args: dict, args) -> dict:
    voxel_root_key = dataset_args["voxel_root_key"]
    voxel_dirname = dataset_args["voxel_dirname"]
    split_root = root / "splits" / split
    filtered_base = root / "splits" / f"{split}_triangle_field_{resolution}"
    base_root = filtered_base if (filtered_base / "metadata.csv").is_file() else split_root
    split_voxel_root = split_root / f"{voxel_dirname}_{resolution}"
    canonical_voxel_root = root / f"{voxel_dirname}_{resolution}"
    voxel_root = (
        split_voxel_root
        if (split_voxel_root / "metadata.csv").is_file()
        else canonical_voxel_root
    )
    data_dir = {
        split: {
            "base": str(base_root),
            voxel_root_key: str(voxel_root),
        }
    }
    return attach_eval_metadata_filter(data_dir, root, split, args)


def restrict_dataset_instances(dataset, instances_path: Path) -> None:
    keep = {
        line.strip()
        for line in instances_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    dataset.instances = [
        (root, sha256)
        for root, sha256 in dataset.instances
        if str(sha256) in keep
    ]
    if len(dataset.metadata) > 0:
        dataset.metadata = dataset.metadata[
            dataset.metadata.index.astype(str).isin(keep)
        ]
    dataset.loads = [
        dataset.metadata.loc[sha256, dataset.num_voxels_column]
        for _, sha256 in dataset.instances
    ]


def limit_dataset_instances(dataset, max_eval_samples: int | None) -> None:
    if max_eval_samples is None:
        return
    if max_eval_samples <= 0:
        raise ValueError("--max_eval_samples must be positive when provided.")
    dataset.instances = dataset.instances[:max_eval_samples]
    keep = {str(sha256) for _, sha256 in dataset.instances}
    if len(dataset.metadata) > 0:
        dataset.metadata = dataset.metadata[
            dataset.metadata.index.astype(str).isin(keep)
        ]
    dataset.loads = dataset.loads[: len(dataset.instances)]


class IndexedTriangleFieldDataset(Dataset):
    """Expose stable mesh identifiers and fail loudly instead of retrying another sample."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        root, instance = self.dataset.instances[index]
        pack = self.dataset.get_instance(root, instance)
        pack["instance"] = str(instance)
        pack["dataset_index"] = torch.tensor(index, dtype=torch.long)
        return pack


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float | int]:
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "iou": safe_ratio(tp, tp + fp + fn),
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def confusion_counts(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[int, int, int, int]:
    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())
    tn = int((~prediction & ~target).sum().item())
    return tp, fp, fn, tn


def limit_prediction_children_per_parent(
    child_coords: torch.Tensor,
    local_probability: torch.Tensor,
    recursive_score: torch.Tensor,
    threshold: float,
    max_children: int,
) -> torch.Tensor:
    """Keep at most ``max_children`` qualifying children for each spatial parent.

    Recursive score determines whether a child qualifies at the requested
    threshold. Among qualifying siblings, the final head's local probability
    determines which children survive. Stable sorting makes ties deterministic
    with respect to the supplied sparse-support row order.
    """
    child_coords = child_coords.reshape(-1, 3).long()
    local_probability = local_probability.reshape(-1)
    recursive_score = recursive_score.reshape(-1)
    if len(child_coords) != len(local_probability) or len(child_coords) != len(recursive_score):
        raise ValueError("Child coordinates and probability tensors must have equal lengths.")
    if max_children <= 0:
        raise ValueError("max_children must be positive.")

    prediction = torch.zeros(
        len(child_coords), device=child_coords.device, dtype=torch.bool
    )
    eligible_rows = torch.nonzero(
        recursive_score >= float(threshold), as_tuple=False
    ).flatten()
    if len(eligible_rows) == 0:
        return prediction

    _, parent_inverse = torch.unique(
        child_coords // 2,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    eligible_parent = parent_inverse[eligible_rows]
    eligible_probability = local_probability[eligible_rows]

    # Sort by probability first, then stably group by parent. Within each
    # parent group, rows therefore remain ordered from highest to lowest score.
    probability_order = torch.argsort(
        eligible_probability, descending=True, stable=True
    )
    grouped_order = probability_order[
        torch.argsort(
            eligible_parent[probability_order],
            stable=True,
        )
    ]
    sorted_parent = eligible_parent[grouped_order]
    _, counts = torch.unique_consecutive(sorted_parent, return_counts=True)
    group_starts = torch.cumsum(counts, dim=0) - counts
    within_group_rank = torch.arange(
        len(grouped_order), device=child_coords.device
    ) - torch.repeat_interleave(group_starts, counts)
    selected_rows = eligible_rows[grouped_order[within_group_rank < max_children]]
    prediction[selected_rows] = True
    return prediction


def compute_stage_child_limit_masks(
    stages: List[Dict[str, torch.Tensor]],
    stage_indices: List[int],
    max_children: int,
    rollout_threshold: float,
) -> Dict[int, torch.Tensor]:
    """Apply child limits hierarchically and propagate them to later stages."""
    limited_stages = set(int(index) for index in stage_indices)
    masks: Dict[int, torch.Tensor] = {}
    previous_coords = None
    previous_mask = None
    previous_resolution = None

    for stage_index, stage in enumerate(stages):
        child_coords = stage["child_support_xyz"].reshape(-1, 3).long()
        recursive_score = stage["recursive_score"].reshape(-1)
        local_probability = stage["local_probability"].reshape(-1)
        if len(child_coords) != len(recursive_score):
            raise ValueError(f"Stage {stage_index} score/support lengths do not match.")

        if previous_mask is None:
            allowed_by_ancestor = torch.ones(
                len(child_coords), device=child_coords.device, dtype=torch.bool
            )
        else:
            selected_parent_coords = previous_coords[previous_mask]
            if len(selected_parent_coords) == 0:
                allowed_by_ancestor = torch.zeros(
                    len(child_coords), device=child_coords.device, dtype=torch.bool
                )
            else:
                selected_parent_keys = voxel_keys(
                    selected_parent_coords, previous_resolution
                )
                child_parent_keys = voxel_keys(
                    child_coords // 2, previous_resolution
                )
                allowed_by_ancestor = torch.isin(
                    child_parent_keys, selected_parent_keys
                )

        if stage_index in limited_stages:
            eligible_recursive_score = recursive_score.masked_fill(
                ~allowed_by_ancestor, float("-inf")
            )
            current_mask = limit_prediction_children_per_parent(
                child_coords,
                local_probability,
                eligible_recursive_score,
                rollout_threshold,
                max_children,
            )
        elif previous_mask is not None:
            current_mask = allowed_by_ancestor & (
                recursive_score >= rollout_threshold
            )
        else:
            # No constraint has started yet; keep the native recursive rollout.
            continue

        masks[stage_index] = current_mask
        previous_coords = child_coords
        previous_mask = current_mask
        previous_resolution = int(stage["child_resolution"].item())

    return masks


def hierarchical_child_limit_gt_statistics(
    child_coords: torch.Tensor,
    target: torch.Tensor,
    max_children: int,
    stage_indices: List[int],
    num_stages: int,
) -> Tuple[int, int]:
    """Return final occupied parents and oracle-retainable GT final children."""
    positive_coords = child_coords.reshape(-1, 3)[target.reshape(-1).bool()]
    if len(positive_coords) == 0:
        return 0, 0
    positive_coords = torch.unique(positive_coords.long(), dim=0)
    final_parent_count = len(torch.unique(positive_coords // 2, dim=0))
    limited_stages = set(int(index) for index in stage_indices)
    minimum_stage = min(limited_stages)
    node_coords = positive_coords
    node_values = torch.ones(
        len(node_coords), device=node_coords.device, dtype=torch.long
    )

    # Dynamic programming from final occupied voxels toward the coarsest
    # constrained stage. Each node value is the maximum number of final GT
    # vertices retainable below that branch under all finer constraints.
    for stage_index in range(num_stages - 1, minimum_stage - 1, -1):
        parent_coords, parent_inverse = torch.unique(
            node_coords // 2,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        keep = torch.ones(
            len(node_coords), device=node_coords.device, dtype=torch.bool
        )
        if stage_index in limited_stages:
            value_order = torch.argsort(
                node_values, descending=True, stable=True
            )
            grouped_order = value_order[
                torch.argsort(parent_inverse[value_order], stable=True)
            ]
            sorted_parent = parent_inverse[grouped_order]
            _, counts = torch.unique_consecutive(
                sorted_parent, return_counts=True
            )
            group_starts = torch.cumsum(counts, dim=0) - counts
            within_group_rank = torch.arange(
                len(grouped_order), device=node_coords.device
            ) - torch.repeat_interleave(group_starts, counts)
            keep = torch.zeros_like(keep)
            keep[grouped_order[within_group_rank < max_children]] = True

        parent_values = torch.zeros(
            len(parent_coords), device=node_coords.device, dtype=torch.long
        )
        parent_values.scatter_add_(
            0,
            parent_inverse[keep],
            node_values[keep],
        )
        node_coords = parent_coords
        node_values = parent_values

    return final_parent_count, int(node_values.sum().item())


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    return Image.fromarray(array.transpose(1, 2, 0), mode="RGB")


def save_image_grid(images: List[torch.Tensor], path: Path) -> None:
    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.stack(images).float().clamp(0, 1)
    nrow = max(1, int(math.ceil(math.sqrt(len(images)))))
    tv_utils.save_image(batch, str(path), nrow=nrow)


def save_vertex_point_cloud(
    path: Path,
    voxel_coords: torch.Tensor,
    resolution: int,
    color: Tuple[float, float, float],
) -> None:
    """Save voxel centers as a colored binary little-endian PLY point cloud."""
    coords = voxel_coords.detach().reshape(-1, 3).cpu().numpy().astype(np.float32)
    points = (coords + 0.5) / float(resolution) - 0.5
    rgb = np.rint(np.asarray(color, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)

    records = np.empty(
        len(points),
        dtype=np.dtype([
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]),
    )
    if len(points):
        records["x"] = points[:, 0]
        records["y"] = points[:, 1]
        records["z"] = points[:, 2]
        records["red"] = rgb[0]
        records["green"] = rgb[1]
        records["blue"] = rgb[2]

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        "comment TRELLIS.2 QEM vertex occupancy voxel centers",
        f"comment source_voxel_resolution {int(resolution)}",
        f"element vertex {len(records)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


class VertexVoxelVisualizer:
    def __init__(self, resolution: int, image_resolution: int, ssaa: int):
        self.resolution = resolution
        self.image_resolution = image_resolution
        self.renderer = VertexVoxelRenderer(image_resolution, ssaa)

    def render(
        self,
        coords: torch.Tensor,
        colors: torch.Tensor,
        resolution: int | None = None,
    ) -> torch.Tensor:
        return self.renderer.render(
            coords,
            colors,
            self.resolution if resolution is None else resolution,
        ).cpu()

    def comparison(
        self,
        gt_sheet: torch.Tensor,
        pred_sheet: torch.Tensor,
        error_sheet: torch.Tensor,
        instance: str,
        threshold: float,
    ) -> Image.Image:
        images = [tensor_to_image(gt_sheet), tensor_to_image(pred_sheet), tensor_to_image(error_sheet)]
        labels = [
            "GT QEM VERTEX VOXELS",
            f"PREDICTED VERTEX VOXELS (p >= {threshold:.2f})",
            "ERRORS: TP GREEN / FP RED / FN BLUE",
        ]
        header = 52
        panel = Image.new(
            "RGB",
            (sum(image.width for image in images), header + images[0].height),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        x_offset = 0
        for image, label in zip(images, labels):
            panel.paste(image, (x_offset, header))
            draw.text((x_offset + 10, 8), label, fill="black")
            draw.text((x_offset + 10, 28), f"R={self.resolution}  {instance}", fill="black")
            x_offset += image.width
        return panel


def load_models(cfg: dict, run_dir: Path, step: int, ema_rate: str) -> Dict[str, torch.nn.Module]:
    loaded = {}
    for name, model_cfg in cfg["models"].items():
        model = getattr(models, model_cfg["name"])(**model_cfg["args"])
        path = checkpoint_path(run_dir, name, step, ema_rate)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.cuda().eval()
        loaded[name] = model
        print(f"Loaded {name}: {path}", flush=True)
    return loaded


def mean_rows(rows: List[dict], keys: Iterable[str]) -> dict:
    return {
        key: float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
        for key in keys
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation and VoxelRenderer visualization require a CUDA GPU.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative.")
    if args.num_visualizations < 0 or args.num_stage_visualizations < 0:
        raise ValueError("Visualization counts must be non-negative.")
    if args.vertex_marker_pixels <= 0:
        raise ValueError("--vertex_marker_pixels must be positive.")
    if args.image_resolution <= 0 or args.ssaa <= 0:
        raise ValueError("image_resolution and ssaa must be positive.")
    if (
        args.final_children_per_parent is not None
        and args.final_children_per_parent <= 0
    ):
        raise ValueError("--final_children_per_parent must be positive when provided.")
    if not (0.0 < args.primary_threshold < 1.0):
        raise ValueError("primary_threshold must be in (0, 1).")
    thresholds = sorted({float(value) for value in args.thresholds} | {args.primary_threshold})
    if any(value <= 0.0 or value >= 1.0 for value in thresholds):
        raise ValueError("All thresholds must be in (0, 1).")
    child_limit_stages_requested = sorted(set(args.child_limit_stages))
    if any(index < 0 or index >= 4 for index in child_limit_stages_requested):
        raise ValueError("--child_limit_stages must contain only stage indices 0, 1, 2, 3.")
    child_limit_applied = (
        args.final_children_per_parent is not None and args.resolution == 512
    )
    child_limit_stages_applied = (
        child_limit_stages_requested if child_limit_applied else []
    )
    child_limit_transitions = [
        (
            f"R{512 // (2 ** (4 - stage_index))}_to_"
            f"R{512 // (2 ** (3 - stage_index))}"
        )
        for stage_index in child_limit_stages_applied
    ]

    torch.manual_seed(args.visualization_seed)
    torch.cuda.manual_seed_all(args.visualization_seed)
    run_dir = args.run_dir.resolve()
    root = args.root.resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    decoder_args = cfg["models"]["decoder"]["args"]
    if not decoder_args.get("pred_vertex_subdiv", False):
        raise ValueError("Run decoder does not have pred_vertex_subdiv=True.")
    if decoder_args.get("pred_subdiv", True):
        raise ValueError("This evaluator expects pred_subdiv=False and supplied sparse support.")

    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    configured_resolutions = [int(value) for value in dataset_args.get("resolutions", [])]
    if configured_resolutions and args.resolution not in configured_resolutions:
        raise ValueError(
            f"Resolution {args.resolution} is not configured in {configured_resolutions}."
        )
    dataset_args["resolution"] = args.resolution
    dataset_args.pop("resolutions", None)
    dataset_args.pop("instances_path", None)
    data_dir = build_data_dir(root, args.split, args.resolution, dataset_args, args)
    dataset = datasets.SparseVoxelTriangleFieldDataset(
        json.dumps(data_dir),
        **dataset_args,
    )
    if args.instances is not None:
        restrict_dataset_instances(dataset, args.instances.resolve())
    limit_dataset_instances(dataset, args.max_eval_samples)
    if len(dataset) == 0:
        raise RuntimeError("No evaluation instances remain after dataset filters.")

    step = find_checkpoint_step(run_dir, args.ckpt, args.ema_rate)
    child_limit_suffix = (
        f"_r512top{args.final_children_per_parent}s"
        + "-".join(str(index) for index in child_limit_stages_requested)
        if args.final_children_per_parent is not None
        else ""
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir
        / f"eval_qem_vertex_{args.split}_step{step:07d}{child_limit_suffix}"
        / f"resolution_{args.resolution}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(dataset, flush=True)
    print(f"Checkpoint step: {step}", flush=True)
    print(f"EMA rate: {args.ema_rate}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    model_dict = load_models(cfg, run_dir, step, args.ema_rate)
    encoder = model_dict["encoder"]
    decoder = model_dict["decoder"]
    decoder_vertex_child_tokens = bool(
        getattr(decoder, "vertex_logits_are_child_tokens", False)
    )
    # Run FP32-master-parameter checkpoints the same way they were trained.
    # In particular, FlashAttention requires FP16/BF16 QKV tensors.
    amp_enabled = str(cfg["trainer"]["args"].get("fp16_mode", "")).lower() == "amp"
    print(
        f"Evaluation autocast: {'FP16 AMP' if amp_enabled else 'disabled'}",
        flush=True,
    )
    rollout_threshold = min(thresholds)
    print(
        "R512 child constraints: "
        + (
            (
                f"top-{args.final_children_per_parent} at stages "
                f"{child_limit_stages_applied}"
            )
            if child_limit_applied
            else "unlimited"
        ),
        flush=True,
    )

    indexed_dataset = IndexedTriangleFieldDataset(dataset)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    visualization_count = min(args.num_visualizations, len(dataset))
    visualization_indices = set(
        random.Random(args.visualization_seed).sample(
            range(len(dataset)), visualization_count
        )
    )
    stage_visualization_count = min(
        args.num_stage_visualizations,
        visualization_count,
    )
    stage_visualization_indices = set(
        random.Random(args.visualization_seed + 1).sample(
            sorted(visualization_indices),
            stage_visualization_count,
        )
    )
    visualizer = (
        VertexVoxelVisualizer(args.resolution, args.image_resolution, args.ssaa)
        if visualization_indices
        else None
    )
    visualization_root = output_dir / "visualizations"
    aggregate_gt: List[torch.Tensor] = []
    aggregate_pred: List[torch.Tensor] = []
    aggregate_error: List[torch.Tensor] = []
    aggregate_mistakes: List[torch.Tensor] = []
    aggregate_support_context: List[torch.Tensor] = []
    aggregate_confidence: List[torch.Tensor] = []
    visualization_manifest: List[dict] = []

    threshold_totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in thresholds
    }
    per_mesh_rows: List[dict] = []
    total_voxels = 0
    total_positives = 0
    total_gt_parent_voxels = 0
    total_gt_retainable_children = 0

    for data in tqdm(loader, desc=f"Vertex-child rollout R{args.resolution}"):
        data = recursive_to_device(data, torch.device("cuda"), non_blocking=True)
        instances = data["instance"]
        dataset_indices = data["dataset_index"].reshape(-1)
        x = data["x"]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            z = encoder(x, sample_posterior=args.sample_posterior)
            if decoder_vertex_child_tokens:
                # Generate candidates once at the lowest requested threshold.  The
                # saved weakest-path scores then evaluate every higher threshold
                # without those candidates having already been pruned.
                y, vertex_logits = decoder(
                    z,
                    return_vertex=True,
                    resolutions=data["resolution"],
                    vertex_threshold=rollout_threshold,
                )
            else:
                y, vertex_logits = decoder(z, return_vertex=True)
        if not isinstance(vertex_logits, list) or len(vertex_logits) == 0:
            raise TypeError("Decoder did not return a non-empty vertex-logit stage list.")
        expected_head_width = 1 if decoder_vertex_child_tokens else 8
        if any(stage.feats.shape[1] != expected_head_width for stage in vertex_logits):
            raise ValueError(
                "Vertex head width does not match the configured decoder mode; "
                f"expected {expected_head_width}."
            )
        if not torch.equal(y.coords, data["target"].coords):
            raise ValueError("Decoded triangle support does not match the GT target support.")
        if not torch.equal(data["vertex_occupancy"].coords, y.coords):
            raise ValueError("Vertex occupancy is not aligned with the final decoder support.")

        batch_ids = y.coords[:, 0].long()
        gt_occupancy = data["vertex_occupancy"].feats.reshape(-1).bool()
        for batch_index, instance in enumerate(instances):
            mask = batch_ids.eq(batch_index)
            sample_coords = y.coords[mask, 1:4].long()
            sample_target = gt_occupancy[mask]
            stages = compute_vertex_hierarchy_predictions(
                vertex_logits,
                sample_coords,
                sample_target,
                batch_index,
                args.resolution,
                threshold=args.primary_threshold,
            )
            final_recursive_score = stages[-1]["recursive_score"]
            final_local_probability = stages[-1]["local_probability"]
            stage_child_selection_masks = {}
            if child_limit_applied:
                # Apply each requested top-k decision once at the lowest
                # evaluated threshold, then propagate selected branches into
                # every finer stage. Higher-threshold predictions are nested
                # subsets of these same selections.
                stage_child_selection_masks = compute_stage_child_limit_masks(
                    stages,
                    child_limit_stages_applied,
                    args.final_children_per_parent,
                    rollout_threshold,
                )
            final_child_selection_mask = stage_child_selection_masks.get(
                len(stages) - 1
            )
            predictions = {}
            for threshold in thresholds:
                if child_limit_applied:
                    if final_child_selection_mask is None:
                        raise RuntimeError(
                            "Child-limit propagation did not reach the final stage."
                        )
                    sample_prediction = final_child_selection_mask & (
                        final_recursive_score >= threshold
                    )
                else:
                    sample_prediction = final_recursive_score >= threshold
                predictions[threshold] = sample_prediction
                tp, fp, fn, tn = confusion_counts(sample_prediction, sample_target)
                threshold_totals[threshold]["tp"] += tp
                threshold_totals[threshold]["fp"] += fp
                threshold_totals[threshold]["fn"] += fn
                threshold_totals[threshold]["tn"] += tn

            sample_prediction = predictions[args.primary_threshold]
            tp, fp, fn, tn = confusion_counts(sample_prediction, sample_target)
            gt_parent_voxels = None
            gt_retainable_children = None
            gt_recall_ceiling = None
            if child_limit_applied:
                (
                    gt_parent_voxels,
                    gt_retainable_children,
                ) = hierarchical_child_limit_gt_statistics(
                    sample_coords,
                    sample_target,
                    args.final_children_per_parent,
                    child_limit_stages_applied,
                    len(stages),
                )
                gt_recall_ceiling = safe_ratio(
                    gt_retainable_children,
                    int(sample_target.sum().item()),
                )
                total_gt_parent_voxels += gt_parent_voxels
                total_gt_retainable_children += gt_retainable_children
            row = {
                "instance": instance,
                "resolution": args.resolution,
                "active_voxels": int(mask.sum().item()),
                "gt_vertex_voxels": int(sample_target.sum().item()),
                "pred_vertex_voxels": int(sample_prediction.sum().item()),
                "gt_positive_frac": safe_ratio(
                    int(sample_target.sum().item()), int(mask.sum().item())
                ),
                "pred_positive_frac": safe_ratio(
                    int(sample_prediction.sum().item()), int(mask.sum().item())
                ),
                "pred_gt_count_ratio": safe_ratio(
                    int(sample_prediction.sum().item()),
                    int(sample_target.sum().item()),
                ),
                "final_children_per_parent": (
                    args.final_children_per_parent
                    if child_limit_applied
                    else None
                ),
                "child_limit_stages": (
                    "-".join(str(index) for index in child_limit_stages_applied)
                    if child_limit_applied
                    else None
                ),
                "gt_parent_voxels": gt_parent_voxels,
                "gt_retainable_vertex_voxels_under_child_limit": gt_retainable_children,
                "gt_recall_ceiling_from_child_limit": gt_recall_ceiling,
                **confusion_metrics(tp, fp, fn, tn),
            }
            per_mesh_rows.append(row)
            total_voxels += int(mask.sum().item())
            total_positives += int(sample_target.sum().item())

            dataset_index = int(dataset_indices[batch_index].item())
            if visualizer is None or dataset_index not in visualization_indices:
                continue
            final_images = build_vertex_renderings(
                visualizer.renderer,
                sample_coords,
                sample_target,
                sample_prediction,
                args.resolution,
                confidence=final_recursive_score,
                minimum_marker_pixels=args.vertex_marker_pixels,
                include_mistakes=True,
            )
            gt_sheet = add_image_label(
                final_images["gt"],
                f"GT QEM VERTICES: {int(sample_target.sum().item()):,}",
            ).cpu()
            pred_sheet = add_image_label(
                final_images["prediction"],
                (
                    f"PREDICTED VERTICES: {int(sample_prediction.sum().item()):,}  "
                    f"THRESHOLD={args.primary_threshold:.2f}"
                    + (
                        (
                            f"  TOP-{args.final_children_per_parent} "
                            f"STAGES={child_limit_stages_applied}"
                        )
                        if child_limit_applied
                        else ""
                    )
                ),
            ).cpu()
            error_sheet = add_image_label(
                final_images["error"],
                f"TP GREEN {tp:,}  |  FP RED {fp:,}  |  FN BLUE {fn:,}",
            ).cpu()
            mistakes_sheet = add_image_label(
                final_images["mistakes"],
                f"MISTAKES ONLY: FP RED {fp:,}  |  FN BLUE {fn:,}",
            ).cpu()
            support_context_sheet = add_image_label(
                final_images["support_context"],
                (
                    f"TRIANGLE SUPPORT GRAY  |  TP GREEN {tp:,}  |  "
                    f"FP RED {fp:,}  |  FN BLUE {fn:,}"
                ),
            ).cpu()
            confidence_sheet = add_image_label(
                final_images["confidence"],
                (
                    "RAW RECURSIVE CONFIDENCE BEFORE CHILD LIMIT: "
                    "CYAN/BLUE LOW  ->  GREEN/YELLOW HIGH"
                    if child_limit_applied
                    else "RECURSIVE CONFIDENCE: CYAN/BLUE LOW  ->  GREEN/YELLOW HIGH"
                ),
            ).cpu()

            instance_dir = visualization_root / instance
            instance_dir.mkdir(parents=True, exist_ok=True)
            tensor_to_image(gt_sheet).save(instance_dir / "gt_vertex_voxels.jpg")
            tensor_to_image(pred_sheet).save(
                instance_dir / f"pred_vertex_voxels_t{args.primary_threshold:.2f}.jpg"
            )
            tensor_to_image(error_sheet).save(instance_dir / "vertex_error_overlay.jpg")
            tensor_to_image(mistakes_sheet).save(
                instance_dir / "vertex_mistakes_only.jpg"
            )
            tensor_to_image(support_context_sheet).save(
                instance_dir / "vertex_error_with_triangle_support.jpg"
            )
            tensor_to_image(confidence_sheet).save(
                instance_dir / "vertex_recursive_confidence.jpg"
            )
            visualizer.comparison(
                gt_sheet,
                pred_sheet,
                error_sheet,
                instance,
                args.primary_threshold,
            ).save(instance_dir / "gt_vs_pred_vertex_voxels.jpg")
            save_vertex_point_cloud(
                instance_dir / "gt_qem_vertex_point_cloud.ply",
                sample_coords[sample_target],
                args.resolution,
                GT_COLOR,
            )
            save_vertex_point_cloud(
                instance_dir
                / f"predicted_vertex_point_cloud_t{args.primary_threshold:.2f}.ply",
                sample_coords[sample_prediction],
                args.resolution,
                PRED_COLOR,
            )
            np.savez_compressed(
                instance_dir / "final_vertex_predictions.npz",
                active_voxel_coords=sample_coords.detach().cpu().numpy().astype(np.int32),
                gt_vertex_target=sample_target.detach().cpu().numpy().astype(np.bool_),
                predicted_vertex_target=sample_prediction.detach().cpu().numpy().astype(np.bool_),
                unconstrained_predicted_vertex_target=(
                    final_recursive_score >= args.primary_threshold
                ).detach().cpu().numpy().astype(np.bool_),
                final_local_probability=final_local_probability.detach().cpu().numpy().astype(np.float32),
                recursive_vertex_score=final_recursive_score.detach().cpu().numpy().astype(np.float32),
                resolution=np.int32(args.resolution),
                threshold=np.float32(args.primary_threshold),
                final_children_per_parent=np.int32(
                    args.final_children_per_parent
                    if child_limit_applied
                    else -1
                ),
                child_limit_stages=np.asarray(
                    child_limit_stages_applied, dtype=np.int32
                ),
            )

            stage_root = instance_dir / "stages"
            render_stage_details = dataset_index in stage_visualization_indices
            for stage_index, stage in enumerate(stages if render_stage_details else []):
                child_resolution = int(stage["child_resolution"].item())
                stage_dir = stage_root / f"stage_{stage_index}_r{child_resolution}"
                stage_dir.mkdir(parents=True, exist_ok=True)
                stage_recursive_prediction = stage["recursive_prediction"]
                stage_example_prediction = stage["example_teacher_prediction"]
                stage_selection_mask = stage_child_selection_masks.get(stage_index)
                if stage_selection_mask is not None:
                    stage_recursive_prediction = stage_selection_mask & (
                        stage["recursive_score"] >= args.primary_threshold
                    )
                    example_coords = (
                        stage["example_parent_xyz"].reshape(1, 3) * 2
                        + child_offsets(stage["example_parent_xyz"].device)
                    )
                    selected_coords = stage["child_support_xyz"][
                        stage_selection_mask
                    ]
                    example_selection = torch.isin(
                        voxel_keys(example_coords, child_resolution),
                        voxel_keys(selected_coords, child_resolution),
                    )
                    stage_example_prediction = example_selection & (
                        stage["example_recursive_score"] >= args.primary_threshold
                    )
                stage_images = build_vertex_renderings(
                    visualizer.renderer,
                    stage["child_support_xyz"],
                    stage["gt_child"],
                    stage_recursive_prediction,
                    child_resolution,
                    confidence=stage["recursive_score"],
                    minimum_marker_pixels=args.vertex_marker_pixels,
                    include_mistakes=True,
                )
                teacher_xyz = stage["child_support_xyz"][stage["teacher_prediction"]]
                teacher_colors = torch.tensor(
                    PRED_COLOR,
                    device=teacher_xyz.device,
                    dtype=torch.float32,
                ).expand(len(teacher_xyz), -1)
                teacher_sheet = visualizer.render(
                    teacher_xyz,
                    teacher_colors,
                    resolution=child_resolution,
                )
                stage_file_map = {
                    "gt_vertex_children.jpg": stage_images["gt"],
                    "recursive_vertex_children.jpg": stage_images["prediction"],
                    "recursive_error_overlay.jpg": stage_images["error"],
                    "recursive_mistakes_only.jpg": stage_images["mistakes"],
                    "recursive_error_with_triangle_support.jpg": stage_images["support_context"],
                    "recursive_confidence.jpg": stage_images["confidence"],
                }
                for filename, image in stage_file_map.items():
                    tensor_to_image(image).save(stage_dir / filename)
                tensor_to_image(teacher_sheet).save(
                    stage_dir / "teacher_forced_vertex_children.jpg"
                )

                child_images = build_child_mask_renderings(
                    visualizer.renderer,
                    stage["example_gt"],
                    stage_example_prediction,
                    stage["example_local_probability"],
                )
                for name in ("gt", "prediction", "error", "confidence"):
                    tensor_to_image(child_images[name]).save(
                        stage_dir / f"example_parent_child_mask_{name}.jpg"
                    )
                np.savez_compressed(
                    stage_dir / "stage_vertex_predictions.npz",
                    child_support_coords=stage["child_support_xyz"].detach().cpu().numpy().astype(np.int32),
                    gt_vertex_children=stage["gt_child"].detach().cpu().numpy().astype(np.bool_),
                    teacher_forced_prediction=stage["teacher_prediction"].detach().cpu().numpy().astype(np.bool_),
                    recursive_prediction=stage_recursive_prediction.detach().cpu().numpy().astype(np.bool_),
                    unconstrained_recursive_prediction=stage["recursive_prediction"].detach().cpu().numpy().astype(np.bool_),
                    local_probability=stage["local_probability"].detach().cpu().numpy().astype(np.float32),
                    recursive_score=stage["recursive_score"].detach().cpu().numpy().astype(np.float32),
                    example_parent_coord=stage["example_parent_xyz"].detach().cpu().numpy().astype(np.int32),
                    example_gt_child_mask=stage["example_gt"].detach().cpu().numpy().astype(np.bool_),
                    example_local_probability=stage["example_local_probability"].detach().cpu().numpy().astype(np.float32),
                    example_recursive_score=stage["example_recursive_score"].detach().cpu().numpy().astype(np.float32),
                    example_prediction=stage_example_prediction.detach().cpu().numpy().astype(np.bool_),
                    unconstrained_example_recursive_prediction=stage["example_recursive_prediction"].detach().cpu().numpy().astype(np.bool_),
                    child_resolution=np.int32(child_resolution),
                    threshold=np.float32(args.primary_threshold),
                    final_children_per_parent=np.int32(
                        args.final_children_per_parent
                        if stage_index in child_limit_stages_applied
                        else -1
                    ),
                    child_limit_directly_applied=np.bool_(
                        stage_index in child_limit_stages_applied
                    ),
                    constrained_by_coarser_stage=np.bool_(
                        stage_selection_mask is not None
                        and stage_index not in child_limit_stages_applied
                    ),
                )
            aggregate_gt.append(gt_sheet)
            aggregate_pred.append(pred_sheet)
            aggregate_error.append(error_sheet)
            aggregate_mistakes.append(mistakes_sheet)
            aggregate_support_context.append(support_context_sheet)
            aggregate_confidence.append(confidence_sheet)
            visualization_manifest.append({
                **row,
                "stage_visualizations": render_stage_details,
            })

    threshold_sweep = []
    for threshold in thresholds:
        counts = threshold_totals[threshold]
        threshold_sweep.append({
            "threshold": threshold,
            **confusion_metrics(**counts),
            "gt_vertex_voxels": counts["tp"] + counts["fn"],
            "pred_vertex_voxels": counts["tp"] + counts["fp"],
            "gt_positive_frac": safe_ratio(
                counts["tp"] + counts["fn"], total_voxels
            ),
            "pred_positive_frac": safe_ratio(
                counts["tp"] + counts["fp"], total_voxels
            ),
            "pred_gt_count_ratio": safe_ratio(
                counts["tp"] + counts["fp"],
                counts["tp"] + counts["fn"],
            ),
        })
    best_threshold = max(threshold_sweep, key=lambda item: item["f1"])

    primary_counts = threshold_totals[args.primary_threshold]
    micro = {
        **confusion_metrics(**primary_counts),
        "active_voxels": total_voxels,
        "gt_vertex_voxels": total_positives,
        "pred_vertex_voxels": primary_counts["tp"] + primary_counts["fp"],
        "gt_positive_frac": safe_ratio(total_positives, total_voxels),
        "pred_positive_frac": safe_ratio(
            primary_counts["tp"] + primary_counts["fp"], total_voxels
        ),
        "pred_gt_count_ratio": safe_ratio(
            primary_counts["tp"] + primary_counts["fp"], total_positives
        ),
    }
    macro_keys = [
        "precision",
        "recall",
        "iou",
        "f1",
        "gt_positive_frac",
        "pred_positive_frac",
        "pred_gt_count_ratio",
    ]
    metrics = {
        "checkpoint_step": step,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "resolution": args.resolution,
        "num_instances": len(per_mesh_rows),
        "num_final_visualizations": visualization_count,
        "num_stage_visualizations": stage_visualization_count,
        "vertex_marker_pixels": args.vertex_marker_pixels,
        "posterior_mode": "sampled" if args.sample_posterior else "mean",
        "primary_threshold": args.primary_threshold,
        "final_children_per_parent_requested": args.final_children_per_parent,
        "final_children_per_parent_applied": (
            args.final_children_per_parent if child_limit_applied else None
        ),
        "child_limit_stages_requested": child_limit_stages_requested,
        "child_limit_stages_applied": child_limit_stages_applied,
        "child_limit_transitions": child_limit_transitions,
        "gt_parent_voxels": (
            total_gt_parent_voxels if child_limit_applied else None
        ),
        "gt_retainable_vertex_voxels_under_child_limit": (
            total_gt_retainable_children if child_limit_applied else None
        ),
        "gt_recall_ceiling_from_child_limit": (
            safe_ratio(total_gt_retainable_children, total_positives)
            if child_limit_applied
            else None
        ),
        "prediction_mode": (
            "recursive_attention_child_token_rollout_intersected_with_triangle_support"
            if decoder_vertex_child_tokens
            else "recursive_eight_child_rollout_intersected_with_triangle_support"
        ) + ("_hierarchical_stage_topk_per_parent" if child_limit_applied else ""),
        "candidate_rollout_threshold": (
            rollout_threshold if decoder_vertex_child_tokens else None
        ),
        "recursive_score_definition": "minimum child probability along the four-stage path",
        "micro": micro,
        "macro": mean_rows(per_mesh_rows, macro_keys),
        "threshold_sweep": threshold_sweep,
        "best_threshold_by_micro_f1": {
            "threshold": best_threshold["threshold"],
            "f1": best_threshold["f1"],
            "precision": best_threshold["precision"],
            "recall": best_threshold["recall"],
            "iou": best_threshold["iou"],
        },
        "metadata_filter_csv": resolve_eval_metadata_filter_csv(root, args.split, args),
        "data_dir": data_dir,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_mesh_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_mesh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_mesh_rows)
    if visualization_manifest:
        with (visualization_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(visualization_manifest[0].keys()))
            writer.writeheader()
            writer.writerows(visualization_manifest)
    save_image_grid(aggregate_gt, visualization_root / "gt_vertex_voxels_grid.jpg")
    save_image_grid(
        aggregate_pred,
        visualization_root / f"pred_vertex_voxels_t{args.primary_threshold:.2f}_grid.jpg",
    )
    save_image_grid(aggregate_error, visualization_root / "vertex_error_overlay_grid.jpg")
    save_image_grid(
        aggregate_mistakes,
        visualization_root / "vertex_mistakes_only_grid.jpg",
    )
    save_image_grid(
        aggregate_support_context,
        visualization_root / "vertex_error_with_triangle_support_grid.jpg",
    )
    save_image_grid(
        aggregate_confidence,
        visualization_root / "vertex_recursive_confidence_grid.jpg",
    )

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved metrics: {output_dir / 'metrics.json'}", flush=True)
    print(f"Saved visualizations: {visualization_root}", flush=True)


if __name__ == "__main__":
    main()
