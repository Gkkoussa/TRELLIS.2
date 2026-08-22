#!/usr/bin/env python3
"""Evaluate R512 QEM vertex prediction with predicted-d_vert clustering.

This is an inference-only companion to ``eval_triangle_field_qem_vertex_vae.py``.
It leaves the trained model and the standard evaluator unchanged.  The vertex
head first produces its normal recursive R512 occupancy prediction.  Spatially
connected R512 support voxels whose decoded d_vert value exceeds a threshold
then define merge regions.  All predicted vertex voxels in one such region are
replaced by a single d_vert-weighted centroid.  A high-d_vert region containing
no vertex-head prediction also creates one vertex at its weighted centroid.

Two representations of the merged result are saved:

* the exact continuous weighted centroids, for point-cloud inspection; and
* one active support voxel per cluster, selected nearest the centroid, so that
  voxel-aligned QEM occupancy metrics remain well-defined.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import csv
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_triangle_field_qem_vertex_vae as base_eval
from eval_metadata_filters import (
    add_eval_metadata_filter_args,
    resolve_eval_metadata_filter_csv,
)
from trellis2 import datasets
from trellis2.utils.data_utils import recursive_to_device
from trellis2.utils.vertex_subdivision import compute_vertex_hierarchy_predictions
from trellis2.utils.vertex_visualization import (
    GT_COLOR,
    PRED_COLOR,
    add_image_label,
    build_vertex_renderings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate R512 recursive QEM vertex prediction, then merge vertex "
            "predictions within connected predicted-d_vert regions."
        )
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="latest", help="Checkpoint step or 'latest'.")
    parser.add_argument(
        "--ema_rate",
        default="0.9999",
        help="EMA checkpoint rate. Use 'none' for ordinary checkpoints.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--primary_threshold", type=float, default=0.5)
    parser.add_argument(
        "--rollout_threshold",
        type=float,
        default=0.1,
        help=(
            "Lowest recursive vertex threshold used to construct attention-model "
            "candidate support before applying the primary threshold."
        ),
    )
    parser.add_argument(
        "--dvert_threshold",
        type=float,
        default=0.8,
        help="Threshold in the original semantic d_vert [0,1] scale.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 26],
        default=26,
        help="Spatial connectivity for thresholded sparse R512 voxels.",
    )
    parser.add_argument(
        "--final_children_per_parent",
        type=int,
        default=None,
        help=(
            "Optional inference-only per-parent child limit. By default no "
            "top-k child restriction is applied."
        ),
    )
    parser.add_argument(
        "--child_limit_stages",
        type=int,
        nargs="+",
        default=[2, 3],
        help=(
            "Zero-based R512 decoder stages constrained by the child limit. "
            "Stages 0,1,2,3 produce R64,R128,R256,R512."
        ),
    )
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--num_visualizations", type=int, default=16)
    parser.add_argument("--visualization_seed", type=int, default=0)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--vertex_marker_pixels", type=int, default=6)
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def neighbor_offsets(connectivity: int) -> List[Tuple[int, int, int]]:
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if connectivity == 6 and abs(dx) + abs(dy) + abs(dz) != 1:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def packed_voxel_key(coord: Sequence[int], resolution: int) -> int:
    return (int(coord[0]) * resolution + int(coord[1])) * resolution + int(coord[2])


def cluster_and_merge_predictions(
    coords: torch.Tensor,
    dvert_unit: torch.Tensor,
    prediction: torch.Tensor,
    resolution: int,
    threshold: float,
    connectivity: int,
    create_empty_components: bool = True,
) -> Dict[str, object]:
    """Cluster high-d_vert support and merge predictions component-wise.

    The returned ``merged_mask`` is aligned with ``coords``.  Its cluster
    representatives are guaranteed to be supplied active support voxels.
    ``continuous_points`` contains the exact weighted centroids in normalized
    object coordinates for every component, plus unchanged centers for
    predictions outside the thresholded d_vert support.
    """
    coords_np = coords.detach().reshape(-1, 3).cpu().numpy().astype(np.int32)
    dvert_np = (
        dvert_unit.detach().reshape(-1).float().cpu().numpy().astype(np.float64)
    )
    prediction_np = prediction.detach().reshape(-1).bool().cpu().numpy()
    if len(coords_np) != len(dvert_np) or len(coords_np) != len(prediction_np):
        raise ValueError("coords, d_vert, and prediction must have equal lengths.")

    high_mask = dvert_np > float(threshold)
    high_rows = np.flatnonzero(high_mask)
    key_to_row = {
        packed_voxel_key(coords_np[row], resolution): int(row)
        for row in high_rows
    }
    visited = np.zeros(len(coords_np), dtype=np.bool_)
    component_id = np.full(len(coords_np), -1, dtype=np.int32)
    offsets = neighbor_offsets(connectivity)

    merged_mask = np.zeros(len(coords_np), dtype=np.bool_)
    continuous_points: List[np.ndarray] = []
    representative_rows: List[int] = []
    centroid_component_ids: List[int] = []
    component_sizes: List[int] = []
    component_prediction_counts: List[int] = []
    component_to_centroid_index: List[int] = []
    num_components = 0
    num_prediction_components = 0
    num_empty_prediction_components = 0
    num_multi_prediction_components = 0
    num_predictions_collapsed_away = 0
    max_predictions_in_component = 0

    for start_row in high_rows:
        start_row = int(start_row)
        if visited[start_row]:
            continue
        current_component_id = num_components
        num_components += 1
        component_to_centroid_index.append(-1)
        visited[start_row] = True
        queue = deque([start_row])
        component_rows: List[int] = []

        while queue:
            row = queue.popleft()
            component_rows.append(row)
            x, y, z = (int(value) for value in coords_np[row])
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (
                    nx < 0
                    or ny < 0
                    or nz < 0
                    or nx >= resolution
                    or ny >= resolution
                    or nz >= resolution
                ):
                    continue
                neighbor_key = (nx * resolution + ny) * resolution + nz
                neighbor_row = key_to_row.get(neighbor_key)
                if neighbor_row is not None and not visited[neighbor_row]:
                    visited[neighbor_row] = True
                    queue.append(neighbor_row)

        component_rows_np = np.asarray(component_rows, dtype=np.int64)
        component_id[component_rows_np] = current_component_id
        predicted_rows = component_rows_np[prediction_np[component_rows_np]]
        predicted_count = int(len(predicted_rows))
        if predicted_count == 0:
            num_empty_prediction_components += 1
        else:
            num_prediction_components += 1
            if predicted_count > 1:
                num_multi_prediction_components += 1
                num_predictions_collapsed_away += predicted_count - 1
            max_predictions_in_component = max(
                max_predictions_in_component, predicted_count
            )

        if predicted_count == 0 and not create_empty_components:
            continue

        weights = np.clip(dvert_np[component_rows_np], 1e-12, None)
        centers_grid = coords_np[component_rows_np].astype(np.float64) + 0.5
        centroid_grid = np.average(centers_grid, axis=0, weights=weights)
        distances_squared = np.sum((centers_grid - centroid_grid[None, :]) ** 2, axis=1)
        # First minimize centroid distance, then prefer larger d_vert, then the
        # stable supplied-support row order.
        representative_local = int(
            np.lexsort(
                (
                    component_rows_np,
                    -dvert_np[component_rows_np],
                    distances_squared,
                )
            )[0]
        )
        representative_row = int(component_rows_np[representative_local])
        component_to_centroid_index[current_component_id] = len(continuous_points)
        merged_mask[representative_row] = True
        representative_rows.append(representative_row)
        continuous_points.append(centroid_grid / float(resolution) - 0.5)
        centroid_component_ids.append(current_component_id)
        component_sizes.append(int(len(component_rows_np)))
        component_prediction_counts.append(predicted_count)

    unmatched_rows = np.flatnonzero(prediction_np & ~high_mask)
    merged_mask[unmatched_rows] = True
    for row in unmatched_rows:
        continuous_points.append(
            (coords_np[int(row)].astype(np.float64) + 0.5) / float(resolution) - 0.5
        )

    return {
        "merged_mask": torch.from_numpy(merged_mask).to(device=prediction.device),
        "high_mask": high_mask,
        "component_id": component_id,
        "continuous_points": np.asarray(continuous_points, dtype=np.float32).reshape(-1, 3),
        "representative_rows": np.asarray(representative_rows, dtype=np.int64),
        "centroid_component_ids": np.asarray(centroid_component_ids, dtype=np.int32),
        "component_sizes": np.asarray(component_sizes, dtype=np.int32),
        "component_prediction_counts": np.asarray(
            component_prediction_counts, dtype=np.int32
        ),
        "component_to_centroid_index": np.asarray(
            component_to_centroid_index, dtype=np.int64
        ),
        "component_gt_counts": np.zeros(len(representative_rows), dtype=np.int32),
        "unmatched_rows": unmatched_rows.astype(np.int64),
        "num_components": num_components,
        "num_prediction_components": num_prediction_components,
        "num_empty_prediction_components": num_empty_prediction_components,
        "num_empty_components_created": (
            num_empty_prediction_components if create_empty_components else 0
        ),
        "num_empty_components_skipped": (
            0 if create_empty_components else num_empty_prediction_components
        ),
        "num_multi_prediction_components": num_multi_prediction_components,
        "num_predictions_collapsed_away": num_predictions_collapsed_away,
        "max_predictions_in_component": max_predictions_in_component,
    }


def populate_gt_component_statistics(
    merge_result: Dict[str, object], target: torch.Tensor
) -> Dict[str, int]:
    component_id = np.asarray(merge_result["component_id"])
    target_np = target.detach().reshape(-1).bool().cpu().numpy()
    high_gt_rows = np.flatnonzero(target_np & (component_id >= 0))
    if len(high_gt_rows):
        ids, counts = np.unique(component_id[high_gt_rows], return_counts=True)
        component_gt_counts = dict(zip(ids.tolist(), counts.tolist()))
    else:
        component_gt_counts = {}

    centroid_component_ids = np.asarray(merge_result["centroid_component_ids"])
    merge_result["component_gt_counts"] = np.asarray(
        [component_gt_counts.get(int(component), 0) for component in centroid_component_ids],
        dtype=np.int32,
    )
    return {
        "gt_vertices_inside_high_dvert": int(len(high_gt_rows)),
        "components_with_multiple_gt_vertices": int(
            sum(count > 1 for count in component_gt_counts.values())
        ),
        "max_gt_vertices_in_component": int(
            max(component_gt_counts.values(), default=0)
        ),
    }


def save_float_point_cloud(
    path: Path,
    points: np.ndarray,
    color: Tuple[float, float, float],
    comment: str,
) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = (
        np.rint(np.asarray(color, dtype=np.float32) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    records = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    if len(points):
        records["x"], records["y"], records["z"] = points.T
        records["red"], records["green"], records["blue"] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"comment {comment}",
            f"element vertex {len(records)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


def component_colors(component_ids: np.ndarray) -> np.ndarray:
    component_ids = np.asarray(component_ids, dtype=np.int64).reshape(-1)
    colors = np.zeros((len(component_ids), 3), dtype=np.float32)
    valid = component_ids >= 0
    for row in np.flatnonzero(valid):
        hue = (int(component_ids[row]) * 0.6180339887498949) % 1.0
        colors[row] = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return colors


def labeled_sheet(image: torch.Tensor, label: str) -> torch.Tensor:
    return add_image_label(image, label).detach().cpu()


def save_summary_panel(images: List[torch.Tensor], path: Path, columns: int = 3) -> None:
    if not images:
        return
    pil_images = [base_eval.tensor_to_image(image) for image in images]
    width, height = pil_images[0].size
    rows = (len(pil_images) + columns - 1) // columns
    panel = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(pil_images):
        row, column = divmod(index, columns)
        panel.paste(image, (column * width, row * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def prefixed_metrics(prefix: str, counts: Tuple[int, int, int, int]) -> Dict[str, object]:
    metrics = base_eval.confusion_metrics(*counts)
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def summed_counts(rows: List[dict], prefix: str) -> Tuple[int, int, int, int]:
    return tuple(
        sum(int(row[f"{prefix}_{key}"]) for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    )


def macro_metrics(rows: List[dict], prefix: str) -> Dict[str, float]:
    return {
        key: float(np.mean([float(row[f"{prefix}_{key}"]) for row in rows]))
        if rows
        else 0.0
        for key in ("precision", "recall", "iou", "f1")
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    resolution = 512
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation and VoxelRenderer visualization require a CUDA GPU.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if args.num_visualizations < 0:
        raise ValueError("num_visualizations must be non-negative.")
    if args.image_resolution <= 0 or args.ssaa <= 0 or args.vertex_marker_pixels <= 0:
        raise ValueError("image_resolution, ssaa, and vertex_marker_pixels must be positive.")
    if not 0.0 < args.rollout_threshold <= args.primary_threshold < 1.0:
        raise ValueError(
            "Require 0 < rollout_threshold <= primary_threshold < 1."
        )
    if not 0.0 < args.dvert_threshold < 1.0:
        raise ValueError("dvert_threshold must be in (0,1) on the semantic scale.")
    if (
        args.final_children_per_parent is not None
        and args.final_children_per_parent <= 0
    ):
        raise ValueError("final_children_per_parent must be positive.")
    child_limit_applied = args.final_children_per_parent is not None
    requested_child_limit_stages = sorted(
        set(int(value) for value in args.child_limit_stages)
    )
    if requested_child_limit_stages and any(
        value < 0 or value >= 4 for value in requested_child_limit_stages
    ):
        raise ValueError("child_limit_stages must contain stage indices from 0 through 3.")
    child_limit_stages = (
        requested_child_limit_stages if child_limit_applied else []
    )

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
        raise ValueError("This evaluator requires pred_subdiv=False and supplied support.")

    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    distance_transform = str(dataset_args.get("distance_transform", "none"))
    if distance_transform not in {"none", "minus_one_one"}:
        raise ValueError(
            f"Unsupported distance_transform={distance_transform!r}; expected "
            "'none' or 'minus_one_one'."
        )
    configured_resolutions = [int(value) for value in dataset_args.get("resolutions", [])]
    if configured_resolutions and resolution not in configured_resolutions:
        raise ValueError(f"R512 is not configured in {configured_resolutions}.")
    dataset_args["resolution"] = resolution
    dataset_args.pop("resolutions", None)
    dataset_args.pop("instances_path", None)
    data_dir = base_eval.build_data_dir(root, args.split, resolution, dataset_args, args)
    dataset = datasets.SparseVoxelTriangleFieldDataset(
        json.dumps(data_dir), **dataset_args
    )
    if args.instances is not None:
        base_eval.restrict_dataset_instances(dataset, args.instances.resolve())
    base_eval.limit_dataset_instances(dataset, args.max_eval_samples)
    if len(dataset) == 0:
        raise RuntimeError("No evaluation instances remain after dataset filters.")

    step = base_eval.find_checkpoint_step(run_dir, args.ckpt, args.ema_rate)
    child_limit_suffix = (
        f"_r512top{args.final_children_per_parent}s"
        + "-".join(str(value) for value in child_limit_stages)
        if child_limit_applied
        else "_r512unlimited"
    )
    default_eval_name = (
        f"eval_qem_vertex_dvert_cluster_{args.split}_step{step:07d}"
        f"_dvert{args.dvert_threshold:.2f}_conn{args.connectivity}"
        f"{child_limit_suffix}"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / default_eval_name / "resolution_512"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(dataset, flush=True)
    print(f"Checkpoint step: {step}", flush=True)
    print(f"EMA rate: {args.ema_rate}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(
        (
            f"R512 child constraint: top-{args.final_children_per_parent} "
            f"at stages {child_limit_stages}"
            if child_limit_applied
            else "R512 child constraint: unlimited"
        ),
        flush=True,
    )
    print(
        f"d_vert clustering: semantic d_vert > {args.dvert_threshold:.3f}, "
        f"{args.connectivity}-connectivity",
        flush=True,
    )

    model_dict = base_eval.load_models(cfg, run_dir, step, args.ema_rate)
    encoder = model_dict["encoder"]
    decoder = model_dict["decoder"]
    decoder_vertex_child_tokens = bool(
        getattr(decoder, "vertex_logits_are_child_tokens", False)
    )

    indexed_dataset = base_eval.IndexedTriangleFieldDataset(dataset)
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
    visualizer = (
        base_eval.VertexVoxelVisualizer(
            resolution, args.image_resolution, args.ssaa
        )
        if visualization_indices
        else None
    )
    visualization_root = output_dir / "visualizations"
    visualization_manifest: List[dict] = []
    aggregate_before: List[torch.Tensor] = []
    aggregate_after: List[torch.Tensor] = []
    aggregate_after_error: List[torch.Tensor] = []
    aggregate_dvert: List[torch.Tensor] = []
    per_mesh_rows: List[dict] = []

    for data in tqdm(loader, desc="R512 vertex rollout + d_vert clustering"):
        data = recursive_to_device(data, torch.device("cuda"), non_blocking=True)
        instances = data["instance"]
        dataset_indices = data["dataset_index"].reshape(-1)
        z = encoder(data["x"], sample_posterior=args.sample_posterior)
        if decoder_vertex_child_tokens:
            y, vertex_logits = decoder(
                z,
                return_vertex=True,
                resolutions=data["resolution"],
                vertex_threshold=args.rollout_threshold,
            )
        else:
            y, vertex_logits = decoder(z, return_vertex=True)

        if not isinstance(vertex_logits, list) or not vertex_logits:
            raise TypeError("Decoder did not return a non-empty vertex-logit list.")
        expected_head_width = 1 if decoder_vertex_child_tokens else 8
        if any(stage.feats.shape[1] != expected_head_width for stage in vertex_logits):
            raise ValueError(
                f"Vertex head width mismatch; expected {expected_head_width}."
            )
        if y.feats.ndim != 2 or y.feats.shape[1] < 2:
            raise ValueError("Decoder output does not contain d_tri and d_vert channels.")
        if not torch.equal(y.coords, data["target"].coords):
            raise ValueError("Decoded support does not match GT target support.")
        if not torch.equal(data["vertex_occupancy"].coords, y.coords):
            raise ValueError("Vertex occupancy is not aligned with decoded support.")

        batch_ids = y.coords[:, 0].long()
        gt_occupancy = data["vertex_occupancy"].feats.reshape(-1).bool()
        # Convert the final decoder's d_vert channel back to the original
        # geometric [0,1] scale before applying the user-facing threshold.
        if distance_transform == "minus_one_one":
            dvert_unit_all = (y.feats[:, 1].float() + 1.0) * 0.5
        else:
            dvert_unit_all = y.feats[:, 1].float()
        dvert_unit_all = dvert_unit_all.clamp(0.0, 1.0)

        for batch_index, instance in enumerate(instances):
            mask = batch_ids.eq(batch_index)
            sample_coords = y.coords[mask, 1:4].long()
            sample_target = gt_occupancy[mask]
            sample_dvert = dvert_unit_all[mask]
            stages = compute_vertex_hierarchy_predictions(
                vertex_logits,
                sample_coords,
                sample_target,
                batch_index,
                resolution,
                threshold=args.primary_threshold,
            )
            final_recursive_score = stages[-1]["recursive_score"]
            if child_limit_applied:
                stage_masks = base_eval.compute_stage_child_limit_masks(
                    stages,
                    child_limit_stages,
                    args.final_children_per_parent,
                    args.rollout_threshold,
                )
                final_child_mask = stage_masks.get(len(stages) - 1)
                if final_child_mask is None:
                    raise RuntimeError("Child-limit propagation did not reach R512.")
                prediction_before = final_child_mask & (
                    final_recursive_score >= args.primary_threshold
                )
            else:
                prediction_before = (
                    final_recursive_score >= args.primary_threshold
                )

            merge_result = cluster_and_merge_predictions(
                sample_coords,
                sample_dvert,
                prediction_before,
                resolution,
                args.dvert_threshold,
                args.connectivity,
            )
            prediction_after = merge_result["merged_mask"]
            gt_cluster_stats = populate_gt_component_statistics(
                merge_result, sample_target
            )

            before_counts = base_eval.confusion_counts(
                prediction_before, sample_target
            )
            after_counts = base_eval.confusion_counts(
                prediction_after, sample_target
            )
            high_mask_np = np.asarray(merge_result["high_mask"])
            prediction_before_np = (
                prediction_before.detach().cpu().numpy().astype(np.bool_)
            )
            row = {
                "instance": instance,
                "resolution": resolution,
                "active_voxels": int(len(sample_coords)),
                "gt_vertex_voxels": int(sample_target.sum().item()),
                "pred_vertex_voxels_before": int(prediction_before.sum().item()),
                "pred_vertex_voxels_after": int(prediction_after.sum().item()),
                "vertices_removed_by_merge": int(
                    merge_result["num_predictions_collapsed_away"]
                ),
                "vertices_created_from_empty_components": int(
                    merge_result["num_empty_prediction_components"]
                ),
                "net_vertex_count_change": int(
                    prediction_after.sum().item() - prediction_before.sum().item()
                ),
                "high_dvert_voxels": int(high_mask_np.sum()),
                "high_dvert_components": int(merge_result["num_components"]),
                "components_with_predictions": int(
                    merge_result["num_prediction_components"]
                ),
                "components_without_predictions": int(
                    merge_result["num_empty_prediction_components"]
                ),
                "components_with_multiple_predictions": int(
                    merge_result["num_multi_prediction_components"]
                ),
                "max_predictions_in_component": int(
                    merge_result["max_predictions_in_component"]
                ),
                "predictions_inside_high_dvert": int(
                    (prediction_before_np & high_mask_np).sum()
                ),
                "predictions_outside_high_dvert": int(
                    (prediction_before_np & ~high_mask_np).sum()
                ),
                **gt_cluster_stats,
                **prefixed_metrics("before", before_counts),
                **prefixed_metrics("after", after_counts),
            }
            per_mesh_rows.append(row)

            dataset_index = int(dataset_indices[batch_index].item())
            if visualizer is None or dataset_index not in visualization_indices:
                continue

            before_images = build_vertex_renderings(
                visualizer.renderer,
                sample_coords,
                sample_target,
                prediction_before,
                resolution,
                confidence=final_recursive_score,
                minimum_marker_pixels=args.vertex_marker_pixels,
                include_mistakes=True,
            )
            after_images = build_vertex_renderings(
                visualizer.renderer,
                sample_coords,
                sample_target,
                prediction_after,
                resolution,
                confidence=sample_dvert,
                minimum_marker_pixels=args.vertex_marker_pixels,
                include_mistakes=True,
            )

            high_mask = torch.from_numpy(high_mask_np).to(sample_coords.device)
            high_scores = sample_dvert[high_mask]
            high_colors = torch.stack(
                [
                    high_scores,
                    0.15 + 0.85 * high_scores,
                    1.0 - high_scores,
                ],
                dim=1,
            ).clamp(0, 1)
            dvert_sheet = visualizer.renderer.render(
                sample_coords[high_mask],
                high_colors,
                resolution,
                minimum_marker_pixels=2,
            ).cpu()

            component_id_np = np.asarray(merge_result["component_id"])
            component_color_np = component_colors(component_id_np[high_mask_np])
            component_color = torch.from_numpy(component_color_np).to(
                device=sample_coords.device, dtype=torch.float32
            )
            component_sheet = visualizer.renderer.render(
                sample_coords[high_mask],
                component_color,
                resolution,
                minimum_marker_pixels=2,
            ).cpu()

            before_sheet = labeled_sheet(
                before_images["prediction"],
                f"BEFORE MERGE: {int(prediction_before.sum().item()):,} VERTICES",
            )
            after_sheet = labeled_sheet(
                after_images["prediction"],
                f"AFTER MERGE: {int(prediction_after.sum().item()):,} VERTICES",
            )
            gt_sheet = labeled_sheet(
                after_images["gt"],
                f"GT QEM: {int(sample_target.sum().item()):,} VERTICES",
            )
            error_sheet = labeled_sheet(
                after_images["error"],
                (
                    f"AFTER: TP {after_counts[0]:,} | FP {after_counts[1]:,} "
                    f"| FN {after_counts[2]:,}"
                ),
            )
            dvert_sheet = labeled_sheet(
                dvert_sheet,
                (
                    f"PREDICTED d_vert > {args.dvert_threshold:.2f}: "
                    f"{int(high_mask.sum().item()):,} VOXELS"
                ),
            )
            component_sheet = labeled_sheet(
                component_sheet,
                (
                    f"{int(merge_result['num_components']):,} CONNECTED COMPONENTS; "
                    f"{int(merge_result['num_prediction_components']):,} WITH PRED; "
                    f"{int(merge_result['num_empty_prediction_components']):,} CREATED"
                ),
            )

            instance_dir = visualization_root / instance
            instance_dir.mkdir(parents=True, exist_ok=True)
            for filename, sheet in (
                ("predicted_vertices_before_merge.jpg", before_sheet),
                ("predicted_vertices_after_merge.jpg", after_sheet),
                ("gt_qem_vertex_voxels.jpg", gt_sheet),
                ("after_merge_vertex_errors.jpg", error_sheet),
                ("predicted_dvert_above_threshold.jpg", dvert_sheet),
                ("predicted_dvert_connected_components.jpg", component_sheet),
            ):
                base_eval.tensor_to_image(sheet).save(instance_dir / filename)
            save_summary_panel(
                [
                    gt_sheet,
                    before_sheet,
                    after_sheet,
                    dvert_sheet,
                    component_sheet,
                    error_sheet,
                ],
                instance_dir / "dvert_cluster_merge_summary.jpg",
            )

            base_eval.save_vertex_point_cloud(
                instance_dir / "gt_qem_vertex_point_cloud.ply",
                sample_coords[sample_target],
                resolution,
                GT_COLOR,
            )
            base_eval.save_vertex_point_cloud(
                instance_dir / "predicted_vertex_point_cloud_before_merge.ply",
                sample_coords[prediction_before],
                resolution,
                PRED_COLOR,
            )
            base_eval.save_vertex_point_cloud(
                instance_dir / "predicted_vertex_point_cloud_after_merge_snapped.ply",
                sample_coords[prediction_after],
                resolution,
                (1.0, 0.8, 0.05),
            )
            save_float_point_cloud(
                instance_dir / "predicted_vertex_point_cloud_after_merge_centroids.ply",
                np.asarray(merge_result["continuous_points"]),
                (1.0, 0.8, 0.05),
                "one exact d_vert-weighted centroid per cluster and unchanged unmatched predictions",
            )

            np.savez_compressed(
                instance_dir / "dvert_cluster_predictions.npz",
                active_voxel_coords=sample_coords.detach().cpu().numpy().astype(np.int32),
                gt_vertex_target=sample_target.detach().cpu().numpy().astype(np.bool_),
                predicted_dvert=sample_dvert.detach().cpu().numpy().astype(np.float32),
                high_dvert_mask=high_mask_np,
                dvert_component_id=component_id_np,
                predicted_vertex_before=prediction_before_np,
                predicted_vertex_after_snapped=(
                    prediction_after.detach().cpu().numpy().astype(np.bool_)
                ),
                continuous_merged_points=np.asarray(
                    merge_result["continuous_points"], dtype=np.float32
                ),
                cluster_representative_rows=np.asarray(
                    merge_result["representative_rows"], dtype=np.int64
                ),
                cluster_component_ids=np.asarray(
                    merge_result["centroid_component_ids"], dtype=np.int32
                ),
                cluster_sizes=np.asarray(
                    merge_result["component_sizes"], dtype=np.int32
                ),
                cluster_prediction_counts=np.asarray(
                    merge_result["component_prediction_counts"], dtype=np.int32
                ),
                cluster_gt_counts=np.asarray(
                    merge_result["component_gt_counts"], dtype=np.int32
                ),
                unmatched_prediction_rows=np.asarray(
                    merge_result["unmatched_rows"], dtype=np.int64
                ),
                vertex_threshold=np.float32(args.primary_threshold),
                dvert_threshold=np.float32(args.dvert_threshold),
                connectivity=np.int32(args.connectivity),
                resolution=np.int32(resolution),
            )
            visualization_manifest.append(
                {
                    "instance": instance,
                    "summary": str(instance_dir / "dvert_cluster_merge_summary.jpg"),
                    "continuous_centroids": str(
                        instance_dir
                        / "predicted_vertex_point_cloud_after_merge_centroids.ply"
                    ),
                    "snapped_centroids": str(
                        instance_dir
                        / "predicted_vertex_point_cloud_after_merge_snapped.ply"
                    ),
                }
            )
            aggregate_before.append(before_sheet)
            aggregate_after.append(after_sheet)
            aggregate_after_error.append(error_sheet)
            aggregate_dvert.append(dvert_sheet)

    if not per_mesh_rows:
        raise RuntimeError("Evaluation produced no per-mesh rows.")

    before_micro = base_eval.confusion_metrics(*summed_counts(per_mesh_rows, "before"))
    after_micro = base_eval.confusion_metrics(*summed_counts(per_mesh_rows, "after"))
    metrics = {
        "checkpoint_step": step,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "resolution": resolution,
        "num_instances": len(per_mesh_rows),
        "posterior_mode": "sampled" if args.sample_posterior else "mean",
        "primary_vertex_threshold": args.primary_threshold,
        "candidate_rollout_threshold": (
            args.rollout_threshold if decoder_vertex_child_tokens else None
        ),
        "dvert_threshold_semantic_0_1": args.dvert_threshold,
        "distance_transform": distance_transform,
        "dvert_threshold_raw_decoder": (
            2.0 * args.dvert_threshold - 1.0
            if distance_transform == "minus_one_one"
            else args.dvert_threshold
        ),
        "dvert_connectivity": args.connectivity,
        "centroid_weighting": "predicted_dvert",
        "voxel_representative": "active high-dvert voxel nearest weighted centroid; ties prefer higher d_vert",
        "outside_cluster_policy": "keep predicted vertex unchanged",
        "empty_cluster_policy": "create one vertex at every high-dvert component centroid",
        "final_children_per_parent": args.final_children_per_parent,
        "child_limit_stages": child_limit_stages,
        "before_merge": {
            "micro": before_micro,
            "macro": macro_metrics(per_mesh_rows, "before"),
            "predicted_vertex_voxels": int(
                sum(row["pred_vertex_voxels_before"] for row in per_mesh_rows)
            ),
        },
        "after_merge_snapped_to_support": {
            "micro": after_micro,
            "macro": macro_metrics(per_mesh_rows, "after"),
            "predicted_vertex_voxels": int(
                sum(row["pred_vertex_voxels_after"] for row in per_mesh_rows)
            ),
        },
        "clustering_totals": {
            key: int(sum(int(row[key]) for row in per_mesh_rows))
            for key in (
                "vertices_removed_by_merge",
                "vertices_created_from_empty_components",
                "net_vertex_count_change",
                "high_dvert_voxels",
                "high_dvert_components",
                "components_with_predictions",
                "components_without_predictions",
                "components_with_multiple_predictions",
                "predictions_inside_high_dvert",
                "predictions_outside_high_dvert",
                "gt_vertices_inside_high_dvert",
                "components_with_multiple_gt_vertices",
            )
        },
        "metadata_filter_csv": resolve_eval_metadata_filter_csv(
            root, args.split, args
        ),
        "data_dir": data_dir,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_mesh_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_mesh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_mesh_rows)
    if visualization_manifest:
        with (visualization_root / "manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(visualization_manifest[0].keys())
            )
            writer.writeheader()
            writer.writerows(visualization_manifest)

    base_eval.save_image_grid(
        aggregate_before, visualization_root / "predicted_vertices_before_merge_grid.jpg"
    )
    base_eval.save_image_grid(
        aggregate_after, visualization_root / "predicted_vertices_after_merge_grid.jpg"
    )
    base_eval.save_image_grid(
        aggregate_after_error,
        visualization_root / "after_merge_vertex_errors_grid.jpg",
    )
    base_eval.save_image_grid(
        aggregate_dvert, visualization_root / "predicted_dvert_above_threshold_grid.jpg"
    )

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved metrics: {output_dir / 'metrics.json'}", flush=True)
    print(f"Saved visualizations: {visualization_root}", flush=True)


if __name__ == "__main__":
    main()
