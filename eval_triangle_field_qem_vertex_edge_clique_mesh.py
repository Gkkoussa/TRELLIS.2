#!/usr/bin/env python3
"""Evaluate R512 vertex/edge prediction and extract a centroid-merged clique mesh.

Edges are predicted on the original final vertex-head predictions.  Afterwards,
predicted vertices inside each connected high-d_vert component are merged onto
that component's d_vert-weighted centroid.  Empty components do not create a
vertex.  Edge endpoints are remapped to the merged vertices before
duplicate/self edges are removed and 3-cliques are extracted.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_triangle_field_qem_vertex_vae as vertex_eval
from eval_triangle_field_qem_vertex_vae_dvert_cluster import (
    cluster_and_merge_predictions,
    save_float_point_cloud,
)
from eval_metadata_filters import (
    add_eval_metadata_filter_args,
    resolve_eval_metadata_filter_csv,
)
from trellis2 import datasets
from trellis2.utils.data_utils import recursive_to_device
from trellis2.utils.vertex_subdivision import (
    compute_vertex_hierarchy_predictions,
    voxel_keys,
)
from trellis2.utils.vertex_visualization import (
    GT_COLOR,
    PRED_COLOR,
    add_image_label,
    build_vertex_renderings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict R512 QEM vertices, score every unordered predicted-vertex "
            "pair with the symmetric edge head, and emit one face for every "
            "three-clique in the predicted graph."
        )
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument(
        "--ema_rate",
        default="none",
        help="Use 'none' for ordinary checkpoints or an EMA rate such as 0.9999.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--vertex_threshold", type=float, default=0.5)
    parser.add_argument(
        "--rollout_threshold",
        type=float,
        default=0.1,
        help=(
            "Candidate-pruning threshold used inside the recursive decoder. It "
            "must not exceed --vertex_threshold."
        ),
    )
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument(
        "--dvert_threshold",
        type=float,
        default=0.8,
        help="Semantic decoded d_vert threshold used to form merge components.",
    )
    parser.add_argument(
        "--cluster_connectivity",
        type=int,
        choices=[6, 26],
        default=26,
        help="Voxel connectivity for high-d_vert merge components.",
    )
    parser.add_argument(
        "--edge_pair_batch_size",
        type=int,
        default=16384,
        help="Number of unordered vertex pairs scored in one GPU chunk.",
    )
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--num_visualizations", type=int, default=16)
    parser.add_argument("--visualization_seed", type=int, default=0)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--vertex_marker_pixels", type=int, default=6)
    parser.add_argument(
        "--obj_up_axis",
        choices=["y", "z"],
        default="y",
        help=(
            "Coordinate convention for the primary OBJ exports. 'y' applies "
            "the proper rotation (x,y,z)->(x,z,-y); model data stays Z-up."
        ),
    )
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def set_metrics(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "iou": safe_ratio(tp, tp + fp + fn),
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def mean_rows(rows: List[dict], keys: Iterable[str]) -> dict:
    return {
        key: float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
        for key in keys
    }


def iter_pair_chunks(num_vertices: int, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield every pair ``u < v`` exactly once without materializing N squared."""
    if num_vertices < 2:
        return
    buffer = np.empty((chunk_size, 2), dtype=np.int64)
    filled = 0
    for u in range(num_vertices - 1):
        v = u + 1
        while v < num_vertices:
            take = min(num_vertices - v, chunk_size - filled)
            buffer[filled : filled + take, 0] = u
            buffer[filled : filled + take, 1] = np.arange(
                v, v + take, dtype=np.int64
            )
            filled += take
            v += take
            if filled == chunk_size:
                yield buffer.copy()
                filled = 0
    if filled:
        yield buffer[:filled].copy()


def match_token_features(
    token_coords: torch.Tensor,
    token_features: torch.Tensor,
    vertex_coords: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    """Select final vertex-token features by exact voxel coordinate."""
    if len(vertex_coords) == 0:
        return token_features.new_empty((0, token_features.shape[1]))
    token_keys = voxel_keys(token_coords.long(), resolution)
    vertex_keys = voxel_keys(vertex_coords.long(), resolution)
    sorted_keys, order = torch.sort(token_keys)
    if len(sorted_keys) > 1 and torch.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("Final decoder tokens contain duplicate voxel coordinates.")
    positions = torch.searchsorted(sorted_keys, vertex_keys)
    valid = positions < len(sorted_keys)
    safe_positions = positions.clamp(max=max(0, len(sorted_keys) - 1))
    valid &= sorted_keys[safe_positions] == vertex_keys
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum().item())} predicted vertices have no final token."
        )
    return token_features[order[safe_positions]]


def score_all_vertex_pairs(
    edge_head: torch.nn.Module,
    vertex_features: torch.Tensor,
    threshold: float,
    chunk_size: int,
    amp_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Apply sigmoid(head([u,v]) + head([v,u])) to every unordered pair."""
    accepted_edges: List[np.ndarray] = []
    accepted_probabilities: List[np.ndarray] = []
    num_vertices = int(vertex_features.shape[0])
    total_pairs = num_vertices * (num_vertices - 1) // 2
    for pairs_np in iter_pair_chunks(num_vertices, chunk_size):
        pairs = torch.from_numpy(pairs_np).to(
            device=vertex_features.device, dtype=torch.long
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            uv = torch.cat(
                [vertex_features[pairs[:, 0]], vertex_features[pairs[:, 1]]],
                dim=1,
            )
            logits = edge_head(uv).reshape(-1)
            del uv
            vu = torch.cat(
                [vertex_features[pairs[:, 1]], vertex_features[pairs[:, 0]]],
                dim=1,
            )
            logits = logits + edge_head(vu).reshape(-1)
            del vu
        probabilities = torch.sigmoid(logits.float())
        keep = probabilities >= float(threshold)
        if keep.any():
            accepted_edges.append(pairs[keep].cpu().numpy().astype(np.int64))
            accepted_probabilities.append(
                probabilities[keep].cpu().numpy().astype(np.float32)
            )
    if accepted_edges:
        return (
            np.concatenate(accepted_edges, axis=0),
            np.concatenate(accepted_probabilities, axis=0),
            total_pairs,
        )
    return (
        np.empty((0, 2), dtype=np.int64),
        np.empty((0,), dtype=np.float32),
        total_pairs,
    )


def build_graph(
    num_vertices: int,
    edges: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_vertices)))
    if probabilities is None:
        graph.add_edges_from((int(u), int(v)) for u, v in edges)
    else:
        graph.add_edges_from(
            (int(u), int(v), {"probability": float(probability)})
            for (u, v), probability in zip(edges, probabilities)
        )
    return graph


def remap_edges_after_dvert_clustering(
    edges: np.ndarray,
    probabilities: np.ndarray,
    prediction: torch.Tensor,
    merge_result: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remap pre-clustering edge endpoints and return a simple deduplicated graph.

    The edge head indexes vertices in ``flatnonzero(prediction)`` order.  Every
    predicted vertex inside a high-d_vert component maps to that component's
    continuous centroid.  Predicted vertices outside high-d_vert support remain
    separate.  High-d_vert components with no predicted vertex are omitted.

    If several original edges collapse onto the same undirected merged edge,
    only the maximum predicted probability is retained.  Collapsed self-edges
    are discarded.
    """
    prediction_np = prediction.detach().reshape(-1).bool().cpu().numpy()
    predicted_rows = np.flatnonzero(prediction_np).astype(np.int64)
    component_id = np.asarray(merge_result["component_id"], dtype=np.int64)
    high_mask = np.asarray(merge_result["high_mask"], dtype=np.bool_)
    unmatched_rows = np.asarray(merge_result["unmatched_rows"], dtype=np.int64)
    component_to_centroid = np.asarray(
        merge_result["component_to_centroid_index"], dtype=np.int64
    )
    num_cluster_centroids = int(len(merge_result["representative_rows"]))

    unmatched_index = {
        int(row): num_cluster_centroids + index
        for index, row in enumerate(unmatched_rows.tolist())
    }
    old_to_merged = np.empty(len(predicted_rows), dtype=np.int64)
    for old_index, support_row in enumerate(predicted_rows.tolist()):
        if high_mask[support_row]:
            component = int(component_id[support_row])
            merged_index = int(component_to_centroid[component])
            if merged_index < 0:
                raise RuntimeError(
                    "Predicted high-d_vert vertex maps to an omitted component."
                )
        else:
            merged_index = unmatched_index[int(support_row)]
        old_to_merged[old_index] = merged_index

    edge_probability_by_pair: Dict[Tuple[int, int], float] = {}
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if len(edges) != len(probabilities):
        raise ValueError("Edge and edge-probability lengths differ.")
    for (old_u, old_v), probability in zip(edges, probabilities):
        u = int(old_to_merged[int(old_u)])
        v = int(old_to_merged[int(old_v)])
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        edge_probability_by_pair[pair] = max(
            edge_probability_by_pair.get(pair, -math.inf), float(probability)
        )

    sorted_pairs = sorted(edge_probability_by_pair)
    remapped_edges = np.asarray(sorted_pairs, dtype=np.int64).reshape(-1, 2)
    remapped_probabilities = np.asarray(
        [edge_probability_by_pair[pair] for pair in sorted_pairs],
        dtype=np.float32,
    )
    return remapped_edges, remapped_probabilities, old_to_merged


def clique_faces(graph: nx.Graph) -> Iterator[Tuple[int, int, int]]:
    """Enumerate each three-clique once in deterministic ``u < v < w`` order."""
    emitted: set[Tuple[int, int, int]] = set()
    for endpoint_a, endpoint_b in graph.edges():
        u, v = sorted((int(endpoint_a), int(endpoint_b)))
        common = set(graph[u]).intersection(graph[v])
        for w in sorted(int(value) for value in common if int(value) > v):
            face = tuple(sorted((u, v, w)))
            if len(set(face)) != 3 or face in emitted:
                continue
            emitted.add(face)
            yield face


def repair_mesh_faces_with_trimesh(
    vertices: np.ndarray,
    faces: Iterable[Sequence[int]],
) -> np.ndarray:
    """Make clique-face winding/normals consistent without processing topology."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces_array = np.asarray(list(faces), dtype=np.int64).reshape(-1, 3)
    if len(faces_array) == 0:
        return faces_array
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces_array,
        process=False,
    )
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    repaired = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    if len(repaired) != len(faces_array):
        raise RuntimeError(
            "Trimesh winding repair unexpectedly changed the face count."
        )
    return repaired


def world_points(coords: np.ndarray, resolution: int) -> np.ndarray:
    return (np.asarray(coords, dtype=np.float32) + 0.5) / float(resolution) - 0.5


def points_for_obj(points: np.ndarray, up_axis: str) -> np.ndarray:
    """Convert canonical TRELLIS Z-up points to the requested OBJ convention."""
    points = np.asarray(points).reshape(-1, 3)
    if up_axis == "z":
        return points
    if up_axis == "y":
        # Proper -90-degree rotation around X: det(R)=+1, so handedness and
        # the existing face winding are preserved.
        return np.stack(
            [points[:, 0], points[:, 2], -points[:, 1]], axis=1
        )
    raise ValueError(f"Unsupported OBJ up axis: {up_axis}")


def coordinate_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def coordinate_edge_set(
    vertex_coords: np.ndarray,
    edges: np.ndarray,
    resolution: int,
) -> set[Tuple[int, int]]:
    keys = coordinate_keys(vertex_coords, resolution)
    output = set()
    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        a, b = int(keys[int(u)]), int(keys[int(v)])
        output.add((min(a, b), max(a, b)))
    return output


def write_edge_graph_obj(
    path: Path,
    points: np.ndarray,
    edges: np.ndarray,
    comment: str,
    up_axis: str,
) -> None:
    unique_edges = sorted(
        {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            if int(u) != int(v)
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {comment}\n")
        handle.write(f"# Coordinate convention: {up_axis.upper()}-up\n")
        for x, y, z in points_for_obj(points, up_axis):
            handle.write(f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
        for u, v in unique_edges:
            handle.write(f"l {int(u) + 1} {int(v) + 1}\n")


def write_clique_mesh_obj(
    path: Path,
    points: np.ndarray,
    graph: nx.Graph,
    vertex_coordinate_keys: Sequence[int],
    reference_triangle_keys: set[Tuple[int, int, int]] | None = None,
    collect_triangle_keys: bool = False,
    up_axis: str = "z",
) -> Tuple[int, int, set[Tuple[int, int, int]]]:
    """Repair and write all graph three-cliques, optionally comparing keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    triangle_count = 0
    reference_matches = 0
    collected: set[Tuple[int, int, int]] = set()
    keys = np.asarray(vertex_coordinate_keys, dtype=np.int64)
    repaired_faces = repair_mesh_faces_with_trimesh(points, clique_faces(graph))
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# One face per undirected predicted-graph 3-clique. "
            "Face winding and normals repaired with trimesh.\n"
        )
        handle.write(f"# Coordinate convention: {up_axis.upper()}-up\n")
        for x, y, z in points_for_obj(points, up_axis):
            handle.write(f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
        for u, v, w in repaired_faces:
            u, v, w = int(u), int(v), int(w)
            handle.write(f"f {u + 1} {v + 1} {w + 1}\n")
            triangle_count += 1
            triangle_key = tuple(
                sorted((int(keys[u]), int(keys[v]), int(keys[w])))
            )
            if collect_triangle_keys:
                collected.add(triangle_key)
            if reference_triangle_keys is not None and triangle_key in reference_triangle_keys:
                reference_matches += 1
    return triangle_count, reference_matches, collected


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    resolution = 512
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model evaluation and vertex rendering.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if args.edge_pair_batch_size <= 0:
        raise ValueError("edge_pair_batch_size must be positive.")
    for name, value in (
        ("vertex_threshold", args.vertex_threshold),
        ("rollout_threshold", args.rollout_threshold),
        ("edge_threshold", args.edge_threshold),
        ("dvert_threshold", args.dvert_threshold),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0,1).")
    if args.rollout_threshold > args.vertex_threshold:
        raise ValueError("rollout_threshold must not exceed vertex_threshold.")
    if args.num_visualizations < 0 or args.vertex_marker_pixels <= 0:
        raise ValueError("Visualization counts must be non-negative and markers positive.")

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
        raise ValueError("The checkpoint decoder has no hierarchical vertex head.")
    if not decoder_args.get("pred_edge", False):
        raise ValueError("The checkpoint decoder has no edge connection head.")
    if decoder_args.get("pred_subdiv", True):
        raise ValueError("This evaluator requires supplied support and pred_subdiv=False.")

    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    distance_transform = str(dataset_args.get("distance_transform", "none"))
    configured_resolutions = [int(value) for value in dataset_args.get("resolutions", [])]
    if configured_resolutions and resolution not in configured_resolutions:
        raise ValueError(f"R512 is not configured in {configured_resolutions}.")
    dataset_args["resolution"] = resolution
    dataset_args.pop("resolutions", None)
    dataset_args.pop("instances_path", None)
    data_dir = vertex_eval.build_data_dir(
        root, args.split, resolution, dataset_args, args
    )
    dataset = datasets.SparseVoxelTriangleFieldDataset(
        json.dumps(data_dir), **dataset_args
    )
    if args.instances is not None:
        vertex_eval.restrict_dataset_instances(dataset, args.instances.resolve())
    vertex_eval.limit_dataset_instances(dataset, args.max_eval_samples)
    if len(dataset) == 0:
        raise RuntimeError("No evaluation instances remain after filtering.")

    step = vertex_eval.find_checkpoint_step(
        run_dir, args.ckpt, args.ema_rate
    )
    default_name = (
        f"eval_qem_vertex_edge_dvert_cluster_3clique_{args.split}_step{step:07d}"
        f"_v{args.vertex_threshold:.2f}_e{args.edge_threshold:.2f}"
        f"_dv{args.dvert_threshold:.2f}_c{args.cluster_connectivity}"
        f"_up{args.obj_up_axis.upper()}"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / default_name / "resolution_512"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(dataset, flush=True)
    print(f"Checkpoint step: {step}", flush=True)
    print(f"EMA rate: {args.ema_rate}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(
        f"Vertex threshold: {args.vertex_threshold}; recursive candidate threshold: "
        f"{args.rollout_threshold}",
        flush=True,
    )
    print(
        f"Edge threshold: {args.edge_threshold}; all unordered predicted-vertex "
        f"pairs; GPU chunk size: {args.edge_pair_batch_size}",
        flush=True,
    )
    print(
        "Post-edge centroid merge: semantic d_vert > "
        f"{args.dvert_threshold}; {args.cluster_connectivity}-connectivity",
        flush=True,
    )
    print(
        f"Primary OBJ coordinate convention: {args.obj_up_axis.upper()}-up; "
        "canonical Z-up copies are also saved",
        flush=True,
    )

    model_dict = vertex_eval.load_models(cfg, run_dir, step, args.ema_rate)
    encoder = model_dict["encoder"]
    decoder = model_dict["decoder"]
    if not getattr(decoder, "vertex_logits_are_child_tokens", False):
        raise ValueError("This evaluator requires explicit hierarchical child tokens.")
    edge_head = getattr(decoder, "edge_connection_head", None)
    if edge_head is None:
        raise ValueError("Decoder is missing edge_connection_head.")
    # Run FP32-master-parameter checkpoints the same way they were trained.
    # FlashAttention accepts FP16/BF16 QKV only, and flex_gemm now casts its
    # invocation-local weights to match autocast sparse activations.
    amp_enabled = str(cfg["trainer"]["args"].get("fp16_mode", "")).lower() == "amp"
    print(
        f"Evaluation autocast: {'FP16 AMP' if amp_enabled else 'disabled'}",
        flush=True,
    )

    indexed_dataset = vertex_eval.IndexedTriangleFieldDataset(dataset)
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
        vertex_eval.VertexVoxelVisualizer(
            resolution, args.image_resolution, args.ssaa
        )
        if visualization_indices
        else None
    )
    visualization_root = output_dir / "visualizations"
    mesh_root = output_dir / "meshes"
    aggregate_gt: List[torch.Tensor] = []
    aggregate_pred: List[torch.Tensor] = []
    aggregate_error: List[torch.Tensor] = []
    visualization_manifest: List[dict] = []
    per_mesh_rows: List[dict] = []

    raw_vertex_totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    vertex_totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    edge_totals = {"tp": 0, "fp": 0, "fn": 0}
    triangle_totals = {"tp": 0, "fp": 0, "fn": 0}
    total_scored_pairs = 0
    total_raw_predicted_edges = 0
    total_predicted_edges = 0
    total_predicted_triangles = 0

    for data in tqdm(loader, desc="R512 vertex -> edge -> 3-clique mesh"):
        data = recursive_to_device(data, torch.device("cuda"), non_blocking=True)
        instances = data["instance"]
        dataset_indices = data["dataset_index"].reshape(-1)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            z = encoder(data["x"], sample_posterior=args.sample_posterior)
            y, vertex_logits, final_vertex_tokens = decoder(
                z,
                return_vertex=True,
                return_vertex_features=True,
                resolutions=data["resolution"],
                vertex_threshold=args.rollout_threshold,
            )
        if not torch.equal(y.coords, data["target"].coords):
            raise ValueError("Decoded triangle support does not match the target support.")
        if not torch.equal(data["vertex_occupancy"].coords, y.coords):
            raise ValueError("Vertex targets are not aligned with decoded support.")

        batch_ids = y.coords[:, 0].long()
        gt_occupancy = data["vertex_occupancy"].feats.reshape(-1).bool()
        if distance_transform == "minus_one_one":
            dvert_unit_all = (y.feats[:, 1].float() + 1.0) * 0.5
        else:
            dvert_unit_all = y.feats[:, 1].float()
        dvert_unit_all = dvert_unit_all.clamp(0.0, 1.0)
        for batch_index, instance in enumerate(instances):
            support_mask = batch_ids.eq(batch_index)
            support_coords = y.coords[support_mask, 1:4].long()
            target = gt_occupancy[support_mask]
            sample_dvert = dvert_unit_all[support_mask]
            stages = compute_vertex_hierarchy_predictions(
                vertex_logits,
                support_coords,
                target,
                batch_index,
                resolution,
                threshold=args.vertex_threshold,
            )
            recursive_probability = stages[-1]["recursive_score"]
            prediction = recursive_probability >= args.vertex_threshold
            pred_coords = support_coords[prediction]
            pred_probabilities = recursive_probability[prediction]
            raw_tp, raw_fp, raw_fn, raw_tn = vertex_eval.confusion_counts(
                prediction, target
            )
            for key, value in zip(
                ("tp", "fp", "fn", "tn"),
                (raw_tp, raw_fp, raw_fn, raw_tn),
            ):
                raw_vertex_totals[key] += value

            token_slice = final_vertex_tokens.layout[batch_index]
            token_coords = final_vertex_tokens.coords[token_slice, 1:4].long()
            token_features = final_vertex_tokens.feats[token_slice]
            pred_features = match_token_features(
                token_coords,
                token_features,
                pred_coords,
                resolution,
            )
            raw_pred_edges, raw_pred_edge_probabilities, scored_pairs = score_all_vertex_pairs(
                edge_head,
                pred_features,
                args.edge_threshold,
                args.edge_pair_batch_size,
                amp_enabled,
            )
            raw_pred_coords_np = pred_coords.cpu().numpy().astype(np.int32)
            raw_pred_probs_np = pred_probabilities.cpu().numpy().astype(np.float32)
            raw_pred_points = world_points(raw_pred_coords_np, resolution)

            # Edge prediction deliberately happens before centroid clustering.
            # The predicted graph is then contracted by the d_vert components.
            merge_result = cluster_and_merge_predictions(
                support_coords,
                sample_dvert,
                prediction,
                resolution,
                args.dvert_threshold,
                args.cluster_connectivity,
                create_empty_components=False,
            )
            merged_prediction = merge_result["merged_mask"]
            representative_rows = np.asarray(
                merge_result["representative_rows"], dtype=np.int64
            )
            unmatched_rows = np.asarray(
                merge_result["unmatched_rows"], dtype=np.int64
            )
            merged_support_rows = np.concatenate(
                [representative_rows, unmatched_rows], axis=0
            )
            merged_coords_np = (
                support_coords[
                    torch.from_numpy(merged_support_rows).to(support_coords.device)
                ]
                .cpu()
                .numpy()
                .astype(np.int32)
            )
            merged_points = np.asarray(
                merge_result["continuous_points"], dtype=np.float32
            ).reshape(-1, 3)
            if len(merged_coords_np) != len(merged_points):
                raise RuntimeError("Snapped and continuous merged vertices differ in length.")
            pred_edges, pred_edge_probabilities, old_to_merged = (
                remap_edges_after_dvert_clustering(
                    raw_pred_edges,
                    raw_pred_edge_probabilities,
                    prediction,
                    merge_result,
                )
            )
            predicted_support_rows = np.flatnonzero(
                prediction.detach().cpu().numpy().astype(np.bool_)
            )
            merged_recursive_probabilities = np.zeros(
                len(merged_points), dtype=np.float32
            )
            for old_index, support_row in enumerate(predicted_support_rows.tolist()):
                merged_index = int(old_to_merged[old_index])
                merged_recursive_probabilities[merged_index] = max(
                    float(merged_recursive_probabilities[merged_index]),
                    float(raw_pred_probs_np[old_index]),
                )
            merged_vertex_created_without_prediction = (
                merged_recursive_probabilities == 0.0
            )
            pred_graph = build_graph(
                len(merged_points), pred_edges, pred_edge_probabilities
            )

            tp, fp, fn, tn = vertex_eval.confusion_counts(
                merged_prediction, target
            )
            for key, value in zip(("tp", "fp", "fn", "tn"), (tp, fp, fn, tn)):
                vertex_totals[key] += value

            gt_coords_np = data["qem_vertex_coords"][batch_index].cpu().numpy().astype(np.int32)
            gt_edges_np = data["qem_edges"][batch_index].cpu().numpy().astype(np.int64)
            gt_points = world_points(gt_coords_np, resolution)
            gt_graph = build_graph(len(gt_coords_np), gt_edges_np)
            pred_edge_keys = coordinate_edge_set(
                merged_coords_np, pred_edges, resolution
            )
            gt_edge_keys = coordinate_edge_set(gt_coords_np, gt_edges_np, resolution)
            edge_tp = len(pred_edge_keys & gt_edge_keys)
            edge_fp = len(pred_edge_keys - gt_edge_keys)
            edge_fn = len(gt_edge_keys - pred_edge_keys)
            edge_counts = set_metrics(edge_tp, edge_fp, edge_fn)
            for key in edge_totals:
                edge_totals[key] += int(edge_counts[key])

            instance_dir = mesh_root / str(instance)
            instance_dir.mkdir(parents=True, exist_ok=True)
            edge_exports = (
                (
                    "predicted_edge_graph_before_centroid_merge.obj",
                    raw_pred_points,
                    raw_pred_edges,
                    "Raw predicted vertices and edges before d_vert centroid merging",
                ),
                (
                    "predicted_edge_graph.obj",
                    merged_points,
                    pred_edges,
                    "d_vert-centroid vertices and deduplicated remapped edge graph",
                ),
                (
                    "gt_qem_edge_graph.obj",
                    gt_points,
                    gt_edges_np,
                    "GT QEM vertex-edge graph",
                ),
            )
            for filename, points, edges, comment in edge_exports:
                write_edge_graph_obj(
                    instance_dir / filename,
                    points,
                    edges,
                    comment,
                    args.obj_up_axis,
                )
                if args.obj_up_axis != "z":
                    z_up_name = f"{Path(filename).stem}_z_up.obj"
                    write_edge_graph_obj(
                        instance_dir / z_up_name,
                        points,
                        edges,
                        comment,
                        "z",
                    )
            gt_triangle_count, _, gt_triangle_keys = write_clique_mesh_obj(
                instance_dir / "gt_qem_3clique_mesh.obj",
                gt_points,
                gt_graph,
                coordinate_keys(gt_coords_np, resolution),
                collect_triangle_keys=True,
                up_axis=args.obj_up_axis,
            )
            (
                pred_triangle_count,
                triangle_tp,
                _,
            ) = write_clique_mesh_obj(
                instance_dir / "predicted_3clique_mesh.obj",
                merged_points,
                pred_graph,
                coordinate_keys(merged_coords_np, resolution),
                reference_triangle_keys=gt_triangle_keys,
                up_axis=args.obj_up_axis,
            )
            if args.obj_up_axis != "z":
                write_clique_mesh_obj(
                    instance_dir / "gt_qem_3clique_mesh_z_up.obj",
                    gt_points,
                    gt_graph,
                    coordinate_keys(gt_coords_np, resolution),
                    up_axis="z",
                )
                write_clique_mesh_obj(
                    instance_dir / "predicted_3clique_mesh_z_up.obj",
                    merged_points,
                    pred_graph,
                    coordinate_keys(merged_coords_np, resolution),
                    up_axis="z",
                )
            triangle_fp = pred_triangle_count - triangle_tp
            triangle_fn = gt_triangle_count - triangle_tp
            triangle_counts = set_metrics(triangle_tp, triangle_fp, triangle_fn)
            for key in triangle_totals:
                triangle_totals[key] += int(triangle_counts[key])

            np.savez_compressed(
                instance_dir / "predicted_vertex_edge_graph.npz",
                vertex_voxel_coords=merged_coords_np,
                vertex_world_positions=merged_points,
                vertex_obj_positions=points_for_obj(
                    merged_points, args.obj_up_axis
                ).astype(np.float32),
                vertex_recursive_probability=merged_recursive_probabilities,
                vertex_created_without_prediction=merged_vertex_created_without_prediction,
                raw_vertex_voxel_coords=raw_pred_coords_np,
                raw_vertex_world_positions=raw_pred_points.astype(np.float32),
                raw_vertex_recursive_probability=raw_pred_probs_np,
                raw_edges=raw_pred_edges,
                raw_edge_probability=raw_pred_edge_probabilities,
                raw_vertex_to_merged_vertex=old_to_merged,
                edges=pred_edges,
                edge_probability=pred_edge_probabilities,
                vertex_threshold=np.float32(args.vertex_threshold),
                edge_threshold=np.float32(args.edge_threshold),
                dvert_threshold=np.float32(args.dvert_threshold),
                cluster_connectivity=np.int32(args.cluster_connectivity),
                obj_up_axis=np.asarray(args.obj_up_axis),
                cluster_representative_support_rows=representative_rows,
                unmatched_predicted_support_rows=unmatched_rows,
                cluster_sizes=np.asarray(merge_result["component_sizes"], dtype=np.int32),
                cluster_prediction_counts=np.asarray(
                    merge_result["component_prediction_counts"], dtype=np.int32
                ),
                resolution=np.int32(resolution),
            )

            vertex_counts = vertex_eval.confusion_metrics(tp, fp, fn, tn)
            raw_vertex_counts = vertex_eval.confusion_metrics(
                raw_tp, raw_fp, raw_fn, raw_tn
            )
            row = {
                "instance": str(instance),
                "resolution": resolution,
                "active_triangle_voxels": int(len(support_coords)),
                "gt_vertices": int(target.sum().item()),
                "raw_predicted_vertices": int(prediction.sum().item()),
                "predicted_vertices": int(merged_prediction.sum().item()),
                "vertices_collapsed_away": int(
                    merge_result["num_predictions_collapsed_away"]
                ),
                "vertices_created_from_empty_dvert_components": int(
                    merge_result["num_empty_components_created"]
                ),
                "empty_dvert_components_skipped": int(
                    merge_result["num_empty_components_skipped"]
                ),
                "high_dvert_components": int(merge_result["num_components"]),
                "raw_vertex_tp": raw_tp,
                "raw_vertex_fp": raw_fp,
                "raw_vertex_fn": raw_fn,
                "raw_vertex_tn": raw_tn,
                "raw_vertex_precision": raw_vertex_counts["precision"],
                "raw_vertex_recall": raw_vertex_counts["recall"],
                "raw_vertex_iou": raw_vertex_counts["iou"],
                "raw_vertex_f1": raw_vertex_counts["f1"],
                "vertex_tp": tp,
                "vertex_fp": fp,
                "vertex_fn": fn,
                "vertex_tn": tn,
                "vertex_precision": vertex_counts["precision"],
                "vertex_recall": vertex_counts["recall"],
                "vertex_iou": vertex_counts["iou"],
                "vertex_f1": vertex_counts["f1"],
                "all_predicted_vertex_pairs_scored": int(scored_pairs),
                "raw_predicted_edges": int(len(raw_pred_edges)),
                "gt_edges": int(len(gt_edge_keys)),
                "predicted_edges": int(len(pred_edge_keys)),
                "edge_tp": edge_tp,
                "edge_fp": edge_fp,
                "edge_fn": edge_fn,
                "edge_precision": edge_counts["precision"],
                "edge_recall": edge_counts["recall"],
                "edge_iou": edge_counts["iou"],
                "edge_f1": edge_counts["f1"],
                "gt_3cliques": gt_triangle_count,
                "predicted_3cliques": pred_triangle_count,
                "triangle_tp": triangle_tp,
                "triangle_fp": triangle_fp,
                "triangle_fn": triangle_fn,
                "triangle_precision": triangle_counts["precision"],
                "triangle_recall": triangle_counts["recall"],
                "triangle_iou": triangle_counts["iou"],
                "triangle_f1": triangle_counts["f1"],
            }
            per_mesh_rows.append(row)
            total_scored_pairs += int(scored_pairs)
            total_raw_predicted_edges += int(len(raw_pred_edges))
            total_predicted_edges += int(len(pred_edge_keys))
            total_predicted_triangles += int(pred_triangle_count)

            dataset_index = int(dataset_indices[batch_index].item())
            if visualizer is None or dataset_index not in visualization_indices:
                continue
            images = build_vertex_renderings(
                visualizer.renderer,
                support_coords,
                target,
                merged_prediction,
                resolution,
                confidence=recursive_probability,
                minimum_marker_pixels=args.vertex_marker_pixels,
                include_mistakes=True,
            )
            gt_sheet = add_image_label(
                images["gt"], f"GT QEM VERTICES: {int(target.sum().item()):,}"
            ).cpu()
            pred_sheet = add_image_label(
                images["prediction"],
                f"CENTROID-MERGED VERTICES: {int(merged_prediction.sum().item()):,}  "
                f"VERTEX={args.vertex_threshold:.2f} D_VERT={args.dvert_threshold:.2f}",
            ).cpu()
            error_sheet = add_image_label(
                images["error"],
                f"TP GREEN {tp:,} | FP RED {fp:,} | FN BLUE {fn:,}",
            ).cpu()
            visual_dir = visualization_root / str(instance)
            visual_dir.mkdir(parents=True, exist_ok=True)
            vertex_eval.tensor_to_image(gt_sheet).save(visual_dir / "gt_vertex_voxels.jpg")
            vertex_eval.tensor_to_image(pred_sheet).save(
                visual_dir / f"pred_vertex_voxels_t{args.vertex_threshold:.2f}.jpg"
            )
            vertex_eval.tensor_to_image(error_sheet).save(
                visual_dir / "vertex_error_overlay.jpg"
            )
            vertex_eval.tensor_to_image(images["mistakes"]).save(
                visual_dir / "vertex_mistakes_only.jpg"
            )
            vertex_eval.tensor_to_image(images["support_context"]).save(
                visual_dir / "vertex_error_with_triangle_support.jpg"
            )
            vertex_eval.tensor_to_image(images["confidence"]).save(
                visual_dir / "vertex_recursive_confidence.jpg"
            )
            visualizer.comparison(
                gt_sheet,
                pred_sheet,
                error_sheet,
                str(instance),
                args.vertex_threshold,
            ).save(visual_dir / "gt_vs_pred_vertex_voxels.jpg")
            vertex_eval.save_vertex_point_cloud(
                visual_dir / "gt_qem_vertex_point_cloud.ply",
                support_coords[target],
                resolution,
                GT_COLOR,
            )
            vertex_eval.save_vertex_point_cloud(
                visual_dir / "predicted_vertex_point_cloud.ply",
                support_coords[merged_prediction],
                resolution,
                PRED_COLOR,
            )
            vertex_eval.save_vertex_point_cloud(
                visual_dir / "predicted_vertex_point_cloud_before_merge.ply",
                pred_coords,
                resolution,
                PRED_COLOR,
            )
            save_float_point_cloud(
                visual_dir / "predicted_vertex_centroids_after_merge.ply",
                merged_points,
                PRED_COLOR,
                "Exact d_vert-weighted centroid vertices after edge prediction",
            )
            aggregate_gt.append(gt_sheet)
            aggregate_pred.append(pred_sheet)
            aggregate_error.append(error_sheet)
            visualization_manifest.append(row)

    raw_vertex_micro = vertex_eval.confusion_metrics(**raw_vertex_totals)
    vertex_micro = vertex_eval.confusion_metrics(**vertex_totals)
    edge_micro = set_metrics(**edge_totals)
    triangle_micro = set_metrics(**triangle_totals)
    macro_keys = [
        "vertex_precision",
        "vertex_recall",
        "vertex_iou",
        "vertex_f1",
        "edge_precision",
        "edge_recall",
        "edge_iou",
        "edge_f1",
        "triangle_precision",
        "triangle_recall",
        "triangle_iou",
        "triangle_f1",
    ]
    metrics = {
        "checkpoint_step": step,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "resolution": resolution,
        "num_instances": len(per_mesh_rows),
        "posterior_mode": "sampled" if args.sample_posterior else "mean",
        "vertex_threshold": args.vertex_threshold,
        "rollout_threshold": args.rollout_threshold,
        "edge_threshold": args.edge_threshold,
        "dvert_threshold": args.dvert_threshold,
        "cluster_connectivity": args.cluster_connectivity,
        "primary_obj_up_axis": args.obj_up_axis,
        "obj_rotation": (
            "Y-up uses proper -90-degree X rotation (x,y,z)->(x,z,-y); "
            "canonical model/QEM coordinates remain Z-up"
        ),
        "edge_pair_batch_size": args.edge_pair_batch_size,
        "edge_candidate_mode": (
            "all unordered pairs of raw predicted final vertices; score edges "
            "before d_vert centroid merging"
        ),
        "edge_probability": "sigmoid(head([u,v]) + head([v,u]))",
        "vertex_postprocess": (
            "merge predicted vertices per connected high-d_vert component onto "
            "the d_vert-weighted centroid; omit components with no predicted vertex"
        ),
        "graph_cleanup": (
            "remap endpoints, remove self-edges, deduplicate undirected edges by "
            "maximum probability, and emit each canonical 3-clique face once"
        ),
        "metric_coordinate_policy": (
            "use the active support voxel nearest each continuous centroid for "
            "coordinate-aligned vertex/edge/triangle metrics"
        ),
        "mesh_extraction": (
            "one OBJ face per unique graph 3-clique, followed by "
            "trimesh.repair.fix_winding and trimesh.repair.fix_normals"
        ),
        "raw_vertex_micro_before_centroid_merge": raw_vertex_micro,
        "vertex_micro": vertex_micro,
        "edge_micro": edge_micro,
        "triangle_3clique_micro": triangle_micro,
        "macro": mean_rows(per_mesh_rows, macro_keys),
        "total_scored_predicted_vertex_pairs": total_scored_pairs,
        "total_raw_predicted_edges": total_raw_predicted_edges,
        "total_predicted_edges": total_predicted_edges,
        "total_predicted_3cliques": total_predicted_triangles,
        "num_visualizations": len(visualization_manifest),
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
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(visualization_manifest, indent=2), encoding="utf-8"
    )
    vertex_eval.save_image_grid(
        aggregate_gt, output_dir / "vertex_gt_grid.jpg"
    )
    vertex_eval.save_image_grid(
        aggregate_pred, output_dir / "vertex_prediction_grid.jpg"
    )
    vertex_eval.save_image_grid(
        aggregate_error, output_dir / "vertex_error_grid.jpg"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
