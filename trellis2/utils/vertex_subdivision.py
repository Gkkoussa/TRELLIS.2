"""Utilities for hierarchical eight-child vertex occupancy prediction."""

from typing import Dict, List

import torch


def voxel_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    """Encode integer xyz voxel coordinates as unique int64 keys."""
    coords = coords.to(dtype=torch.long)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def child_offsets(device: torch.device) -> torch.Tensor:
    """Return offsets in TRELLIS subdivision order: x + 2*y + 4*z."""
    return torch.tensor(
        [
            [child_index % 2, (child_index // 2) % 2, child_index // 4]
            for child_index in range(8)
        ],
        device=device,
        dtype=torch.long,
    )


def _match_rows(
    query_keys: torch.Tensor,
    source_keys: torch.Tensor,
    context: str,
) -> torch.Tensor:
    """Return source row indices for exact key matches without an NxM comparison."""
    if len(source_keys) == 0:
        raise ValueError(f'{context}: source coordinate set is empty.')
    sorted_keys, order = torch.sort(source_keys)
    if len(sorted_keys) > 1 and torch.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError(f'{context}: source coordinate set contains duplicates.')
    positions = torch.searchsorted(sorted_keys, query_keys)
    valid = positions < len(sorted_keys)
    matched = torch.zeros_like(valid)
    matched[valid] = sorted_keys[positions[valid]] == query_keys[valid]
    if not matched.all():
        raise ValueError(
            f'{context}: {int((~matched).sum().item())} coordinates have no exact match.'
        )
    return order[positions]


def _match_optional_rows(
    query_keys: torch.Tensor,
    source_keys: torch.Tensor,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source rows plus a mask for query keys present in the source."""
    if len(source_keys) == 0:
        return torch.zeros_like(query_keys), torch.zeros_like(
            query_keys, dtype=torch.bool
        )
    sorted_keys, order = torch.sort(source_keys)
    if len(sorted_keys) > 1 and torch.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError(f'{context}: source coordinate set contains duplicates.')
    positions = torch.searchsorted(sorted_keys, query_keys)
    valid = positions < len(sorted_keys)
    safe_positions = positions.clamp(max=len(sorted_keys) - 1)
    valid &= sorted_keys[safe_positions] == query_keys
    return order[safe_positions], valid


@torch.no_grad()
def _compute_vertex_token_hierarchy_predictions(
    vertex_logits: List,
    final_support_xyz: torch.Tensor,
    final_vertex_target: torch.Tensor,
    batch_index: int,
    final_resolution: int,
    threshold: float,
) -> List[Dict[str, torch.Tensor]]:
    """Roll out explicit scalar child tokens produced by the attention decoder."""
    num_stages = len(vertex_logits)
    final_positive_xyz = final_support_xyz[final_vertex_target]
    offsets = child_offsets(final_support_xyz.device)
    stages = []
    previous_child_keys = None
    previous_recursive_score = None

    for stage_index, stage_logits in enumerate(vertex_logits):
        child_resolution = final_resolution // (
            2 ** (num_stages - stage_index - 1)
        )
        parent_resolution = child_resolution // 2
        sample_mask = stage_logits.coords[:, 0].long().eq(batch_index)
        candidate_xyz = stage_logits.coords[sample_mask, 1:4].long()
        candidate_logits = stage_logits.feats[sample_mask, 0].float()
        if len(candidate_xyz) == 0:
            raise ValueError(
                f'Vertex-token stage {stage_index} has no children for batch '
                f'sample {batch_index}.'
            )
        candidate_keys = voxel_keys(candidate_xyz, child_resolution)
        if len(torch.unique(candidate_keys)) != len(candidate_keys):
            raise ValueError(
                f'Vertex-token stage {stage_index} contains duplicate children.'
            )
        local_probability_raw = torch.sigmoid(candidate_logits)

        if stage_index == 0:
            parent_path_score = torch.ones_like(local_probability_raw)
        else:
            parent_keys = voxel_keys(candidate_xyz // 2, parent_resolution)
            previous_rows = _match_rows(
                parent_keys,
                previous_child_keys,
                f'vertex-token stage {stage_index} recursive parent mapping',
            )
            parent_path_score = previous_recursive_score[previous_rows]
        recursive_score_raw = torch.minimum(
            parent_path_score, local_probability_raw
        )

        downsample_factor = final_resolution // child_resolution
        positive_child_xyz = torch.unique(
            final_positive_xyz // downsample_factor,
            dim=0,
        )
        positive_child_keys = voxel_keys(positive_child_xyz, child_resolution)
        gt_raw = torch.isin(candidate_keys, positive_child_keys)

        # The final evaluator compares against every supplied final-resolution
        # triangle voxel.  A child omitted by recursive pruning receives score
        # zero, making it an explicit false negative rather than disappearing
        # from the metric denominator.
        if stage_index + 1 == num_stages:
            child_support_xyz = final_support_xyz
            child_support_keys = voxel_keys(child_support_xyz, child_resolution)
            candidate_rows, present = _match_optional_rows(
                child_support_keys,
                candidate_keys,
                f'vertex-token stage {stage_index} final support mapping',
            )
            local_probability = torch.zeros(
                len(child_support_xyz),
                device=local_probability_raw.device,
                dtype=local_probability_raw.dtype,
            )
            recursive_score = torch.zeros_like(local_probability)
            local_probability[present] = local_probability_raw[candidate_rows[present]]
            recursive_score[present] = recursive_score_raw[candidate_rows[present]]
            gt_child = final_vertex_target
        else:
            child_support_xyz = candidate_xyz
            child_support_keys = candidate_keys
            local_probability = local_probability_raw
            recursive_score = recursive_score_raw
            gt_child = gt_raw

        teacher_prediction = local_probability >= threshold
        recursive_prediction = recursive_score >= threshold

        raw_parent_xyz = candidate_xyz // 2
        unique_parent_xyz = torch.unique(raw_parent_xyz, dim=0)
        raw_parent_keys = voxel_keys(raw_parent_xyz, parent_resolution)
        positive_raw_rows = torch.nonzero(gt_raw, as_tuple=False).flatten()
        example_parent_xyz = (
            raw_parent_xyz[positive_raw_rows[0]]
            if len(positive_raw_rows) > 0
            else unique_parent_xyz[0]
        )
        example_child_xyz = example_parent_xyz[None] * 2 + offsets
        example_child_keys = voxel_keys(example_child_xyz, child_resolution)
        example_rows, example_present = _match_optional_rows(
            example_child_keys,
            candidate_keys,
            f'vertex-token stage {stage_index} example child mapping',
        )
        example_local_probability = torch.zeros(
            8,
            device=local_probability_raw.device,
            dtype=local_probability_raw.dtype,
        )
        example_recursive_score = torch.zeros_like(example_local_probability)
        example_local_probability[example_present] = local_probability_raw[
            example_rows[example_present]
        ]
        example_recursive_score[example_present] = recursive_score_raw[
            example_rows[example_present]
        ]
        example_gt = torch.isin(example_child_keys, positive_child_keys)

        stages.append({
            'stage_index': torch.tensor(stage_index, device=candidate_xyz.device),
            'parent_resolution': torch.tensor(parent_resolution, device=candidate_xyz.device),
            'child_resolution': torch.tensor(child_resolution, device=candidate_xyz.device),
            'parent_xyz': unique_parent_xyz,
            'child_support_xyz': child_support_xyz,
            'gt_child': gt_child,
            'local_probability': local_probability,
            'teacher_prediction': teacher_prediction,
            'recursive_score': recursive_score,
            'recursive_prediction': recursive_prediction,
            'example_parent_xyz': example_parent_xyz,
            'example_gt': example_gt,
            'example_local_probability': example_local_probability,
            'example_recursive_score': example_recursive_score,
            'example_teacher_prediction': example_local_probability >= threshold,
            'example_recursive_prediction': example_recursive_score >= threshold,
        })
        previous_child_keys = candidate_keys
        previous_recursive_score = recursive_score_raw

    if not torch.equal(stages[-1]['gt_child'], final_vertex_target):
        raise AssertionError('Final hierarchy target does not reproduce final QEM occupancy.')
    return stages


@torch.no_grad()
def compute_vertex_hierarchy_predictions(
    vertex_logits: List,
    final_support_xyz: torch.Tensor,
    final_vertex_target: torch.Tensor,
    batch_index: int,
    final_resolution: int,
    threshold: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """Compute GT, teacher-forced, and recursive predictions at every stage.

    Returned tensors are aligned with ``child_support_xyz`` at each stage.  The
    recursive score is the minimum child probability along the complete path;
    therefore ``recursive_score >= threshold`` is exactly equivalent to every
    parent-to-child decision on that path passing the same threshold.
    """
    if not vertex_logits:
        raise ValueError('The decoder returned no vertex-child stages.')
    if not (0.0 < threshold < 1.0):
        raise ValueError('threshold must be in (0, 1).')
    num_stages = len(vertex_logits)
    if final_resolution <= 0 or final_resolution % (2 ** num_stages) != 0:
        raise ValueError(
            f'Final resolution {final_resolution} must be divisible by '
            f'2 ** num_stages ({2 ** num_stages}).'
        )

    final_support_xyz = final_support_xyz.to(dtype=torch.long)
    final_vertex_target = final_vertex_target.reshape(-1).bool()
    if len(final_support_xyz) != len(final_vertex_target):
        raise ValueError('Final support and final vertex target lengths must match.')
    if len(final_support_xyz) == 0:
        raise ValueError('Final sparse support is empty.')
    final_positive_xyz = final_support_xyz[final_vertex_target]
    if len(final_positive_xyz) == 0:
        raise ValueError('Final QEM target contains no positive vertex voxels.')

    head_widths = [
        stage.feats.shape[1]
        if stage.feats.ndim == 2
        else -1
        for stage in vertex_logits
    ]
    if all(width == 1 for width in head_widths):
        return _compute_vertex_token_hierarchy_predictions(
            vertex_logits,
            final_support_xyz,
            final_vertex_target,
            batch_index,
            final_resolution,
            threshold,
        )
    if not all(width == 8 for width in head_widths):
        raise ValueError(
            'Vertex stages must consistently emit either one scalar per '
            f'explicit child or eight logits per parent; got widths {head_widths}.'
        )

    offsets = child_offsets(final_support_xyz.device)
    stages = []
    previous_child_keys = None
    previous_recursive_score = None

    for stage_index, stage_logits in enumerate(vertex_logits):
        if stage_logits.feats.ndim != 2 or stage_logits.feats.shape[1] != 8:
            raise ValueError(
                f'Vertex stage {stage_index} must emit [num_parents, 8], got '
                f'{tuple(stage_logits.feats.shape)}.'
            )
        parent_resolution = final_resolution // (2 ** (num_stages - stage_index))
        child_resolution = parent_resolution * 2

        sample_mask = stage_logits.coords[:, 0].long().eq(batch_index)
        parent_xyz = stage_logits.coords[sample_mask, 1:4].long()
        logits = stage_logits.feats[sample_mask].float()
        if len(parent_xyz) == 0:
            raise ValueError(
                f'Vertex stage {stage_index} has no parents for batch sample {batch_index}.'
            )
        parent_keys = voxel_keys(parent_xyz, parent_resolution)
        if len(torch.unique(parent_keys)) != len(parent_keys):
            raise ValueError(f'Vertex stage {stage_index} contains duplicate parents.')

        if stage_index + 1 < num_stages:
            next_stage = vertex_logits[stage_index + 1]
            next_mask = next_stage.coords[:, 0].long().eq(batch_index)
            child_support_xyz = next_stage.coords[next_mask, 1:4].long()
        else:
            child_support_xyz = final_support_xyz
        child_support_keys = voxel_keys(child_support_xyz, child_resolution)
        if len(torch.unique(child_support_keys)) != len(child_support_keys):
            raise ValueError(f'Vertex stage {stage_index} child support contains duplicates.')

        support_parent_xyz = child_support_xyz // 2
        support_parent_keys = voxel_keys(support_parent_xyz, parent_resolution)
        parent_rows = _match_rows(
            support_parent_keys,
            parent_keys,
            f'vertex stage {stage_index} child-to-parent mapping',
        )
        remainder = child_support_xyz % 2
        support_child_index = (
            remainder[:, 0] + 2 * remainder[:, 1] + 4 * remainder[:, 2]
        )
        local_probability = torch.sigmoid(logits)[parent_rows, support_child_index]

        downsample_factor = final_resolution // child_resolution
        positive_child_xyz = torch.unique(
            final_positive_xyz // downsample_factor,
            dim=0,
        )
        positive_child_keys = voxel_keys(positive_child_xyz, child_resolution)
        gt_child = torch.isin(child_support_keys, positive_child_keys)
        if int(gt_child.sum().item()) != len(positive_child_keys):
            raise ValueError(
                f'Vertex stage {stage_index}: some GT children are outside decoder support.'
            )
        positive_parent_xyz = torch.unique(positive_child_xyz // 2, dim=0)
        positive_parent_keys = voxel_keys(positive_parent_xyz, parent_resolution)
        gt_parent = torch.isin(parent_keys, positive_parent_keys)
        if int(gt_parent.sum().item()) != len(positive_parent_keys):
            raise ValueError(
                f'Vertex stage {stage_index}: some GT parents are outside decoder support.'
            )

        teacher_parent = (
            torch.ones_like(gt_parent)
            if stage_index == 0
            else gt_parent
        )
        teacher_prediction = (
            local_probability >= threshold
        ) & teacher_parent[parent_rows]

        if stage_index == 0:
            parent_path_score = torch.ones(
                len(parent_xyz),
                device=local_probability.device,
                dtype=local_probability.dtype,
            )
        else:
            previous_rows = _match_rows(
                parent_keys,
                previous_child_keys,
                f'vertex stage {stage_index} recursive parent mapping',
            )
            parent_path_score = previous_recursive_score[previous_rows]
        recursive_score = torch.minimum(
            parent_path_score[parent_rows],
            local_probability,
        )
        recursive_prediction = recursive_score >= threshold

        example_parent_row = int(torch.nonzero(gt_parent, as_tuple=False)[0, 0].item())
        example_child_xyz = parent_xyz[example_parent_row][None] * 2 + offsets
        example_child_keys = voxel_keys(example_child_xyz, child_resolution)
        example_gt = torch.isin(example_child_keys, positive_child_keys)
        example_local_probability = torch.sigmoid(logits[example_parent_row])
        example_recursive_score = torch.minimum(
            parent_path_score[example_parent_row],
            example_local_probability,
        )

        stages.append({
            'stage_index': torch.tensor(stage_index, device=parent_xyz.device),
            'parent_resolution': torch.tensor(parent_resolution, device=parent_xyz.device),
            'child_resolution': torch.tensor(child_resolution, device=parent_xyz.device),
            'parent_xyz': parent_xyz,
            'child_support_xyz': child_support_xyz,
            'gt_child': gt_child,
            'local_probability': local_probability,
            'teacher_prediction': teacher_prediction,
            'recursive_score': recursive_score,
            'recursive_prediction': recursive_prediction,
            'example_parent_xyz': parent_xyz[example_parent_row],
            'example_gt': example_gt,
            'example_local_probability': example_local_probability,
            'example_recursive_score': example_recursive_score,
            'example_teacher_prediction': example_local_probability >= threshold,
            'example_recursive_prediction': example_recursive_score >= threshold,
        })
        previous_child_keys = child_support_keys
        previous_recursive_score = recursive_score

    if not torch.equal(stages[-1]['gt_child'], final_vertex_target):
        raise AssertionError('Final hierarchy target does not reproduce final QEM occupancy.')
    return stages
