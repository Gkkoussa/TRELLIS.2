import argparse
import importlib
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

import o_voxel


def apply_transform(
    distances: np.ndarray,
    kind: str,
    d_max: float | None,
    sigma: float | None,
    log_a: float,
) -> np.ndarray:
    if kind == 'raw':
        return distances
    if kind == 'linear':
        return np.clip(distances / d_max, 0.0, 1.0)
    if kind == 'sqrt':
        return np.sqrt(np.clip(distances / d_max, 0.0, 1.0))
    if kind == 'log':
        # clipped = np.clip(distances, 0.0, d_max)
        # return np.log1p(log_a * clipped) / np.log1p(log_a * d_max)
        return np.log(distances + 1e-4)
    if kind == 'exp':
        return np.exp(-distances / sigma)
    if kind == 'inv':
        return 1.0 / (1.0 + distances / sigma)

    raise ValueError(f'Unsupported transform kind: {kind}')


def normalize_dump(dump):
    dump = pickle.loads(pickle.dumps(dump))
    dump['objects'] = [
        obj for obj in dump['objects']
        if obj['vertices'].size != 0 and obj['faces'].size != 0
    ]
    if len(dump['objects']) == 0:
        return dump

    vertices = torch.from_numpy(
        np.concatenate([obj['vertices'] for obj in dump['objects']], axis=0)
    ).float()
    vertices_min = vertices.min(dim=0)[0]
    vertices_max = vertices.max(dim=0)[0]
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()

    for obj in dump['objects']:
        obj['vertices'] = ((torch.from_numpy(obj['vertices']).float() - center) * scale).numpy()

    return dump


def build_global_mesh(dump) -> Tuple[torch.Tensor, torch.Tensor]:
    vertices = []
    edges = []
    offset = 0
    for obj in dump['objects']:
        v = torch.from_numpy(obj['vertices']).float()
        f = torch.from_numpy(obj['faces']).long()
        if v.numel() == 0 or f.numel() == 0:
            continue
        vertices.append(v)
        f = f + offset
        tri_edges = torch.stack([
            f[:, [0, 1]],
            f[:, [1, 2]],
            f[:, [2, 0]],
        ], dim=1).reshape(-1, 2)
        tri_edges = torch.sort(tri_edges, dim=1).values
        edges.append(tri_edges)
        offset += v.shape[0]

    if not vertices:
        return torch.empty((0, 3)), torch.empty((0, 2), dtype=torch.long)

    vertices = torch.cat(vertices, dim=0)
    edges = torch.unique(torch.cat(edges, dim=0), dim=0)
    return vertices, edges


def load_active_voxel_centers(dump, resolution: int) -> torch.Tensor:
    coords, _ = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        mip_level_offset=0,
        verbose=False,
        timing=False,
    )
    coords = coords.float()
    return coords / resolution - 0.5


def point_segment_dist2(points: torch.Tensor, seg_a: torch.Tensor, seg_b: torch.Tensor) -> torch.Tensor:
    ab = seg_b - seg_a
    denom = (ab * ab).sum(dim=-1).clamp_min(1e-12)
    ap = points[:, None, :] - seg_a[None, :, :]
    t = (ap * ab[None, :, :]).sum(dim=-1) / denom[None, :]
    t = t.clamp(0.0, 1.0)
    closest = seg_a[None, :, :] + t[..., None] * ab[None, :, :]
    diff = points[:, None, :] - closest
    return (diff * diff).sum(dim=-1)


def nearest_edge_distances(
    points: torch.Tensor,
    vertices: torch.Tensor,
    edges: torch.Tensor,
    device: torch.device,
    point_batch: int,
    edge_batch: int,
) -> torch.Tensor:
    if points.numel() == 0 or edges.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)

    vertices = vertices.to(device, non_blocking=True)
    edges = edges.to(device, non_blocking=True)
    seg_a_all = vertices[edges[:, 0]]
    seg_b_all = vertices[edges[:, 1]]

    out = []
    for p0 in range(0, points.shape[0], point_batch):
        p = points[p0:p0 + point_batch].to(device, non_blocking=True)
        best = torch.full((p.shape[0],), float('inf'), device=device)
        for e0 in range(0, edges.shape[0], edge_batch):
            seg_a = seg_a_all[e0:e0 + edge_batch]
            seg_b = seg_b_all[e0:e0 + edge_batch]
            dist2 = point_segment_dist2(p, seg_a, seg_b)
            best = torch.minimum(best, dist2.min(dim=1).values)
        out.append(best.sqrt().cpu())
    return torch.cat(out, dim=0)


def reservoir_update(reservoir, values: np.ndarray, seen: int, sample_size: int, rng: np.random.Generator):
    if sample_size <= 0 or values.size == 0:
        return seen
    if reservoir.size == 0:
        take = min(sample_size, values.size)
        reservoir.resize(take, refcheck=False)
        reservoir[:] = values[:take]
        seen += take
        values = values[take:]
    for value in values:
        seen += 1
        if reservoir.size < sample_size:
            reservoir.resize(reservoir.size + 1, refcheck=False)
            reservoir[-1] = value
            continue
        j = rng.integers(0, seen)
        if j < sample_size:
            reservoir[j] = value
    return seen


def pick_instances(metadata: pd.DataFrame, instances_arg: str | None):
    if instances_arg is None:
        return metadata
    if os.path.exists(instances_arg):
        with open(instances_arg, 'r') as f:
            instances = f.read().splitlines()
    else:
        instances = instances_arg.split(',')
    return metadata[metadata['sha256'].isin(instances)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subset', type=str)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--pbr_dump_root', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--kind', type=str, default='raw',
                        choices=['raw', 'linear', 'sqrt', 'log', 'exp', 'inv'])
    parser.add_argument('--bins', type=int, default=256)
    parser.add_argument('--sample_size', type=int, default=2000000)
    parser.add_argument('--point_batch', type=int, default=8192)
    parser.add_argument('--edge_batch', type=int, default=16384)
    parser.add_argument('--d_max', type=float, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--log_a', type=float, default=100.0)
    parser.add_argument('--instances', type=str, default=None)
    parser.add_argument('--max_meshes', type=int, default=None)
    parser.add_argument('--output', type=str, default='edge_distance_hist.png')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    opt = edict(vars(args))
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root

    dataset_utils = importlib.import_module(f'datasets.{opt.subset}')
    _ = dataset_utils  # keeps the interface parallel with other toolkit scripts

    metadata_path = os.path.join(opt.root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(metadata_path).set_index('sha256')
    pbr_meta_path = os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')
    if os.path.exists(pbr_meta_path):
        metadata = metadata.combine_first(pd.read_csv(pbr_meta_path).set_index('sha256'))
    metadata = metadata.reset_index()
    metadata = metadata[metadata['pbr_dumped'] == True]
    metadata = pick_instances(metadata, opt.instances)
    if opt.max_meshes is not None:
        metadata = metadata.head(opt.max_meshes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(opt.seed)
    reservoir = np.empty((0,), dtype=np.float32)

    global_min = float('inf')
    global_max = 0.0
    total_count = 0
    total_sum = 0.0

    pbar = tqdm(metadata.itertuples(index=False), total=len(metadata), desc='Edge-distance histogram')
    for row in pbar:
        sha256 = row.sha256
        dump_path = os.path.join(opt.pbr_dump_root, 'pbr_dumps', f'{sha256}.pickle')
        if not os.path.exists(dump_path):
            continue

        with open(dump_path, 'rb') as f:
            dump = pickle.load(f)
        dump = normalize_dump(dump)
        vertices, edges = build_global_mesh(dump)
        if vertices.numel() == 0 or edges.numel() == 0:
            continue

        points = load_active_voxel_centers(dump, opt.resolution)
        if points.numel() == 0:
            continue

        distances = nearest_edge_distances(
            points=points,
            vertices=vertices,
            edges=edges,
            device=device,
            point_batch=opt.point_batch,
            edge_batch=opt.edge_batch,
        ).numpy()
        if distances.size == 0:
            continue

        global_min = min(global_min, float(distances.min()))
        global_max = max(global_max, float(distances.max()))
        total_sum += float(distances.sum())
        total_count += int(distances.size)
        seen_before = max(total_count - distances.size, 0)
        reservoir_update(reservoir, distances, seen_before, opt.sample_size, rng)

        pbar.set_postfix({
            'device': device.type,
            'voxels': total_count,
            'sampled': reservoir.size,
            'max': f'{global_max:.5f}',
        })

    if total_count == 0 or reservoir.size == 0:
        raise RuntimeError('No valid edge distances were computed.')

    mean = total_sum / total_count
    q = np.quantile(reservoir, [0.5, 0.9, 0.95, 0.99, 0.999])
    transformed = apply_transform(
        reservoir,
        kind=opt.kind,
        d_max=opt.d_max,
        sigma=opt.sigma,
        log_a=opt.log_a,
    )

    plt.figure(figsize=(9, 5))
    plt.hist(transformed, bins=opt.bins, color='black')
    plt.xlabel('Edge distance' if opt.kind == 'raw' else f'{opt.kind} transform(edge distance)')
    plt.ylabel('Count (reservoir sample)')
    plt.title(f'Edge distance distribution @ resolution {opt.resolution} ({opt.kind})')
    plt.tight_layout()
    plt.savefig(opt.output, dpi=180)

    print(f'Wrote histogram to {opt.output}')
    print(f'device={device.type}')
    print(f'kind={opt.kind}')
    print(f'total_count={total_count}')
    print(f'sample_size={reservoir.size}')
    print(f'min={global_min:.8f}')
    print(f'mean={mean:.8f}')
    print(f'max={global_max:.8f}')
    print(f'p50={q[0]:.8f}')
    print(f'p90={q[1]:.8f}')
    print(f'p95={q[2]:.8f}')
    print(f'p99={q[3]:.8f}')
    print(f'p99.9={q[4]:.8f}')
    if opt.kind != 'raw':
        tq = np.quantile(transformed, [0.5, 0.9, 0.95, 0.99, 0.999])
        print(f'transformed_min={float(transformed.min()):.8f}')
        print(f'transformed_max={float(transformed.max()):.8f}')
        print(f'transformed_p50={tq[0]:.8f}')
        print(f'transformed_p90={tq[1]:.8f}')
        print(f'transformed_p95={tq[2]:.8f}')
        print(f'transformed_p99={tq[3]:.8f}')
        print(f'transformed_p99.9={tq[4]:.8f}')


if __name__ == '__main__':
    main()
