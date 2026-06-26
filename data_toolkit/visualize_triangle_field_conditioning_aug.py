import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trellis2.datasets.sparse_voxel_triangle_field import (
    SparseVoxelTriangleFieldVisMixin,
    find_triangle_field_path,
    load_triangle_field_npz,
)
from trellis2.modules import sparse as sp


class TriangleFieldPanelVisualizer(SparseVoxelTriangleFieldVisMixin):
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.distance_transform = 'minus_one_one'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize sparse blur/noise augmentation for duplicated triangle-field conditioning.'
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--instances', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--resolutions', nargs='+', type=int, default=[32, 64, 128, 256])
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.15, 0.25])
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--sigma', type=float, default=1.0)
    return parser.parse_args()


def load_instances(path: str) -> List[str]:
    with open(path, 'r') as fp:
        return [line.strip() for line in fp if line.strip()]


def read_target(root: str, instance: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = find_triangle_field_path(root, instance)
    with load_triangle_field_npz(path) as data:
        coords = torch.from_numpy(data['coords'].astype(np.int32, copy=False))
        features = torch.from_numpy(data['features'].astype(np.float32, copy=False))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'{path} has invalid coords shape {tuple(coords.shape)}')
    if features.ndim != 2 or features.shape[1] < 2:
        raise ValueError(f'{path} has invalid features shape {tuple(features.shape)}')
    target = features[:, :2].float() * 2.0 - 1.0
    return coords.int(), target


def coord_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.long()
    return coords[:, 0] * (resolution * resolution) + coords[:, 1] * resolution + coords[:, 2]


def low_to_high_support(
    low_coords: torch.Tensor,
    low_target: torch.Tensor,
    high_coords: torch.Tensor,
    low_resolution: int,
    high_resolution: int,
) -> torch.Tensor:
    factor = high_resolution // low_resolution
    parent_coords = torch.div(high_coords, factor, rounding_mode='floor')
    low_keys = coord_keys(low_coords, low_resolution)
    parent_keys = coord_keys(parent_coords, low_resolution)
    order = torch.argsort(low_keys)
    sorted_keys = low_keys[order]
    sorted_feats = low_target[order]
    idx = torch.searchsorted(sorted_keys, parent_keys)
    valid = (idx < sorted_keys.numel()) & (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == parent_keys)
    cond = torch.zeros((high_coords.shape[0], low_target.shape[1]), dtype=low_target.dtype)
    if valid.any():
        cond[valid] = sorted_feats[idx[valid]]
    return cond


def gaussian_sparse_blur(
    coords: torch.Tensor,
    feats: torch.Tensor,
    resolution: int,
    sigma: float,
) -> torch.Tensor:
    keys = coord_keys(coords, resolution)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    sorted_feats = feats[order]

    accum = torch.zeros_like(feats)
    denom = torch.zeros((feats.shape[0], 1), dtype=feats.dtype)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for dx, dy, dz in offsets:
        offset = torch.tensor([dx, dy, dz], dtype=coords.dtype)
        neighbor = coords + offset
        valid = (
            (neighbor[:, 0] >= 0) & (neighbor[:, 0] < resolution) &
            (neighbor[:, 1] >= 0) & (neighbor[:, 1] < resolution) &
            (neighbor[:, 2] >= 0) & (neighbor[:, 2] < resolution)
        )
        if not valid.any():
            continue
        q = coord_keys(neighbor[valid], resolution)
        idx = torch.searchsorted(sorted_keys, q)
        hit = (idx < sorted_keys.numel()) & (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == q)
        if not hit.any():
            continue
        rows = valid.nonzero(as_tuple=False).reshape(-1)[hit]
        src = sorted_feats[idx[hit]]
        weight = float(np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)))
        accum[rows] += src * weight
        denom[rows] += weight
    return accum / denom.clamp_min(1e-6)


def add_noise_interp(feats: torch.Tensor, level: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(feats.shape, generator=generator, dtype=feats.dtype)
    return ((1.0 - level) * feats + level * noise).clamp(-1.0, 1.0)


@torch.no_grad()
def render_panel(
    visualizer: TriangleFieldPanelVisualizer,
    coords: torch.Tensor,
    feats: torch.Tensor,
    resolution: int,
) -> Dict[str, Image.Image]:
    visualizer.resolution = resolution
    sparse_coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1).int()
    x = sp.SparseTensor(feats.float(), sparse_coords)
    rendered = visualizer.visualize_sample({'target': x})
    images = {}
    for key, value in rendered.items():
        img = value[0].detach().float().clamp(0, 1).cpu()
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        images[key] = Image.fromarray(img)
    return images


def add_header(img: Image.Image, text: str, height: int = 34) -> Image.Image:
    out = Image.new('RGB', (img.width, img.height + height), (245, 242, 235))
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((8, 10), text, fill=(25, 31, 38), font=font)
    return out


def make_grid(rows: List[List[Image.Image]]) -> Image.Image:
    gap = 10
    widths = [max(row[col].width for row in rows) for col in range(len(rows[0]))]
    heights = [max(img.height for img in row) for row in rows]
    total_w = sum(widths) + gap * (len(widths) - 1)
    total_h = sum(heights) + gap * (len(heights) - 1)
    canvas = Image.new('RGB', (total_w, total_h), (235, 232, 224))
    y = 0
    for r, row in enumerate(rows):
        x = 0
        for c, img in enumerate(row):
            canvas.paste(img, (x, y))
            x += widths[c] + gap
        y += heights[r] + gap
    return canvas


def find_complete_instances(root: str, instances: Iterable[str], resolutions: List[int]) -> List[str]:
    complete = []
    needed = sorted(set(resolutions + [r * 2 for r in resolutions]))
    for instance in instances:
        ok = True
        for resolution in needed:
            voxel_root = os.path.join(root, f'triangle_field_voxels_{resolution}')
            try:
                find_triangle_field_path(voxel_root, instance)
            except FileNotFoundError:
                ok = False
                break
        if ok:
            complete.append(instance)
    return complete


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    instances_path = args.instances or os.path.join(args.root, 'instances_no_triangle_dense.txt')
    output_dir = Path(args.output_dir or os.path.join(args.root, 'outputs', 'conditioning_aug_vis'))
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(instances_path)
    complete = find_complete_instances(args.root, instances, args.resolutions)
    if len(complete) < args.num_samples:
        raise ValueError(f'Only found {len(complete)} complete instances for resolutions {args.resolutions}')
    selected = random.sample(complete, args.num_samples)

    visualizer = TriangleFieldPanelVisualizer(resolution=max(args.resolutions) * 2)
    generator = torch.Generator().manual_seed(args.seed + 1000)

    summary = []
    for instance in selected:
        per_level: Dict[int, Dict[str, torch.Tensor]] = {}
        for low_resolution in args.resolutions:
            high_resolution = low_resolution * 2
            low_root = os.path.join(args.root, f'triangle_field_voxels_{low_resolution}')
            high_root = os.path.join(args.root, f'triangle_field_voxels_{high_resolution}')
            low_coords, low_target = read_target(low_root, instance)
            high_coords, high_target = read_target(high_root, instance)
            dup = low_to_high_support(low_coords, low_target, high_coords, low_resolution, high_resolution)
            blur = gaussian_sparse_blur(high_coords, dup, high_resolution, args.sigma).clamp(-1.0, 1.0)
            per_level[low_resolution] = {
                'high_coords': high_coords,
                'high_target': high_target,
                'dup': dup,
                'blur': blur,
            }

        for noise_level in args.noise_levels:
            rows_by_field = {'d_tri': [], 'd_vert': []}
            for low_resolution in args.resolutions:
                high_resolution = low_resolution * 2
                item = per_level[low_resolution]
                aug = add_noise_interp(item['blur'], noise_level, generator)
                panels = [
                    (item['high_target'], f'{low_resolution}->{high_resolution} GT'),
                    (item['dup'], 'duplicated cond'),
                    (item['blur'], f'blur sigma={args.sigma:g}'),
                    (aug, f'blur + noise {noise_level:g}'),
                ]
                rendered_panels = [
                    (render_panel(visualizer, item['high_coords'], feats, high_resolution), title)
                    for feats, title in panels
                ]
                for field in ('d_tri', 'd_vert'):
                    rows_by_field[field].append([
                        add_header(images[field], f'{title} {field}')
                        for images, title in rendered_panels
                    ])

            for field, rows in rows_by_field.items():
                grid = make_grid(rows)
                out_path = output_dir / f'{instance}_{field}_noise{noise_level:g}.png'
                grid.save(out_path)
                summary.append(str(out_path))

    summary_path = output_dir / 'summary.txt'
    summary_path.write_text('\n'.join(summary) + '\n')
    print(f'Selected {len(selected)} samples:')
    for instance in selected:
        print(f'  {instance}')
    print(f'Wrote {len(summary)} images to {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
