import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import o_voxel
import torch
from PIL import Image, ImageDraw

from data_toolkit.voxelize_triangle_field import (
    build_global_triangles,
    normalize_dump,
    triangle_features_from_native_candidates,
    voxel_coords_to_centers,
)
from visualize_triangle_field_downsampled_fields import render_channel


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize vertex-averaged triangle geometry fields.')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--instances', type=Path, required=True)
    parser.add_argument('--duplicate_filter_csv', type=Path, default=None)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--point_batch', type=int, default=8192)
    parser.add_argument('--render_resolution', type=int, default=320)
    parser.add_argument('--ssaa', type=int, default=4)
    parser.add_argument('--field', choices=['elongation', 'edge_density'], default='elongation')
    return parser.parse_args()


def filtered_instances(args):
    instances = [line.strip() for line in args.instances.read_text().splitlines() if line.strip()]
    if args.duplicate_filter_csv is not None:
        with args.duplicate_filter_csv.open() as handle:
            keep = {row['sha256'] for row in csv.DictReader(handle)}
        instances = [instance for instance in instances if instance in keep]
    return list(np.random.default_rng(args.seed).permutation(instances))


def triangle_elongation(vertices, faces):
    tri = vertices[faces]
    edge_sq = torch.stack([
        (tri[:, 1] - tri[:, 0]).square().sum(dim=1),
        (tri[:, 2] - tri[:, 1]).square().sum(dim=1),
        (tri[:, 0] - tri[:, 2]).square().sum(dim=1),
    ], dim=1)
    area = 0.5 * torch.linalg.norm(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1)
    quality = (4.0 * np.sqrt(3.0) * area / edge_sq.sum(dim=1).clamp_min(1e-20)).clamp(1e-8, 1.0)
    return -torch.log(quality)


def triangle_edge_density(vertices, faces, resolution):
    tri = vertices[faces]
    edge_lengths = torch.stack([
        torch.linalg.norm(tri[:, 1] - tri[:, 0], dim=1),
        torch.linalg.norm(tri[:, 2] - tri[:, 1], dim=1),
        torch.linalg.norm(tri[:, 0] - tri[:, 2], dim=1),
    ], dim=1)
    area = 0.5 * torch.linalg.norm(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1),
        dim=1,
    )
    edge_length_density = edge_lengths.sum(dim=1) / (2.0 * area.clamp_min(1e-20))
    return torch.log((edge_length_density / float(resolution)).clamp_min(1e-8))


def vertex_mean(values, faces, num_vertices):
    sums = torch.zeros(num_vertices, dtype=values.dtype, device=values.device)
    counts = torch.zeros_like(sums)
    ones = torch.ones_like(values)
    for corner in range(3):
        sums.scatter_add_(0, faces[:, corner], values)
        counts.scatter_add_(0, faces[:, corner], ones)
    return sums / counts.clamp_min(1.0)


def compute_sample(args, instance):
    with (args.root / 'pbr_dumps' / f'{instance}.pickle').open('rb') as handle:
        dump = normalize_dump(pickle.load(handle))
    vertices, faces = build_global_triangles(dump, filter_degenerate=False)
    coords, offsets, triangle_ids, barycentric = o_voxel.convert.blender_dump_to_voxel_triangle_candidates(
        dump,
        grid_size=args.resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        verbose=False,
        timing=False,
    )
    points = voxel_coords_to_centers(coords, args.resolution)
    features, _, selected_triangles, selected_barycentric = triangle_features_from_native_candidates(
        coords=coords,
        points=points,
        vertices=vertices,
        faces=faces,
        candidate_offsets=offsets,
        candidate_triangle_ids=triangle_ids,
        candidate_barycentric=barycentric,
        device=torch.device('cuda'),
        point_batch=args.point_batch,
        projection_mode='inside_barycentric',
        resolution=args.resolution,
        return_projection=True,
    )

    vertices = vertices.cuda()
    faces = faces.cuda()
    if args.field == 'elongation':
        per_triangle = triangle_elongation(vertices, faces)
    else:
        per_triangle = triangle_edge_density(vertices, faces, args.resolution)
    per_vertex = vertex_mean(per_triangle, faces, vertices.shape[0])
    selected_triangles = torch.from_numpy(selected_triangles).long().cuda()
    bary = torch.from_numpy(selected_barycentric).float().cuda().clamp(0.0, 1.0)
    bary = bary / bary.sum(dim=1, keepdim=True).clamp_min(1e-8)
    field = (per_vertex[faces[selected_triangles]] * bary).sum(dim=1).cpu()
    return coords.cpu(), torch.from_numpy(features[:, 0]), field, per_triangle.cpu()


def make_grid(items, out_path, field_name, lower, upper):
    pad, label_h, title_h, header_h = 10, 32, 38, 28
    tile_w = max(image.width for _, edge_image, elongation_image, _ in items for image in (edge_image, elongation_image))
    tile_h = max(image.height for _, edge_image, elongation_image, _ in items for image in (edge_image, elongation_image))
    canvas = Image.new(
        'RGB',
        (2 * tile_w + 3 * pad, title_h + header_h + len(items) * (tile_h + label_h + pad)),
        'white',
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (pad, 10),
        f'Edge distance vs {field_name}; shared p01={lower:.3f}, p99={upper:.3f}',
        fill='black',
    )
    draw.text((pad, title_h + 4), 'Edge distance (d_tri, native [0,1])', fill='black')
    draw.text((2 * pad + tile_w, title_h + 4), f'Vertex-averaged {field_name}', fill='black')
    for row, (instance, edge_image, elongation_image, stats) in enumerate(items):
        y = title_h + header_h + row * (tile_h + label_h + pad)
        draw.text((pad, y), f'{instance[:12]}  median={stats[0]:.3f}  p99={stats[1]:.3f}', fill='black')
        canvas.paste(edge_image, (pad, y + label_h))
        canvas.paste(elongation_image, (2 * pad + tile_w, y + label_h))
    canvas.save(out_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for instance in filtered_instances(args):
        try:
            coords, edge_distance, field, per_triangle = compute_sample(args, instance)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f'Skipping {instance}: {exc}', flush=True)
            continue
        samples.append((instance, coords, edge_distance, field, per_triangle))
        print(f'Processed {instance}: {coords.shape[0]:,} voxels', flush=True)
        if len(samples) == args.num_samples:
            break
    if len(samples) < args.num_samples:
        raise RuntimeError(f'Only processed {len(samples)} of {args.num_samples} requested samples')

    all_field_values = torch.cat([sample[3] for sample in samples])
    lower = float(torch.quantile(all_field_values, 0.01))
    upper = float(torch.quantile(all_field_values, 0.99))
    scale = max(upper - lower, 1e-6)
    rendered = []
    formulas = {
        'elongation': '-log(4*sqrt(3)*area/sum(edge_length^2) + eps)',
        'edge_density': 'log((perimeter/(2*area))*voxel_size + eps)',
    }
    summary = {
        'field': args.field,
        'formula': formulas[args.field],
        'shared_p01': lower,
        'shared_p99': upper,
        'samples': [],
    }
    for instance, coords, edge_distance, field, per_triangle in samples:
        normalized = ((field - lower) / scale).clamp(0.0, 1.0).reshape(-1, 1)
        edge_image = render_channel(
            coords,
            edge_distance.reshape(-1, 1),
            args.resolution,
            0,
            'magma',
            args.render_resolution,
            args.ssaa,
        )
        field_image = render_channel(
            coords,
            normalized,
            args.resolution,
            0,
            'magma',
            args.render_resolution,
            args.ssaa,
        )
        edge_image.save(args.output_dir / f'{instance}_edge_distance.png')
        field_image.save(args.output_dir / f'{instance}_{args.field}.png')
        stats = (float(torch.median(field)), float(torch.quantile(field, 0.99)))
        rendered.append((instance, edge_image, field_image, stats))
        summary['samples'].append({
            'instance': instance,
            'num_voxels': int(coords.shape[0]),
            'voxel_median': stats[0],
            'voxel_p99': stats[1],
            'triangle_median': float(torch.median(per_triangle)),
            'triangle_p99': float(torch.quantile(per_triangle, 0.99)),
        })
    make_grid(
        rendered,
        args.output_dir / f'edge_distance_vs_{args.field}_grid.png',
        args.field.replace('_', ' '),
        lower,
        upper,
    )
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Wrote visualizations to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
