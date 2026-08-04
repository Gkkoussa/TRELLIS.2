"""Visualize point-flow density predictions on projected 512 voxel centers."""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from eval_triangle_field_latent_sr_cascade import save_image_grid
from eval_obj_folder_latent_sr_stage_repeat_cascade import (
    collect_meshes,
    mesh_sha1,
    support_coords_from_mesh,
)
from trellis2 import models
from trellis2.datasets.point_density_mesh import (
    PointDensityMeshDataset,
    load_normalized_mesh,
    sample_surface_field,
)
from trellis2.datasets.sparse_voxel_triangle_field import (
    INPUT_LAYOUT,
    SparseVoxelTriangleFieldVisMixin,
    find_triangle_field_path,
    load_triangle_field_npz,
)
from trellis2.modules import sparse as sp
from trellis2.trainers.flow_matching.point_density_flow import PointDensityFlowEulerSampler


class DensityVoxelVisualizer(SparseVoxelTriangleFieldVisMixin):
    resolution = 512
    target_layout = {'density_field': slice(0, 1)}
    input_layout = target_layout


class FixedScaleDensityVoxelVisualizer(SparseVoxelTriangleFieldVisMixin):
    """Render pre-mapped [0, 1] density without per-mesh percentile stretching."""

    resolution = 512
    target_layout = {'d_tri': slice(0, 1)}
    input_layout = target_layout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--stats_path', required=True)
    parser.add_argument('--mesh_dir', default=None)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--support_cache_dir', default=None)
    parser.add_argument('--projection_chunk_size', type=int, default=65536)
    parser.add_argument('--voxel_root', default=None)
    parser.add_argument('--gt_voxel_root', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument(
        '--density_payload_dir',
        default=None,
        help='Optionally save raw 512-resolution coords and predicted density per mesh.',
    )
    parser.add_argument('--ckpt', default='latest')
    parser.add_argument('--ema_rate', default='0.9999')
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_latents', type=int, default=None)
    parser.add_argument(
        '--randomize_latent_points_each_step',
        action='store_true',
        help='Rerun random-start FPS for the latent anchor set on every Euler step.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_active_voxels', type=int, default=1_000_000)
    parser.add_argument(
        '--sampling_density_target_mean',
        type=float,
        default=None,
        help='Shift predicted x0 to this density mean at every Euler step.',
    )
    parser.add_argument(
        '--sampling_density_clamp_max',
        type=float,
        default=None,
        help='Upper-clamp predicted x0 after mean shifting at every Euler step.',
    )
    parser.add_argument(
        '--sampling_density_target_std',
        type=float,
        default=None,
        help='Rescale predicted x0 to this standard deviation after mean shifting.',
    )
    parser.add_argument(
        '--sampling_density_base_resolution',
        type=int,
        default=128,
        help='Resolution scale at which the sampling density constraints are specified.',
    )
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, ckpt: str, weights: str, ema_rate: str):
    prefix = f'denoiser_ema{ema_rate}' if weights == 'ema' else 'denoiser'
    if ckpt == 'latest':
        paths = list((run_dir / 'ckpts').glob(f'{prefix}_step*.pt'))
        if not paths:
            raise FileNotFoundError(f'No {weights} checkpoints found in {run_dir / "ckpts"}')
        path = max(paths, key=lambda item: int(item.stem.split('step')[-1]))
    else:
        path = run_dir / 'ckpts' / f'{prefix}_step{int(ckpt):07d}.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    return int(path.stem.split('step')[-1]), path


def load_projected_voxel_queries(voxel_root: Path, sha256: str):
    path = find_triangle_field_path(str(voxel_root), sha256)
    with load_triangle_field_npz(path) as payload:
        coords = payload['coords'].astype(np.int32, copy=False)
        features = payload['features']
        offsets = features[:, INPUT_LAYOUT['offset_to_projection']].astype(
            np.float32, copy=False
        )
        gt_density = (
            features[:, 20:21].astype(np.float32, copy=False)
            if features.shape[1] > 20
            else None
        )
    centers = (coords.astype(np.float32) + 0.5) / 512.0 - 0.5
    projected_z_up = centers + offsets
    # Point-flow meshes use the source OBJ's Y-up frame; triangle-field voxels
    # use TRELLIS's Z-up frame. This is the inverse of preprocess_mesh().
    projected_y_up = projected_z_up[:, [0, 2, 1]].copy()
    projected_y_up[:, 2] *= -1.0
    return coords, projected_y_up, gt_density


def sample_mesh_folder(mesh, dataset_args, stats, seed):
    context_count = int(dataset_args['context_points'])
    query_count = int(dataset_args['query_points'])
    points, normals, density = sample_surface_field(
        mesh,
        context_count + query_count,
        field_name='density',
        resolution=512,
        seed=seed,
    )
    density = (density - float(stats['mean'])) / float(stats['std'])
    return {
        'context_points': torch.from_numpy(points[:context_count]),
        'context_normals': torch.from_numpy(normals[:context_count]),
        'context_density': torch.from_numpy(density[:context_count]),
        'query_points': torch.from_numpy(points[context_count:]),
        'query_density': torch.from_numpy(density[context_count:]),
    }


def project_support_centers(mesh, coords, chunk_size):
    centers = (coords.astype(np.float32) + 0.5) / 512.0 - 0.5
    projected = []
    for start in range(0, len(centers), chunk_size):
        closest, _, _ = mesh.nearest.on_surface(centers[start:start + chunk_size])
        projected.append(closest.astype(np.float32, copy=False))
    return np.concatenate(projected, axis=0)


def make_sparse(values: torch.Tensor, coords: np.ndarray, device):
    sparse_coords = torch.cat([
        torch.zeros((len(coords), 1), dtype=torch.int32, device=device),
        torch.from_numpy(coords).to(device=device, dtype=torch.int32),
    ], dim=1)
    return sp.SparseTensor(values, sparse_coords)


def normalized_metrics(prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    correlation = (
        (centered_prediction * centered_target).mean()
        / (
            centered_prediction.square().mean().sqrt()
            * centered_target.square().mean().sqrt()
        ).clamp_min(1e-8)
    )
    return {
        'normalized_mse': float((prediction - target).square().mean()),
        'normalized_mae': float((prediction - target).abs().mean()),
        'correlation': float(correlation),
    }


def scale_density(value: float, source_resolution: int, target_resolution: int):
    return float(value) - 2.0 * math.log(target_resolution / source_resolution)


def serialize_constraint_stats(step_stats, stats, base_resolution):
    mean = float(stats['mean'])
    std = float(stats['std'])
    resolution_offset = 2.0 * math.log(512 / base_resolution)
    serialized = []
    for index, step in enumerate(step_stats):
        values = {'step': index, 'timestep': step['timestep']}
        for name in (
            'pre_shift_mean',
            'shifted_mean',
            'scaled_mean',
            'post_clamp_mean',
        ):
            normalized = float(step[name].float().mean())
            physical_512 = normalized * std + mean
            values[f'{name}_normalized'] = normalized
            values[f'{name}_512'] = physical_512
            values[f'{name}_base_resolution'] = physical_512 + resolution_offset
        for name in ('pre_scale_std', 'scaled_std', 'post_clamp_std'):
            normalized = float(step[name].float().mean())
            values[f'{name}_normalized'] = normalized
            values[f'{name}_physical'] = normalized * std
        values['clipped_fraction'] = float(
            step['clipped_fraction'].float().mean()
        )
        serialized.append(values)
    return serialized


@torch.no_grad()
def main():
    total_start = time.perf_counter()
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    root = Path(args.root).resolve()
    stats_path = Path(args.stats_path).resolve()
    voxel_root = (
        Path(args.voxel_root).resolve()
        if args.voxel_root
        else root / 'triangle_field_voxels_512'
    )
    gt_voxel_root = (
        Path(args.gt_voxel_root).resolve()
        if args.gt_voxel_root
        else root
        / 'triangle_field_voxels_density_elongation_field'
        / 'triangle_field_voxels_512'
    )
    step, ckpt_path = resolve_checkpoint(
        run_dir, args.ckpt, args.weights, args.ema_rate
    )
    sample_label = args.num_samples if args.num_samples is not None else 'all'
    source_label = f'_meshfolder_{Path(args.mesh_dir).name}' if args.mesh_dir else ''
    anchor_label = '_randomanchors' if args.randomize_latent_points_each_step else ''
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / (
        f'eval_voxel_density_r512_{args.weights}'
        f'{args.ema_rate if args.weights == "ema" else ""}_step{step:07d}'
        f'_latents{args.num_latents or "config"}'
        f'_steps{args.steps}_n{sample_label}_seed{args.seed}{source_label}{anchor_label}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    density_payload_dir = (
        Path(args.density_payload_dir).resolve()
        if args.density_payload_dir
        else None
    )
    if density_payload_dir is not None:
        density_payload_dir.mkdir(parents=True, exist_ok=True)

    setup_start = time.perf_counter()
    start = time.perf_counter()
    with open(run_dir / 'config.json') as file:
        config = json.load(file)
    with open(stats_path) as file:
        stats = json.load(file)
    if args.sampling_density_base_resolution <= 0:
        raise ValueError('--sampling_density_base_resolution must be positive')
    constraint_values = {}
    for name, value in (
        ('target_mean', args.sampling_density_target_mean),
        ('clamp_max', args.sampling_density_clamp_max),
    ):
        if value is None:
            constraint_values[name] = None
            constraint_values[f'{name}_512'] = None
            constraint_values[f'{name}_normalized'] = None
            continue
        value_512 = scale_density(
            value, args.sampling_density_base_resolution, 512
        )
        constraint_values[name] = value
        constraint_values[f'{name}_512'] = value_512
        constraint_values[f'{name}_normalized'] = (
            value_512 - float(stats['mean'])
        ) / float(stats['std'])
    constraint_values['target_std'] = args.sampling_density_target_std
    constraint_values['target_std_normalized'] = (
        args.sampling_density_target_std / float(stats['std'])
        if args.sampling_density_target_std is not None
        else None
    )
    if (
        constraint_values['target_std_normalized'] is not None
        and constraint_values['target_std_normalized'] <= 0
    ):
        raise ValueError('Density target standard deviation must be positive')
    if (
        constraint_values['target_mean_normalized'] is not None
        and constraint_values['clamp_max_normalized'] is not None
        and constraint_values['target_mean_normalized']
        > constraint_values['clamp_max_normalized']
    ):
        raise ValueError('Density target mean must not exceed the upper clamp')
    setup_timings = {'config_and_stats_load_s': time.perf_counter() - start}

    start = time.perf_counter()
    dataset_args = dict(config['validation_dataset']['args'])
    dataset_args['density_stats_path'] = str(stats_path)
    dataset_args['deterministic_sampling'] = True
    mesh_paths = None
    if args.mesh_dir:
        mesh_paths = collect_meshes(Path(args.mesh_dir).resolve(), args.recursive)
        if args.num_samples is not None:
            mesh_paths = mesh_paths[:args.num_samples]
        dataset = None
    else:
        filters = ','.join([
            str(root / 'splits/test_triangle_field_512/metadata.csv'),
            str(root / 'metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv'),
        ])
        roots = {'filtered_test': {'mesh': str(root), '_metadata_filter_csv': filters}}
        dataset = PointDensityMeshDataset(json.dumps(roots), **dataset_args)
    setup_timings['input_source_setup_s'] = time.perf_counter() - start

    device = torch.device('cuda')
    model_config = config['models']['denoiser']
    model_args = dict(model_config['args'])
    if args.num_latents is not None:
        model_args['num_latents'] = args.num_latents
    start = time.perf_counter()
    model = getattr(models, model_config['name'])(**model_args)
    setup_timings['model_construction_s'] = time.perf_counter() - start
    start = time.perf_counter()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    setup_timings['checkpoint_load_s'] = time.perf_counter() - start
    start = time.perf_counter()
    model.to(device).eval()
    torch.cuda.synchronize(device)
    setup_timings['model_to_gpu_s'] = time.perf_counter() - start
    setup_s = time.perf_counter() - setup_start
    sigma_min = float(config['trainer']['args']['sigma_min'])
    percentile_visualizer = DensityVoxelVisualizer()
    fixed_visualizer = FixedScaleDensityVoxelVisualizer()
    rendered_fixed = []
    rendered_percentile = []
    records = []
    autocast = (
        lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device.type == 'cuda'
        else nullcontext()
    )

    num_items = len(mesh_paths) if mesh_paths is not None else min(
        args.num_samples if args.num_samples is not None else 4,
        len(dataset),
    )
    for index in range(num_items):
        sample_start = time.perf_counter()
        timings = {}
        if mesh_paths is not None:
            mesh_path = mesh_paths[index]
            sha256 = mesh_sha1(mesh_path)
            sample_seed = (args.seed + int(sha256[:8], 16)) % (2**31)
            start = time.perf_counter()
            mesh = load_normalized_mesh(str(mesh_path))
            timings['mesh_load_s'] = time.perf_counter() - start
            start = time.perf_counter()
            sample = sample_mesh_folder(mesh, dataset_args, stats, sample_seed)
            timings['surface_sampling_s'] = time.perf_counter() - start
        else:
            root_entry, sha256 = dataset.instances[index]
            mesh_path = Path(dataset._mesh_path(root_entry, sha256))
            start = time.perf_counter()
            sample = dataset[index]
            timings['dataset_load_and_surface_sampling_s'] = time.perf_counter() - start

        if mesh_paths is not None:
            cache_dir = (
                Path(args.support_cache_dir).resolve()
                if args.support_cache_dir
                else output_dir / 'support_cache'
            )
            start = time.perf_counter()
            coords = support_coords_from_mesh(mesh_path, 512, cache_dir)
            timings['support_extraction_or_cache_load_s'] = time.perf_counter() - start
            start = time.perf_counter()
            query_points = project_support_centers(
                mesh, coords, args.projection_chunk_size
            )
            timings['voxel_center_surface_projection_s'] = time.perf_counter() - start
            gt_density = None
        else:
            start = time.perf_counter()
            query_root = gt_voxel_root if gt_voxel_root.exists() else voxel_root
            coords, query_points, gt_density = load_projected_voxel_queries(
                query_root, str(sha256)
            )
            timings['voxel_query_payload_load_s'] = time.perf_counter() - start
        if len(coords) > args.max_active_voxels:
            raise ValueError(
                f'{mesh_path} has {len(coords)} support voxels, above '
                f'--max_active_voxels {args.max_active_voxels}'
            )
        with autocast():
            start = time.perf_counter()
            points = sample['context_points'][None].to(device)
            normals = sample['context_normals'][None].to(device)
            query_points_tensor = torch.from_numpy(query_points)[None].to(device)
            surface_query_points = sample['query_points'][None].to(device)
            context_density_target = sample['context_density'].to(device)
            surface_density_target = sample['query_density'].to(device)
            torch.cuda.synchronize(device)
            timings['host_to_gpu_transfer_s'] = time.perf_counter() - start

            start = time.perf_counter()
            alignment_queries = query_points_tensor[:, :1024].float()
            nearest_context_distance = torch.cdist(
                alignment_queries, points.float()
            ).amin(dim=-1)
            torch.cuda.synchronize(device)
            timings['alignment_diagnostic_s'] = time.perf_counter() - start

            start = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            noise = torch.randn(
                (1, model.context_points, 1), device=device, generator=generator
            )
            if args.randomize_latent_points_each_step:
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                anchor_indices = None
            else:
                anchor_indices = model.select_anchor_indices(points, random_start=False)
            torch.cuda.synchronize(device)
            timings['noise_and_fps_anchor_setup_s'] = time.perf_counter() - start
            sampler = PointDensityFlowEulerSampler(
                sigma_min,
                x0_target_mean=constraint_values['target_mean_normalized'],
                x0_target_std=constraint_values['target_std_normalized'],
                x0_clamp_max=constraint_values['clamp_max_normalized'],
            )
            start = time.perf_counter()
            condition = {
                'context_points': points,
                'context_normals': normals,
            }
            if args.randomize_latent_points_each_step:
                condition['anchor_random_start'] = True
            else:
                condition['anchor_indices'] = anchor_indices
            result = sampler.sample(
                model,
                noise=noise,
                cond=condition,
                steps=args.steps,
                verbose=True,
                tqdm_desc='Sampling surface density',
            )
            torch.cuda.synchronize(device)
            timings['flow_sampling_s'] = time.perf_counter() - start
            constraint_step_stats = serialize_constraint_stats(
                sampler.constraint_stats,
                stats,
                args.sampling_density_base_resolution,
            )
            start = time.perf_counter()
            latents, _ = model.encode(
                points,
                normals,
                result.samples,
                torch.zeros(1, device=device),
                anchor_indices=anchor_indices,
                anchor_random_start=(
                    True if args.randomize_latent_points_each_step else None
                ),
            )
            torch.cuda.synchronize(device)
            timings['final_reencode_s'] = time.perf_counter() - start
            start = time.perf_counter()
            density_normalized = model.decode(
                latents,
                query_points_tensor,
            )[0].float()
            surface_query_normalized = model.decode(
                latents,
                surface_query_points,
            )[0].float()
            torch.cuda.synchronize(device)
            timings['voxel_query_decode_s'] = time.perf_counter() - start
        start = time.perf_counter()
        density = density_normalized * float(stats['std']) + float(stats['mean'])
        if density_payload_dir is not None:
            start = time.perf_counter()
            np.savez_compressed(
                density_payload_dir / f'{sha256}.npz',
                coords=coords.astype(np.int32, copy=False),
                density=density.cpu().numpy().astype(np.float32, copy=False),
                resolution=np.int32(512),
            )
            timings['density_payload_save_s'] = time.perf_counter() - start
        density_sparse = make_sparse(density, coords, device)
        fixed_sparse = make_sparse(torch.sigmoid(density_normalized), coords, device)
        torch.cuda.synchronize(device)
        timings['denormalize_and_sparse_assembly_s'] = time.perf_counter() - start
        start = time.perf_counter()
        percentile_image = percentile_visualizer.visualize_sample(
            density_sparse
        )['density_field'][0].cpu()
        fixed_image = fixed_visualizer.visualize_sample(fixed_sparse)['d_tri'][0].cpu()
        torch.cuda.synchronize(device)
        timings['render_s'] = time.perf_counter() - start
        rendered_percentile.append(percentile_image)
        rendered_fixed.append(fixed_image)
        start = time.perf_counter()
        save_image_grid(
            fixed_image[None], output_dir / f'{index:02d}_{sha256}_density_fixed.jpg'
        )
        save_image_grid(
            percentile_image[None],
            output_dir / f'{index:02d}_{sha256}_density_percentile.jpg',
        )
        timings['image_save_s'] = time.perf_counter() - start
        start = time.perf_counter()
        metrics = {}
        context_metrics = normalized_metrics(
            result.samples[0], context_density_target
        )
        surface_metrics = normalized_metrics(
            surface_query_normalized, surface_density_target
        )
        metrics.update({f'context_{key}': value for key, value in context_metrics.items()})
        metrics.update({f'surface_query_{key}': value for key, value in surface_metrics.items()})
        if gt_density is not None:
            gt_density = torch.from_numpy(gt_density).to(device)
            gt_normalized = (
                gt_density - float(stats['mean'])
            ) / float(stats['std'])
            voxel_metrics = normalized_metrics(density_normalized, gt_normalized)
            metrics.update({f'voxel_query_{key}': value for key, value in voxel_metrics.items()})
            gt_fixed = fixed_visualizer.visualize_sample(
                make_sparse(torch.sigmoid(gt_normalized), coords, device)
            )['d_tri'][0].cpu()
            gt_percentile = percentile_visualizer.visualize_sample(
                make_sparse(gt_density, coords, device)
            )['density_field'][0].cpu()
            save_image_grid(
                torch.stack([gt_fixed, fixed_image]),
                output_dir / f'{index:02d}_{sha256}_gt_pred_fixed.jpg',
            )
            save_image_grid(
                torch.stack([gt_percentile, percentile_image]),
                output_dir / f'{index:02d}_{sha256}_gt_pred_percentile.jpg',
            )
        timings['metrics_and_gt_render_s'] = time.perf_counter() - start
        timings['sample_total_s'] = time.perf_counter() - sample_start
        records.append({
            'index': index,
            'sha256': str(sha256),
            'mesh': str(mesh_path),
            'support_voxels': int(len(coords)),
            'density_min': float(density.min()),
            'density_mean': float(density.mean()),
            'density_std': float(density.std(correction=0)),
            'density_max': float(density.max()),
            'sampling_constraint_steps': constraint_step_stats,
            'context_bounds_min': points.amin(dim=1)[0].float().cpu().tolist(),
            'context_bounds_max': points.amax(dim=1)[0].float().cpu().tolist(),
            'voxel_query_bounds_min': query_points_tensor.amin(dim=1)[0].float().cpu().tolist(),
            'voxel_query_bounds_max': query_points_tensor.amax(dim=1)[0].float().cpu().tolist(),
            'voxel_to_context_nearest_mean': float(nearest_context_distance.mean()),
            'voxel_to_context_nearest_p95': float(torch.quantile(nearest_context_distance, 0.95)),
            **metrics,
            'timings': timings,
        })
        print(json.dumps(records[-1]), flush=True)

    start = time.perf_counter()
    save_image_grid(torch.stack(rendered_fixed), output_dir / 'density_grid_fixed.jpg')
    save_image_grid(
        torch.stack(rendered_percentile), output_dir / 'density_grid_percentile.jpg'
    )
    final_grid_save_s = time.perf_counter() - start
    summary = {
        'checkpoint': str(ckpt_path),
        'num_latents': model.num_latents,
        'randomize_latent_points_each_step': args.randomize_latent_points_each_step,
        'stats': str(stats_path),
        'voxel_root': str(voxel_root),
        'gt_voxel_root': str(gt_voxel_root),
        'mesh_dir': str(Path(args.mesh_dir).resolve()) if args.mesh_dir else None,
        'output_dir': str(output_dir),
        'density_payload_dir': (
            str(density_payload_dir) if density_payload_dir is not None else None
        ),
        'sampling_density_constraints': {
            'order': (
                'shift_mean_then_std_then_upper_clamp'
                if constraint_values['target_std'] is not None
                else 'shift_mean_then_upper_clamp'
            ),
            'base_resolution': args.sampling_density_base_resolution,
            **constraint_values,
        },
        'setup_s': setup_s,
        'setup_timings': setup_timings,
        'final_grid_save_s': final_grid_save_s,
        'total_s': time.perf_counter() - total_start,
        'samples': records,
    }
    with open(output_dir / 'summary.json', 'w') as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
