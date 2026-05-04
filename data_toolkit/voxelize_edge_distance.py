import os
import sys
import importlib
import argparse
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

import o_voxel

from plot_edge_distance_hist import (
    apply_transform,
    build_global_mesh,
    load_active_voxel_centers,
    nearest_edge_distances,
    normalize_dump,
)
from voxelize_gaussian_distance import sanitize_dump_for_volumetric_convert, validate_dump_for_volumetric_convert


def encode_base_color(distances, kind, d_max, sigma, log_a):
    transformed = apply_transform(
        distances.astype(np.float32),
        kind=kind,
        d_max=d_max,
        sigma=sigma,
        log_a=log_a,
    )
    transformed = np.clip(transformed, 0.0, 1.0)
    encoded = np.round(transformed * 255.0).astype(np.uint8)
    return np.repeat(encoded[:, None], 3, axis=1)


def build_attr_dict(base_color):
    n = base_color.shape[0]
    return {
        'base_color': torch.from_numpy(base_color),
        'metallic': torch.zeros((n, 1), dtype=torch.uint8),
        'roughness': torch.full((n, 1), 128, dtype=torch.uint8),
        'alpha': torch.full((n, 1), 255, dtype=torch.uint8),
    }


def _edge_distance_voxelize(file, metadatum, pbr_dump_root, root, device):
    sha256 = metadatum['sha256']
    try:
        pack = {'sha256': sha256}
        dump = None
        vertices = None
        edges = None

        for res in opt.resolution:
            need_process = False
            out_path = os.path.join(root, f'edge_distance_voxels_{res}', f'{sha256}.vxz')

            if os.path.exists(out_path):
                try:
                    info = o_voxel.io.read_vxz_info(out_path)
                    pack[f'edge_distance_voxelized_{res}'] = True
                    pack[f'num_edge_distance_voxels_{res}'] = info['num_voxel']
                except Exception:
                    need_process = True
            else:
                need_process = True

            if not need_process:
                continue

            if dump is None:
                with open(os.path.join(pbr_dump_root, 'pbr_dumps', f'{sha256}.pickle'), 'rb') as f:
                    dump = pickle.load(f)
                dump = normalize_dump(dump)
                sanitize_dump_for_volumetric_convert(dump)
                validation = validate_dump_for_volumetric_convert(dump)
                if validation['errors']:
                    raise ValueError('volumetric validation failed: ' + '; '.join(validation['errors'][:8]))
                vertices, edges = build_global_mesh(dump)
                if vertices.numel() == 0 or edges.numel() == 0:
                    raise ValueError('mesh has no valid edges')

            coords, _ = o_voxel.convert.blender_dump_to_volumetric_attr(
                dump,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                mip_level_offset=0,
                verbose=False,
                timing=False,
            )
            if len(coords) == 0:
                attr = build_attr_dict(np.zeros((0, 3), dtype=np.uint8))
                o_voxel.io.write_vxz(out_path, coords, attr)
                pack[f'edge_distance_voxelized_{res}'] = True
                pack[f'num_edge_distance_voxels_{res}'] = 0
                continue

            points = load_active_voxel_centers(dump, res)
            distances = nearest_edge_distances(
                points=points,
                vertices=vertices,
                edges=edges,
                device=device,
                point_batch=opt.point_batch,
                edge_batch=opt.edge_batch,
            ).numpy()

            base_color = encode_base_color(
                distances,
                kind=opt.kind,
                d_max=opt.d_max,
                sigma=opt.sigma,
                log_a=opt.log_a,
            )
            attr = build_attr_dict(base_color)
            o_voxel.io.write_vxz(out_path, coords, attr)
            pack[f'edge_distance_voxelized_{res}'] = True
            pack[f'num_edge_distance_voxels_{res}'] = len(coords)

        return pack
    except Exception as e:
        print(f'Error voxelizing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to load pbr dumps')
    parser.add_argument('--edge_voxel_root', type=str, default=None,
                        help='Directory to save voxelized edge distances')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    parser.add_argument('--kind', type=str, default='linear',
                        choices=['linear', 'sqrt', 'log', 'exp', 'inv'])
    parser.add_argument('--d_max', type=float, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--log_a', type=float, default=100.0)
    parser.add_argument('--point_batch', type=int, default=8192)
    parser.add_argument('--edge_batch', type=int, default=16384)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.resolution = sorted([int(x) for x in opt.resolution.split(',')], reverse=True)
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.edge_voxel_root = opt.edge_voxel_root or opt.root

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.edge_voxel_root, f'edge_distance_voxels_{res}', 'new_records'), exist_ok=True)

    metadata_path = os.path.join(opt.root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(metadata_path).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(
            pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256')
        )
    if os.path.exists(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(
            pd.read_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')).set_index('sha256')
        )
    for res in opt.resolution:
        edge_meta_path = os.path.join(opt.edge_voxel_root, f'edge_distance_voxels_{res}', 'metadata.csv')
        if os.path.exists(edge_meta_path):
            edge_meta = pd.read_csv(edge_meta_path).set_index('sha256')
            edge_meta = edge_meta.rename(columns={
                'edge_distance_voxelized': f'edge_distance_voxelized_{res}',
                'num_edge_distance_voxels': f'num_edge_distance_voxels_{res}',
            })
            metadata = metadata.combine_first(edge_meta)
    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['pbr_dumped'] == True]
        mask = np.zeros(len(metadata), dtype=bool)
        for res in opt.resolution:
            col = f'edge_distance_voxelized_{res}'
            if col in metadata.columns:
                mask |= metadata[col] != True
            else:
                mask[:] = True
                break
        metadata = metadata[mask]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    print(f'Processing {len(metadata)} objects...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    func = partial(_edge_distance_voxelize, pbr_dump_root=opt.pbr_dump_root, root=opt.edge_voxel_root, device=device)
    voxelized = dataset_utils.foreach_instance(metadata, None, func, max_workers=opt.max_workers, no_file=True, desc='Voxelizing edge distances')

    if 'error' in voxelized.columns:
        errors = voxelized[voxelized['error'].notna()]
        if len(errors) > 0:
            with open('errors.txt', 'w') as f:
                f.write('\n'.join(errors['sha256'].tolist()))

    for res in opt.resolution:
        col_ok = f'edge_distance_voxelized_{res}'
        col_n = f'num_edge_distance_voxels_{res}'
        if col_ok in voxelized.columns:
            edge_meta = voxelized[voxelized[col_ok] == True]
            if len(edge_meta) > 0:
                edge_meta = edge_meta[['sha256', col_ok, col_n]].rename(columns={
                    col_ok: 'edge_distance_voxelized',
                    col_n: 'num_edge_distance_voxels',
                })
                edge_meta.to_csv(
                    os.path.join(opt.edge_voxel_root, f'edge_distance_voxels_{res}', 'new_records', f'part_{opt.rank}.csv'),
                    index=False,
                )
