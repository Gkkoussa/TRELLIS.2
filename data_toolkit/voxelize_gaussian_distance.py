import os
import sys
import importlib
import argparse
import pickle
import time
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

import o_voxel


EXPECTED_ATTR_LAYOUT = [['base_color', 3], ['emissive', 3]]
DEBUG_VERBOSE = False
BENCHMARK_ENABLED = False


def debug_log(message: str):
    if DEBUG_VERBOSE:
        print(f'[gaussian-debug] {message}', flush=True)


def benchmark_log(message: str):
    if BENCHMARK_ENABLED:
        print(f'[gaussian-benchmark] {message}', flush=True)


def _format_shape(value) -> str:
    if value is None:
        return 'None'
    shape = getattr(value, 'shape', None)
    dtype = getattr(value, 'dtype', None)
    return f'shape={shape}, dtype={dtype}'


def make_default_material_pack():
    return {
        'baseColorFactor': [1.0, 1.0, 1.0, 1.0],
        'alphaFactor': 1.0,
        'metallicFactor': 0.0,
        'roughnessFactor': 1.0,
        'alphaMode': 'OPAQUE',
        'alphaCutoff': 0.5,
        'baseColorTexture': None,
        'alphaTexture': None,
        'metallicTexture': None,
        'roughnessTexture': None,
    }


def sanitize_dump_for_volumetric_convert(dump):
    changes = []
    materials = dump.get('materials')
    if materials is None:
        dump['materials'] = []
        materials = dump['materials']
        changes.append('created empty materials list')

    if len(materials) == 0:
        materials.append(make_default_material_pack())
        changes.append('added default material because dump had no materials')

    num_materials = len(materials)
    for object_idx, obj in enumerate(dump.get('objects', [])):
        faces = obj.get('faces')
        mat_ids = obj.get('mat_ids')
        if not isinstance(faces, np.ndarray) or faces.ndim != 2:
            continue
        num_faces = faces.shape[0]

        if mat_ids is None:
            obj['mat_ids'] = np.zeros((num_faces,), dtype=np.int32)
            changes.append(f'object[{object_idx}]: created mat_ids filled with 0 for {num_faces} faces')
            continue

        if isinstance(mat_ids, np.ndarray) and mat_ids.shape == (num_faces,) and np.issubdtype(mat_ids.dtype, np.integer):
            invalid_mask = (mat_ids < 0) | (mat_ids >= num_materials)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                remapped = mat_ids.astype(np.int32, copy=True)
                remapped[invalid_mask] = 0
                obj['mat_ids'] = remapped
                changes.append(
                    f'object[{object_idx}]: remapped {invalid_count} invalid mat_ids to 0 (valid range 0..{num_materials - 1})'
                )

    return changes


def validate_dump_for_volumetric_convert(dump):
    report = {
        'num_materials': len(dump.get('materials', [])),
        'summaries': [],
        'warnings': [],
        'errors': [],
    }

    objects = dump.get('objects', [])
    if len(objects) == 0:
        report['errors'].append('dump has no objects after normalization')
        return report

    for object_idx, obj in enumerate(objects):
        prefix = f'object[{object_idx}]'
        vertices = obj.get('vertices')
        faces = obj.get('faces')
        normals = obj.get('normals')
        uvs = obj.get('uvs')
        mat_ids = obj.get('mat_ids')

        report['summaries'].append(
            f'{prefix}: '
            f'vertices={_format_shape(vertices)} '
            f'faces={_format_shape(faces)} '
            f'normals={_format_shape(normals)} '
            f'uvs={_format_shape(uvs)} '
            f'mat_ids={_format_shape(mat_ids)}'
        )

        if not isinstance(vertices, np.ndarray) or vertices.ndim != 2 or vertices.shape[1] != 3:
            report['errors'].append(f'{prefix}: vertices must have shape [N, 3], got {_format_shape(vertices)}')
            continue
        if vertices.shape[0] == 0:
            report['errors'].append(f'{prefix}: vertices is empty')
        elif not np.isfinite(vertices).all():
            report['errors'].append(f'{prefix}: vertices contains NaN/Inf values')

        if not isinstance(faces, np.ndarray) or faces.ndim != 2 or faces.shape[1] != 3:
            report['errors'].append(f'{prefix}: faces must have shape [F, 3], got {_format_shape(faces)}')
            continue
        num_faces = faces.shape[0]
        if num_faces == 0:
            report['errors'].append(f'{prefix}: faces is empty')
        else:
            if np.issubdtype(faces.dtype, np.integer):
                face_min = int(faces.min())
                face_max = int(faces.max())
                if face_min < 0:
                    report['errors'].append(f'{prefix}: faces contains negative vertex index {face_min}')
                if face_max >= vertices.shape[0]:
                    report['errors'].append(
                        f'{prefix}: faces references vertex index {face_max} but only {vertices.shape[0]} vertices exist'
                    )
            else:
                report['errors'].append(f'{prefix}: faces dtype must be integer, got {faces.dtype}')

        if not isinstance(normals, np.ndarray) or normals.shape != (num_faces, 3, 3):
            report['errors'].append(
                f'{prefix}: normals must have shape {(num_faces, 3, 3)}, got {_format_shape(normals)}'
            )
        elif not np.isfinite(normals).all():
            report['errors'].append(f'{prefix}: normals contains NaN/Inf values')

        if uvs is not None:
            if not isinstance(uvs, np.ndarray) or uvs.shape != (num_faces, 3, 2):
                report['errors'].append(
                    f'{prefix}: uvs must have shape {(num_faces, 3, 2)} when present, got {_format_shape(uvs)}'
                )
            elif not np.isfinite(uvs).all():
                report['errors'].append(f'{prefix}: uvs contains NaN/Inf values')
        else:
            report['warnings'].append(f'{prefix}: uvs is None; converter will substitute zeros')

        if not isinstance(mat_ids, np.ndarray) or mat_ids.shape != (num_faces,):
            report['errors'].append(
                f'{prefix}: mat_ids must have shape {(num_faces,)}, got {_format_shape(mat_ids)}'
            )
        elif not np.issubdtype(mat_ids.dtype, np.integer):
            report['errors'].append(f'{prefix}: mat_ids dtype must be integer, got {mat_ids.dtype}')
        elif num_faces > 0:
            mat_min = int(mat_ids.min())
            mat_max = int(mat_ids.max())
            if mat_min < 0:
                report['errors'].append(
                    f'{prefix}: mat_ids contains negative value {mat_min}; native converter may segfault on this'
                )
            if report['num_materials'] == 0:
                report['errors'].append(f'{prefix}: dump has no materials but faces reference material ids')
            elif mat_max >= report['num_materials']:
                report['errors'].append(
                    f'{prefix}: mat_ids references material {mat_max} but dump only has {report["num_materials"]} materials'
                )

    return report


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


def build_global_vertices(dump):
    vertices = []
    for obj in dump['objects']:
        v = torch.from_numpy(obj['vertices']).float()
        if v.numel() == 0:
            continue
        vertices.append(v)
    if not vertices:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.cat(vertices, dim=0)


def voxel_coords_to_centers(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.float()
    return (coords + 0.5) / resolution - 0.5


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


def nearest_vertex_distances(
    points: torch.Tensor,
    vertices: torch.Tensor,
    device: torch.device,
    point_batch: int,
    vertex_batch: int,
) -> torch.Tensor:
    if points.numel() == 0 or vertices.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)

    vertices = vertices.to(device, non_blocking=True)
    out = []
    for p0 in range(0, points.shape[0], point_batch):
        p = points[p0:p0 + point_batch].to(device, non_blocking=True)
        best = torch.full((p.shape[0],), float('inf'), device=device)
        for v0 in range(0, vertices.shape[0], vertex_batch):
            v = vertices[v0:v0 + vertex_batch]
            diff = p[:, None, :] - v[None, :, :]
            dist2 = (diff * diff).sum(dim=-1)
            best = torch.minimum(best, dist2.min(dim=1).values)
        out.append(best.sqrt().cpu())
    return torch.cat(out, dim=0)


def encode_gaussian_channels(distances: np.ndarray, sigma_values: np.ndarray) -> np.ndarray:
    transformed = 1.0 - np.exp(-0.5 * np.square(distances[:, None] / sigma_values[None, :]))
    transformed = np.clip(transformed, 0.0, 1.0)
    return np.round(transformed * 255.0).astype(np.uint8)


def build_attr_dict(edge_channels: np.ndarray, vertex_channels: np.ndarray):
    return {
        'base_color': torch.from_numpy(edge_channels),
        'emissive': torch.from_numpy(vertex_channels),
    }


def output_matches_expected_layout(out_path: str) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        info = o_voxel.io.read_vxz_info(out_path)
    except Exception:
        return False
    return info.get('attr') == EXPECTED_ATTR_LAYOUT


def _gaussian_distance_voxelize(file, metadatum, pbr_dump_root, root, device):
    sha256 = metadatum['sha256']
    try:
        object_start_t = time.perf_counter()
        stage_timings = {
            'prep_s': 0.0,
            'voxelize_s': 0.0,
            'coords_to_points_s': 0.0,
            'edge_distance_s': 0.0,
            'vertex_distance_s': 0.0,
            'encode_s': 0.0,
            'write_s': 0.0,
        }
        debug_log(f'{sha256}: start object')
        pack = {'sha256': sha256}
        dump = None
        vertices = None
        edges = None
        global_vertices = None

        for res in opt.resolution:
            debug_log(f'{sha256}: resolution={res} start')
            out_path = os.path.join(root, f'gaussian_distance_voxels_{res}', f'{sha256}.vxz')
            need_process = not output_matches_expected_layout(out_path)

            if not need_process:
                debug_log(f'{sha256}: resolution={res} existing output matches expected layout')
                info = o_voxel.io.read_vxz_info(out_path)
                pack[f'gaussian_distance_voxelized_{res}'] = True
                pack[f'num_gaussian_distance_voxels_{res}'] = info['num_voxel']
                debug_log(f'{sha256}: resolution={res} skip done num_voxel={info["num_voxel"]}')
                continue

            if dump is None:
                prep_start_t = time.perf_counter()
                debug_log(f'{sha256}: loading pbr dump')
                with open(os.path.join(pbr_dump_root, 'pbr_dumps', f'{sha256}.pickle'), 'rb') as f:
                    dump = pickle.load(f)
                debug_log(f'{sha256}: normalizing dump')
                dump = normalize_dump(dump)
                sanitize_changes = sanitize_dump_for_volumetric_convert(dump)
                for change in sanitize_changes:
                    debug_log(f'{sha256}: sanitize: {change}')
                validation = validate_dump_for_volumetric_convert(dump)
                for warning in validation['warnings']:
                    debug_log(f'{sha256}: validation warning: {warning}')
                if validation['errors']:
                    raise ValueError('volumetric validation failed: ' + '; '.join(validation['errors'][:8]))
                debug_log(f'{sha256}: building global mesh')
                vertices, edges = build_global_mesh(dump)
                debug_log(f'{sha256}: building global vertices')
                global_vertices = build_global_vertices(dump)
                debug_log(
                    f'{sha256}: mesh stats vertices={vertices.shape[0]} edges={edges.shape[0]} global_vertices={global_vertices.shape[0]}'
                )
                if vertices.numel() == 0 or edges.numel() == 0:
                    raise ValueError('mesh has no valid edges')
                if global_vertices.numel() == 0:
                    raise ValueError('mesh has no valid vertices')
                stage_timings['prep_s'] += time.perf_counter() - prep_start_t

            debug_log(f'{sha256}: resolution={res} voxelizing active coords')
            stage_start_t = time.perf_counter()
            coords, _ = o_voxel.convert.blender_dump_to_volumetric_attr(
                dump,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                mip_level_offset=0,
                verbose=False,
                timing=False,
            )
            stage_timings['voxelize_s'] += time.perf_counter() - stage_start_t
            debug_log(f'{sha256}: resolution={res} voxelized active coords count={len(coords)}')
            if len(coords) == 0:
                empty = np.zeros((0, 3), dtype=np.uint8)
                debug_log(f'{sha256}: resolution={res} writing empty vxz')
                stage_start_t = time.perf_counter()
                o_voxel.io.write_vxz(
                    out_path,
                    coords,
                    build_attr_dict(empty, empty),
                    compression=opt.compression,
                    compression_level=opt.compression_level,
                )
                stage_timings['write_s'] += time.perf_counter() - stage_start_t
                pack[f'gaussian_distance_voxelized_{res}'] = True
                pack[f'num_gaussian_distance_voxels_{res}'] = 0
                debug_log(f'{sha256}: resolution={res} empty vxz write done')
                continue

            debug_log(f'{sha256}: resolution={res} converting active coords to voxel centers')
            stage_start_t = time.perf_counter()
            points = voxel_coords_to_centers(coords, res)
            stage_timings['coords_to_points_s'] += time.perf_counter() - stage_start_t
            debug_log(f'{sha256}: resolution={res} active voxel centers loaded count={points.shape[0]}')
            debug_log(
                f'{sha256}: resolution={res} computing edge distances point_batch={opt.point_batch} edge_batch={opt.edge_batch}'
            )
            stage_start_t = time.perf_counter()
            edge_distances = nearest_edge_distances(
                points=points,
                vertices=vertices,
                edges=edges,
                device=device,
                point_batch=opt.point_batch,
                edge_batch=opt.edge_batch,
            ).numpy()
            stage_timings['edge_distance_s'] += time.perf_counter() - stage_start_t
            debug_log(f'{sha256}: resolution={res} edge distances done')
            debug_log(
                f'{sha256}: resolution={res} computing vertex distances point_batch={opt.point_batch} vertex_batch={opt.vertex_batch}'
            )
            stage_start_t = time.perf_counter()
            vertex_distances = nearest_vertex_distances(
                points=points,
                vertices=global_vertices,
                device=device,
                point_batch=opt.point_batch,
                vertex_batch=opt.vertex_batch,
            ).numpy()
            stage_timings['vertex_distance_s'] += time.perf_counter() - stage_start_t
            debug_log(f'{sha256}: resolution={res} vertex distances done')

            voxel_length = 1.0 / float(res)
            sigma_values = opt.sigma_multipliers * voxel_length
            debug_log(
                f'{sha256}: resolution={res} encoding channels voxel_length={voxel_length} sigma_values={sigma_values.tolist()}'
            )
            stage_start_t = time.perf_counter()
            edge_channels = encode_gaussian_channels(edge_distances.astype(np.float32), sigma_values)
            vertex_channels = encode_gaussian_channels(vertex_distances.astype(np.float32), sigma_values)
            stage_timings['encode_s'] += time.perf_counter() - stage_start_t
            debug_log(f'{sha256}: resolution={res} channel encoding done')

            debug_log(f'{sha256}: resolution={res} writing vxz {out_path}')
            stage_start_t = time.perf_counter()
            o_voxel.io.write_vxz(
                out_path,
                coords,
                build_attr_dict(edge_channels, vertex_channels),
                compression=opt.compression,
                compression_level=opt.compression_level,
            )
            stage_timings['write_s'] += time.perf_counter() - stage_start_t
            pack[f'gaussian_distance_voxelized_{res}'] = True
            pack[f'num_gaussian_distance_voxels_{res}'] = len(coords)
            debug_log(f'{sha256}: resolution={res} write done num_voxel={len(coords)}')

        if BENCHMARK_ENABLED:
            total_time_s = time.perf_counter() - object_start_t
            pack['benchmark_total_s'] = total_time_s
            for key, value in stage_timings.items():
                pack[f'benchmark_{key}'] = value
            benchmark_log(
                f'{sha256}: total={total_time_s:.3f}s '
                + ' '.join(f'{key}={value:.3f}s' for key, value in stage_timings.items())
            )
        debug_log(f'{sha256}: object done')
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
    parser.add_argument('--gaussian_distance_voxel_root', type=str, default=None,
                        help='Directory to save combined gaussian distance voxels')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    parser.add_argument('--sigma_multipliers', type=str, default='0.5,1.0,2.0',
                        help='Comma-separated sigma multipliers relative to voxel length')
    parser.add_argument('--point_batch', type=int, default=8192)
    parser.add_argument('--edge_batch', type=int, default=16384)
    parser.add_argument('--vertex_batch', type=int, default=16384)
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed per-object debug logging')
    parser.add_argument('--benchmark', action='store_true',
                        help='Print per-object timings and an aggregate timing summary')
    parser.add_argument('--compression', type=str, default='lzma',
                        choices=['none', 'deflate', 'lzma', 'zstd'],
                        help='Compression algorithm for output .vxz files')
    parser.add_argument('--compression_level', type=int, default=None,
                        help='Optional compression level override passed to write_vxz')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.resolution = sorted([int(x) for x in opt.resolution.split(',')], reverse=True)
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.gaussian_distance_voxel_root = opt.gaussian_distance_voxel_root or opt.root
    opt.sigma_multipliers = np.array([float(x) for x in opt.sigma_multipliers.split(',')], dtype=np.float32)
    if opt.sigma_multipliers.shape[0] != 3:
        raise ValueError('Expected exactly 3 sigma multipliers for the 3 edge and 3 vertex channels.')
    if np.any(opt.sigma_multipliers <= 0):
        raise ValueError('Sigma multipliers must be positive.')
    DEBUG_VERBOSE = bool(opt.verbose)
    BENCHMARK_ENABLED = bool(opt.benchmark)

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.gaussian_distance_voxel_root, f'gaussian_distance_voxels_{res}', 'new_records'), exist_ok=True)

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
        voxel_meta_path = os.path.join(opt.gaussian_distance_voxel_root, f'gaussian_distance_voxels_{res}', 'metadata.csv')
        if os.path.exists(voxel_meta_path):
            voxel_meta = pd.read_csv(voxel_meta_path).set_index('sha256')
            voxel_meta = voxel_meta.rename(columns={
                'gaussian_distance_voxelized': f'gaussian_distance_voxelized_{res}',
                'num_gaussian_distance_voxels': f'num_gaussian_distance_voxels_{res}',
            })
            metadata = metadata.combine_first(voxel_meta)
    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['pbr_dumped'] == True]
        mask = np.zeros(len(metadata), dtype=bool)
        for res in opt.resolution:
            col = f'gaussian_distance_voxelized_{res}'
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
    print(f'Sigma multipliers: {opt.sigma_multipliers.tolist()}')
    print(f'Output compression: {opt.compression}'
          + (f' (level {opt.compression_level})' if opt.compression_level is not None else ''))
    if len(metadata) > 0:
        debug_log(f'first sha256 in shard: {metadata.iloc[0]["sha256"]}')

    func = partial(
        _gaussian_distance_voxelize,
        pbr_dump_root=opt.pbr_dump_root,
        root=opt.gaussian_distance_voxel_root,
        device=device,
    )
    voxelized = dataset_utils.foreach_instance(
        metadata,
        None,
        func,
        max_workers=opt.max_workers,
        no_file=True,
        desc='Voxelizing gaussian distance features',
    )

    if 'error' in voxelized.columns:
        errors = voxelized[voxelized['error'].notna()]
        if len(errors) > 0:
            with open('gaussian_distance_errors.txt', 'w') as f:
                f.write('\n'.join(errors['sha256'].tolist()))

    if opt.benchmark and len(voxelized) > 0 and 'benchmark_total_s' in voxelized.columns:
        benchmark_cols = [
            'benchmark_total_s',
            'benchmark_prep_s',
            'benchmark_voxelize_s',
            'benchmark_coords_to_points_s',
            'benchmark_edge_distance_s',
            'benchmark_vertex_distance_s',
            'benchmark_encode_s',
            'benchmark_write_s',
        ]
        benchmark_df = voxelized[[c for c in benchmark_cols if c in voxelized.columns]].dropna(how='all')
        if len(benchmark_df) > 0:
            means = benchmark_df.mean(numeric_only=True)
            totals = benchmark_df.sum(numeric_only=True)
            benchmark_log(
                f'aggregate over {len(benchmark_df)} object(s): '
                + ' '.join(f'{col} mean={means[col]:.3f}s total={totals[col]:.3f}s' for col in benchmark_df.columns)
            )

    for res in opt.resolution:
        col_ok = f'gaussian_distance_voxelized_{res}'
        col_n = f'num_gaussian_distance_voxels_{res}'
        if col_ok in voxelized.columns:
            voxel_meta = voxelized[voxelized[col_ok] == True]
            if len(voxel_meta) > 0:
                voxel_meta = voxel_meta[['sha256', col_ok, col_n]].rename(columns={
                    col_ok: 'gaussian_distance_voxelized',
                    col_n: 'num_gaussian_distance_voxels',
                })
                voxel_meta.to_csv(
                    os.path.join(
                        opt.gaussian_distance_voxel_root,
                        f'gaussian_distance_voxels_{res}',
                        'new_records',
                        f'part_{opt.rank}.csv',
                    ),
                    index=False,
                )
