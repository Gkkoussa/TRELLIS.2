#!/usr/bin/env python3
"""Build voxel-constrained QEM meshes as a sharded TRELLIS dataset stage."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from visualize_voxel_constrained_qem import (
    coords_to_keys,
    load_normalized_pbr_mesh,
    load_voxel_payload,
    unique_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse same-active-voxel mesh edges using QEM."
    )
    parser.add_argument("dataset", choices=["ObjaverseXL"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pbr_dump_root", type=Path, default=None)
    parser.add_argument("--triangle_field_voxel_root", type=Path, default=None)
    parser.add_argument("--qem_edge_collapsed_root", type=Path, default=None)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--instances", default=None)
    parser.add_argument("--boundary_weight", type=float, default=1.0)
    parser.add_argument(
        "--max_support_distance",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--zstd_level", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def read_instances(value: str | None) -> set[str] | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        instances = {
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        }
    else:
        if "/" in value or "\\" in value or path.suffix:
            raise FileNotFoundError(f"Instances file does not exist: {path}")
        instances = {item.strip() for item in value.split(",") if item.strip()}
    invalid = sorted(
        item for item in instances if re.fullmatch(r"[0-9a-fA-F]{64}", item) is None
    )
    if invalid:
        raise ValueError(f"Invalid SHA256 instance identifier(s): {invalid[:5]}")
    return instances


def load_worklist(
    root: Path,
    pbr_root: Path,
    voxel_root: Path,
    output_root: Path,
    instances: set[str] | None,
    overwrite: bool,
) -> list[str]:
    base_path = root / "metadata.csv"
    voxel_metadata_path = voxel_root / "metadata.csv"
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base metadata: {base_path}")
    if not voxel_metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing finalized triangle-field metadata: {voxel_metadata_path}"
        )

    base = pd.read_csv(base_path)
    voxel = pd.read_csv(voxel_metadata_path)
    if "sha256" not in base or "sha256" not in voxel:
        raise ValueError("Both metadata files must contain sha256")
    if "triangle_field_voxelized" in voxel:
        voxel = voxel[truthy_series(voxel["triangle_field_voxelized"])]
    if "num_triangle_field_voxels" in voxel:
        voxel = voxel[
            pd.to_numeric(voxel["num_triangle_field_voxels"], errors="coerce").fillna(0)
            > 0
        ]

    available = set(base["sha256"].astype(str)) & set(voxel["sha256"].astype(str))
    if "pbr_dumped" in base:
        available &= set(base.loc[truthy_series(base["pbr_dumped"]), "sha256"].astype(str))
    if instances is not None:
        available &= instances

    shas = sorted(
        sha
        for sha in available
        if (pbr_root / "pbr_dumps" / f"{sha}.pickle").is_file()
    )
    if not overwrite:
        completed_path = output_root / "metadata.csv"
        if completed_path.is_file():
            completed = pd.read_csv(completed_path)
            if "qem_edge_collapsed" in completed:
                done = set(
                    completed.loc[
                        truthy_series(completed["qem_edge_collapsed"]), "sha256"
                    ].astype(str)
                )
                shas = [sha for sha in shas if sha not in done]
    return shas


def assign_vertices_to_support(
    vertices: np.ndarray,
    support_coords: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, int, int]:
    vertex_coords = np.floor((vertices + 0.5) * resolution).astype(np.int32)
    vertex_coords = np.clip(vertex_coords, 0, resolution - 1)
    support_coords = np.asarray(support_coords, dtype=np.int32)
    if support_coords.shape[0] == 0:
        raise ValueError("cannot assign vertices to empty sparse support")
    support_keys = {
        int(key): coord
        for key, coord in zip(coords_to_keys(support_coords, resolution), support_coords)
    }

    reassigned = 0
    maximum_used = 0
    nearest_shell_cache: dict[tuple[int, int, int], np.ndarray] = {}
    for index, coord in enumerate(vertex_coords):
        key = int((int(coord[0]) * resolution + int(coord[1])) * resolution + int(coord[2]))
        if key in support_keys:
            continue
        coord_key = (int(coord[0]), int(coord[1]), int(coord[2]))
        candidate_array = nearest_shell_cache.get(coord_key)
        if candidate_array is None:
            chebyshev_distances = np.max(
                np.abs(support_coords.astype(np.int64) - coord.astype(np.int64)),
                axis=1,
            )
            nearest_distance = int(chebyshev_distances.min())
            candidate_array = support_coords[chebyshev_distances == nearest_distance]
            nearest_shell_cache[coord_key] = candidate_array
        else:
            nearest_distance = int(
                np.max(
                    np.abs(
                        candidate_array[0].astype(np.int64) - coord.astype(np.int64)
                    )
                )
            )

        centers = (candidate_array.astype(np.float64) + 0.5) / resolution - 0.5
        selected = candidate_array[
            int(np.argmin(np.sum((centers - vertices[index]) ** 2, axis=1)))
        ]
        maximum_used = max(maximum_used, nearest_distance)
        vertex_coords[index] = selected
        reassigned += 1
    return vertex_coords, reassigned, maximum_used


def compute_normals_and_areas(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    raw = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(raw, axis=1)
    if faces.shape[0] == 0 or not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("collapsed mesh contains empty, non-finite, or degenerate faces")
    face_normals = raw / lengths[:, None]
    face_areas = 0.5 * lengths
    vertex_accumulator = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_accumulator, faces[:, corner], raw)
    vertex_lengths = np.linalg.norm(vertex_accumulator, axis=1)
    vertex_normals = np.zeros_like(vertex_accumulator)
    valid = vertex_lengths > 1e-30
    vertex_normals[valid] = vertex_accumulator[valid] / vertex_lengths[valid, None]
    return vertex_normals, face_normals, face_areas


def voxel_qem_collapse(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_voxels: np.ndarray,
    support_coords: np.ndarray,
    resolution: int,
    boundary_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Replace every vertex group with the center of its assigned active voxel."""
    unique_voxels, inverse = np.unique(vertex_voxels, axis=0, return_inverse=True)
    support_keys = set(int(key) for key in coords_to_keys(support_coords, resolution))
    group_keys = coords_to_keys(unique_voxels, resolution)
    if any(int(key) not in support_keys for key in group_keys):
        raise ValueError("a vertex group is assigned outside the saved sparse support")

    representatives = (
        (unique_voxels.astype(np.float64) + 0.5) / resolution - 0.5
    )

    remapped_faces = inverse[faces]
    valid = (
        (remapped_faces[:, 0] != remapped_faces[:, 1])
        & (remapped_faces[:, 1] != remapped_faces[:, 2])
        & (remapped_faces[:, 2] != remapped_faces[:, 0])
    )
    removed_degenerate_indices = int((~valid).sum())
    remapped_faces = remapped_faces[valid]
    if remapped_faces.shape[0] == 0:
        raise ValueError("voxel collapse removed every face")

    triangles = representatives[remapped_faces]
    area2 = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    valid_geometry = np.isfinite(area2) & (area2 > 1e-14)
    removed_degenerate_geometry = int((~valid_geometry).sum())
    remapped_faces = remapped_faces[valid_geometry]
    if remapped_faces.shape[0] == 0:
        raise ValueError("voxel collapse produced no non-degenerate faces")

    canonical_faces = np.sort(remapped_faces, axis=1)
    _, first = np.unique(canonical_faces, axis=0, return_index=True)
    first = np.sort(first)
    removed_duplicate_faces = int(len(remapped_faces) - len(first))
    remapped_faces = remapped_faces[first]

    used = np.unique(remapped_faces.reshape(-1))
    compact_remap = np.full(len(representatives), -1, dtype=np.int64)
    compact_remap[used] = np.arange(len(used), dtype=np.int64)
    out_vertices = representatives[used]
    out_voxels = unique_voxels[used]
    out_faces = compact_remap[remapped_faces]
    out_edges = unique_edges(out_faces)

    recovered_voxels = np.floor(
        (out_vertices + 0.5) * resolution
    ).astype(np.int32)
    if not np.array_equal(recovered_voxels, out_voxels):
        mismatch = np.flatnonzero(np.any(recovered_voxels != out_voxels, axis=1))
        raise ValueError(
            f"{len(mismatch)} collapsed representatives moved outside their assigned voxel"
        )
    output_keys = coords_to_keys(out_voxels, resolution)
    if any(int(key) not in support_keys for key in output_keys):
        raise ValueError("a collapsed representative lies outside saved sparse support")
    _, occupancy = np.unique(out_voxels, axis=0, return_counts=True)
    if occupancy.max(initial=0) > 1:
        raise ValueError("voxel collapse failed to produce at most one vertex per voxel")

    return out_vertices, out_faces, out_edges, out_voxels, {
        "num_input_vertex_groups": int(len(unique_voxels)),
        "num_group_collapses": int(len(vertices) - len(unique_voxels)),
        "num_unused_representatives_removed": int(len(unique_voxels) - len(used)),
        "num_degenerate_index_faces_removed": removed_degenerate_indices,
        "num_degenerate_geometry_faces_removed": removed_degenerate_geometry,
        "num_duplicate_faces_removed": removed_duplicate_faces,
    }


def encode_payload(arrays: dict[str, np.ndarray], zstd_level: int) -> bytes:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError("Writing .npz.zst requires zstandard") from exc
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return zstd.ZstdCompressor(level=zstd_level).compress(buffer.getvalue())


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def process_one(
    sha256: str,
    pbr_root: Path,
    voxel_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.perf_counter()
    vertices, faces, normalization = load_normalized_pbr_mesh(
        pbr_root / "pbr_dumps" / f"{sha256}.pickle"
    )
    support_coords, _, source_voxel_path = load_voxel_payload(voxel_root, sha256)
    vertex_voxels, reassigned, max_distance_used = assign_vertices_to_support(
        vertices,
        support_coords,
        args.resolution,
    )
    out_vertices, out_faces, out_edges, out_voxels, collapse_stats = voxel_qem_collapse(
        vertices,
        faces,
        vertex_voxels,
        support_coords,
        args.resolution,
        args.boundary_weight,
    )
    if out_faces.shape[0] == 0:
        raise ValueError("QEM produced no faces")
    vertex_normals, face_normals, face_areas = compute_normals_and_areas(
        out_vertices, out_faces
    )
    local_offsets = (out_vertices + 0.5) * args.resolution - out_voxels

    summary = {
        "sha256": sha256,
        "resolution": args.resolution,
        "source_voxel_path": str(source_voxel_path),
        "source_num_vertices": int(vertices.shape[0]),
        "source_num_faces": int(faces.shape[0]),
        "source_num_edges": int(unique_edges(faces).shape[0]),
        "num_active_voxels": int(support_coords.shape[0]),
        "vertices_reassigned_to_active_support": int(reassigned),
        "max_support_assignment_distance_used": int(max_distance_used),
        "num_collapses": collapse_stats["num_group_collapses"],
        "num_qem_vertices": int(out_vertices.shape[0]),
        "num_qem_faces": int(out_faces.shape[0]),
        "num_qem_edges": int(out_edges.shape[0]),
        "boundary_weight": float(args.boundary_weight),
        "representative_position": "active_voxel_center",
        "elapsed_seconds": float(time.perf_counter() - started),
        **collapse_stats,
        **normalization,
    }
    payload = encode_payload(
        {
            "vertices": out_vertices.astype(np.float32),
            "faces": out_faces.astype(np.int32),
            "edges": out_edges.astype(np.int32),
            "vertex_normals": vertex_normals.astype(np.float32),
            "face_normals": face_normals.astype(np.float32),
            "face_areas": face_areas.astype(np.float32),
            "vertex_voxel_coords": out_voxels.astype(np.int32),
            "vertex_local_offsets": local_offsets.astype(np.float32),
            "metadata_json": np.asarray(json.dumps(summary)),
        },
        args.zstd_level,
    )
    atomic_write(output_root / f"{sha256}.npz.zst", payload)
    return {
        "sha256": sha256,
        "qem_edge_collapsed": True,
        "qem_resolution": args.resolution,
        "num_qem_vertices": summary["num_qem_vertices"],
        "num_qem_faces": summary["num_qem_faces"],
        "num_qem_edges": summary["num_qem_edges"],
        "num_qem_collapses": summary["num_collapses"],
        "num_qem_reassigned_vertices": reassigned,
        "qem_max_support_distance_used": max_distance_used,
        "qem_elapsed_seconds": summary["elapsed_seconds"],
    }


def main() -> None:
    args = parse_args()
    if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
        raise ValueError("Require 0 <= rank < world_size")
    if args.overwrite and args.instances is None:
        raise ValueError("--overwrite requires a targeted --instances list")
    if args.max_support_distance is not None:
        print(
            "Ignoring legacy --max_support_distance; every absent vertex voxel "
            "is assigned to its nearest active support voxel.",
            flush=True,
        )
    args.pbr_dump_root = args.pbr_dump_root or args.root
    args.triangle_field_voxel_root = args.triangle_field_voxel_root or args.root
    args.qem_edge_collapsed_root = args.qem_edge_collapsed_root or args.root

    voxel_root = (
        args.triangle_field_voxel_root
        / f"triangle_field_voxels_{args.resolution}"
    )
    output_root = (
        args.qem_edge_collapsed_root
        / f"qem_edge_collapsed_meshes_{args.resolution}"
    )
    records_root = output_root / "new_records"
    errors_root = output_root / "errors"
    records_root.mkdir(parents=True, exist_ok=True)
    errors_root.mkdir(parents=True, exist_ok=True)

    worklist = load_worklist(
        args.root,
        args.pbr_dump_root,
        voxel_root,
        output_root,
        read_instances(args.instances),
        args.overwrite,
    )
    start = len(worklist) * args.rank // args.world_size
    end = len(worklist) * (args.rank + 1) // args.world_size
    shard = worklist[start:end]
    print(
        f"QEM resolution={args.resolution} rank={args.rank}/{args.world_size} "
        f"processing={len(shard)} global_candidates={len(worklist)}",
        flush=True,
    )

    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for sha256 in tqdm(shard, desc=f"QEM shard {args.rank}"):
        try:
            records.append(
                process_one(
                    sha256,
                    args.pbr_dump_root,
                    voxel_root,
                    output_root,
                    args,
                )
            )
        except Exception as exc:
            errors.append({"sha256": sha256, "error": repr(exc)})
            if args.verbose:
                print(f"FAILED {sha256}: {exc!r}", flush=True)

    record_columns = [
        "sha256",
        "qem_edge_collapsed",
        "qem_resolution",
        "num_qem_vertices",
        "num_qem_faces",
        "num_qem_edges",
        "num_qem_collapses",
        "num_qem_reassigned_vertices",
        "qem_max_support_distance_used",
        "qem_elapsed_seconds",
    ]
    pd.DataFrame(records, columns=record_columns).to_csv(
        records_root / f"part_{args.rank}.csv", index=False
    )
    pd.DataFrame(errors, columns=["sha256", "error"]).to_csv(
        errors_root / f"part_{args.rank}.csv", index=False
    )
    print(
        f"Finished rank {args.rank}: succeeded={len(records)} failed={len(errors)} "
        f"records={records_root / f'part_{args.rank}.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
