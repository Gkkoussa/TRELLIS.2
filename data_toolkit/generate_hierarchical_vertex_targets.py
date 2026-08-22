#!/usr/bin/env python3
"""Generate complete-chain vertex occupancy targets without support reassignment.

Each mesh is processed jointly at every requested 2x resolution.  A referenced
source vertex is retained only when its exact voxel exists in the triangle-field
support at every resolution.  Retained vertices become positive voxel targets;
present voxels belonging only to rejected vertices become ignore targets.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from check_triangle_field_vertex_support import (
    coord_keys,
    find_payload,
    load_float32_normalized_referenced_vertices,
    load_support_coords,
    load_worklist,
)


DEFAULT_RESOLUTIONS = (32, 64, 128, 256, 512)
RECORD_COLUMNS = (
    "sha256",
    "hierarchical_vertex_targets_generated",
    "hierarchical_vertex_resolution",
    "num_hierarchical_vertex_voxels",
    "num_ignored_vertex_voxels",
    "num_retained_source_vertices",
    "num_excluded_source_vertices",
    "num_source_vertices",
    "hierarchical_vertex_retention_fraction",
    "hierarchical_vertex_elapsed_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one-per-voxel vertex targets from source vertices whose "
            "exact voxel exists throughout a complete 2x resolution hierarchy."
        )
    )
    parser.add_argument("dataset", choices=["ObjaverseXL"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pbr_dump_root", type=Path, default=None)
    parser.add_argument("--triangle_field_voxel_root", type=Path, default=None)
    parser.add_argument("--hierarchical_vertex_target_root", type=Path, default=None)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument(
        "--instances",
        type=Path,
        default=None,
        help="Optional newline-separated SHA256 restriction.",
    )
    parser.add_argument("--zstd_level", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_resolutions(values: list[int]) -> tuple[int, ...]:
    resolutions = tuple(sorted(int(value) for value in values))
    if not resolutions:
        raise ValueError("--resolutions must not be empty")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("--resolutions contains duplicates")
    for coarse, fine in zip(resolutions, resolutions[1:]):
        if fine != coarse * 2:
            raise ValueError(
                "--resolutions must be a consecutive 2x hierarchy, got "
                f"{coarse} -> {fine}"
            )
    return resolutions


def target_dir(root: Path, resolution: int) -> Path:
    return root / f"hierarchical_vertex_targets_{resolution}"


def output_path(root: Path, resolution: int, sha256: str) -> Path:
    return target_dir(root, resolution) / f"{sha256}.npz.zst"


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


def unique_coords(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.unique(np.asarray(coords, dtype=np.int32), axis=0)


def subtract_coords(
    coords: np.ndarray,
    remove: np.ndarray,
    resolution: int,
) -> np.ndarray:
    if len(coords) == 0 or len(remove) == 0:
        return coords
    keep = ~np.isin(
        coord_keys(coords, resolution),
        coord_keys(remove, resolution),
        assume_unique=False,
    )
    return coords[keep]


def source_vertex_hierarchy(
    vertices: np.ndarray,
    resolutions: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Quantize once at the finest level, then derive exact integer parents."""
    finest = resolutions[-1]
    finest_coords = np.floor(
        (vertices.astype(np.float64) + 0.5) * finest
    ).astype(np.int64)
    finest_coords = np.clip(finest_coords, 0, finest - 1)
    return {
        resolution: (finest_coords // (finest // resolution)).astype(np.int32)
        for resolution in resolutions
    }


def read_existing_record(
    path: Path,
    sha256: str,
    resolution: int,
) -> dict[str, object]:
    from check_triangle_field_vertex_support import load_npz

    with load_npz(path) as payload:
        required = {
            "vertex_voxel_coords",
            "ignored_vertex_voxel_coords",
            "metadata_json",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing arrays {sorted(missing)}")
        positives = np.asarray(payload["vertex_voxel_coords"])
        ignored = np.asarray(payload["ignored_vertex_voxel_coords"])
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    return {
        "sha256": sha256,
        "hierarchical_vertex_targets_generated": True,
        "hierarchical_vertex_resolution": resolution,
        "num_hierarchical_vertex_voxels": int(len(positives)),
        "num_ignored_vertex_voxels": int(len(ignored)),
        "num_retained_source_vertices": int(metadata["num_retained_source_vertices"]),
        "num_excluded_source_vertices": int(metadata["num_excluded_source_vertices"]),
        "num_source_vertices": int(metadata["num_source_vertices"]),
        "hierarchical_vertex_retention_fraction": float(
            metadata["source_vertex_retention_fraction"]
        ),
        "hierarchical_vertex_elapsed_seconds": 0.0,
    }


def process_one(
    sha256: str,
    *,
    pbr_root: Path,
    triangle_field_root: Path,
    output_root: Path,
    resolutions: tuple[int, ...],
    zstd_level: int,
    overwrite: bool,
) -> dict[int, dict[str, object]]:
    outputs = {
        resolution: output_path(output_root, resolution, sha256)
        for resolution in resolutions
    }
    existing = {resolution: path.is_file() for resolution, path in outputs.items()}
    if all(existing.values()) and not overwrite:
        return {
            resolution: read_existing_record(path, sha256, resolution)
            for resolution, path in outputs.items()
        }
    if any(existing.values()) and not overwrite:
        present = [resolution for resolution, value in existing.items() if value]
        absent = [resolution for resolution, value in existing.items() if not value]
        raise RuntimeError(
            f"partial existing target hierarchy for {sha256}; present={present}, "
            f"absent={absent}. Rerun this instance with --overwrite."
        )

    started = time.perf_counter()
    vertices, _, num_valid_triangles = (
        load_float32_normalized_referenced_vertices(
            pbr_root / "pbr_dumps" / f"{sha256}.pickle"
        )
    )
    vertex_coords = source_vertex_hierarchy(vertices, resolutions)

    support_coords: dict[int, np.ndarray] = {}
    source_support_paths: dict[int, str] = {}
    present_by_resolution: dict[int, np.ndarray] = {}
    complete_chain = np.ones(len(vertices), dtype=bool)
    for resolution in resolutions:
        support_root = triangle_field_root / f"triangle_field_voxels_{resolution}"
        support_path = find_payload(support_root, sha256)
        if support_path is None:
            raise FileNotFoundError(
                f"missing R{resolution} triangle-field support for {sha256}"
            )
        coords = load_support_coords(support_path, resolution)
        support_coords[resolution] = coords
        source_support_paths[resolution] = str(support_path)
        present = np.isin(
            coord_keys(vertex_coords[resolution], resolution),
            coord_keys(coords, resolution),
            assume_unique=False,
        )
        present_by_resolution[resolution] = present
        complete_chain &= present

    retained_count = int(complete_chain.sum())
    source_count = int(len(vertices))
    excluded_count = source_count - retained_count
    if retained_count == 0:
        raise ValueError(f"{sha256} has no source vertex with a complete support chain")

    positive_coords: dict[int, np.ndarray] = {}
    ignored_coords: dict[int, np.ndarray] = {}
    for resolution in resolutions:
        positives = unique_coords(vertex_coords[resolution][complete_chain])
        excluded_present = (~complete_chain) & present_by_resolution[resolution]
        ignored = unique_coords(vertex_coords[resolution][excluded_present])
        ignored = subtract_coords(ignored, positives, resolution)
        positive_coords[resolution] = positives
        ignored_coords[resolution] = ignored

        support_keys = coord_keys(support_coords[resolution], resolution)
        if not np.isin(
            coord_keys(positives, resolution), support_keys, assume_unique=False
        ).all():
            raise AssertionError(f"R{resolution} positive target outside support")
        if len(ignored) and not np.isin(
            coord_keys(ignored, resolution), support_keys, assume_unique=False
        ).all():
            raise AssertionError(f"R{resolution} ignore target outside support")

    for coarse, fine in zip(resolutions, resolutions[1:]):
        expected_coarse = unique_coords(positive_coords[fine] // 2)
        if not np.array_equal(expected_coarse, positive_coords[coarse]):
            raise AssertionError(
                f"hierarchy mismatch for {sha256}: R{coarse} targets do not "
                f"equal parents of R{fine} targets"
            )

    elapsed = float(time.perf_counter() - started)
    records: dict[int, dict[str, object]] = {}
    for resolution in resolutions:
        positives = positive_coords[resolution]
        ignored = ignored_coords[resolution]
        centers = (
            (positives.astype(np.float64) + 0.5) / resolution - 0.5
        ).astype(np.float32)
        summary = {
            "sha256": sha256,
            "resolution": resolution,
            "hierarchy_resolutions": list(resolutions),
            "normalization": "voxelizer-compatible float32",
            "quantization": "finest floor voxel followed by exact integer parents",
            "support_assignment": "exact containing voxel only; no reassignment",
            "representative_position": "active_voxel_center",
            "vertex_population": "vertices referenced by finite non-degenerate triangles",
            "num_source_vertices": source_count,
            "num_valid_triangles": num_valid_triangles,
            "num_retained_source_vertices": retained_count,
            "num_excluded_source_vertices": excluded_count,
            "source_vertex_retention_fraction": retained_count / source_count,
            "num_positive_vertex_voxels": int(len(positives)),
            "num_ignored_vertex_voxels": int(len(ignored)),
            "num_active_support_voxels": int(len(support_coords[resolution])),
            "source_triangle_field_path": source_support_paths[resolution],
            "elapsed_seconds_all_resolutions": elapsed,
        }
        payload = encode_payload(
            {
                "vertices": centers,
                "vertex_voxel_coords": positives.astype(np.int32),
                "ignored_vertex_voxel_coords": ignored.astype(np.int32),
                "resolution": np.asarray(resolution, dtype=np.int32),
                "hierarchy_resolutions": np.asarray(resolutions, dtype=np.int32),
                "num_source_vertices": np.asarray(source_count, dtype=np.int64),
                "num_retained_source_vertices": np.asarray(
                    retained_count, dtype=np.int64
                ),
                "num_excluded_source_vertices": np.asarray(
                    excluded_count, dtype=np.int64
                ),
                "metadata_json": np.asarray(json.dumps(summary)),
            },
            zstd_level,
        )
        atomic_write(outputs[resolution], payload)
        records[resolution] = {
            "sha256": sha256,
            "hierarchical_vertex_targets_generated": True,
            "hierarchical_vertex_resolution": resolution,
            "num_hierarchical_vertex_voxels": int(len(positives)),
            "num_ignored_vertex_voxels": int(len(ignored)),
            "num_retained_source_vertices": retained_count,
            "num_excluded_source_vertices": excluded_count,
            "num_source_vertices": source_count,
            "hierarchical_vertex_retention_fraction": retained_count / source_count,
            "hierarchical_vertex_elapsed_seconds": elapsed,
        }
    return records


def main() -> None:
    args = parse_args()
    resolutions = validate_resolutions(args.resolutions)
    if args.world_size <= 0 or args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("Require 0 <= rank < world_size")
    if args.overwrite and args.instances is None:
        raise ValueError("--overwrite requires a targeted --instances file")
    if args.zstd_level < 1:
        raise ValueError("--zstd_level must be positive")

    pbr_root = args.pbr_dump_root or args.root
    triangle_field_root = args.triangle_field_voxel_root or args.root
    output_root = args.hierarchical_vertex_target_root or args.root
    for resolution in resolutions:
        directory = target_dir(output_root, resolution)
        (directory / "new_records").mkdir(parents=True, exist_ok=True)
        (directory / "merged_records").mkdir(parents=True, exist_ok=True)
    errors_root = output_root / "hierarchical_vertex_target_errors"
    errors_root.mkdir(parents=True, exist_ok=True)

    worklist = load_worklist(
        args.root,
        resolutions,
        args.instances,
        None,
    )
    start = len(worklist) * args.rank // args.world_size
    end = len(worklist) * (args.rank + 1) // args.world_size
    shard = worklist[start:end]
    print(
        f"Hierarchical vertex targets resolutions={list(resolutions)} "
        f"rank={args.rank}/{args.world_size} processing={len(shard)} "
        f"global_candidates={len(worklist)} output_root={output_root}",
        flush=True,
    )

    records = {resolution: [] for resolution in resolutions}
    errors: list[dict[str, str]] = []
    for sha256 in tqdm(shard, desc=f"Hierarchical targets shard {args.rank}"):
        try:
            result = process_one(
                sha256,
                pbr_root=pbr_root,
                triangle_field_root=triangle_field_root,
                output_root=output_root,
                resolutions=resolutions,
                zstd_level=args.zstd_level,
                overwrite=args.overwrite,
            )
            for resolution in resolutions:
                records[resolution].append(result[resolution])
        except Exception as exc:
            errors.append({"sha256": sha256, "error": repr(exc)})
            if args.verbose:
                print(f"FAILED {sha256}: {exc!r}", flush=True)

    for resolution in resolutions:
        record_path = (
            target_dir(output_root, resolution)
            / "new_records"
            / f"part_{args.rank}.csv"
        )
        pd.DataFrame(records[resolution], columns=RECORD_COLUMNS).to_csv(
            record_path, index=False
        )
    error_path = errors_root / f"part_{args.rank}.csv"
    pd.DataFrame(errors, columns=["sha256", "error"]).to_csv(
        error_path, index=False
    )
    print(
        f"Finished rank {args.rank}: succeeded={len(shard) - len(errors)} "
        f"failed={len(errors)} errors={error_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
