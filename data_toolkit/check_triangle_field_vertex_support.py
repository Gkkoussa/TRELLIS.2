#!/usr/bin/env python3
"""Audit whether source-mesh vertex voxels exist in triangle-field support.

The audit reproduces the triangle-field voxelizer's float32 normalization,
keeps vertices referenced by finite non-degenerate triangles, and tests

    floor((vertex + 0.5) * resolution)

against the saved ``coords`` array at each requested resolution.  It never
modifies PBR dumps or triangle-field payloads.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


DEFAULT_RESOLUTIONS = (32, 64, 128, 256, 512)

PER_MESH_FIELDS = (
    "sha256",
    "resolution",
    "status",
    "error",
    "num_referenced_vertices",
    "num_valid_triangles",
    "num_active_support_voxels",
    "num_represented_vertices",
    "num_missing_vertices",
    "represented_vertex_fraction",
    "num_unique_vertex_voxels",
    "num_unique_missing_vertex_voxels",
    "represented_unique_voxel_fraction",
)

EXAMPLE_FIELDS = (
    "sha256",
    "resolution",
    "referenced_vertex_index",
    "global_vertex_index",
    "vertex_x",
    "vertex_y",
    "vertex_z",
    "voxel_x",
    "voxel_y",
    "voxel_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether every normalized source-mesh vertex has its exact "
            "containing voxel in saved triangle-field sparse support."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Process one dataset shard.")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    audit.add_argument("--rank", type=int, default=0)
    audit.add_argument("--world_size", type=int, default=1)
    audit.add_argument("--num_workers", type=int, default=1)
    audit.add_argument("--chunksize", type=int, default=2)
    audit.add_argument(
        "--instances",
        type=Path,
        default=None,
        help="Optional newline-separated SHA256 restriction.",
    )
    audit.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sorted worklist limit before sharding.",
    )
    audit.add_argument(
        "--examples_per_mesh",
        type=int,
        default=3,
        help="Maximum missing-vertex examples saved per mesh and resolution.",
    )
    audit.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <root>/outputs/triangle_field_vertex_support_audit.",
    )

    finalize = subparsers.add_parser(
        "finalize", help="Merge completed shard CSV files and aggregate statistics."
    )
    finalize.add_argument("--output_dir", type=Path, required=True)
    finalize.add_argument("--world_size", type=int, required=True)
    return parser.parse_args()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_instances(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Instances file not found: {path}")
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    }


def load_worklist(
    root: Path,
    resolutions: Iterable[int],
    instances: Path | None,
    limit: int | None,
) -> list[str]:
    """Use the intersection of successful requested-resolution metadata."""
    allowed = read_instances(instances) if instances is not None else None
    pbr_root = root / "pbr_dumps"
    candidates: set[str] | None = None
    for resolution in resolutions:
        metadata_path = root / f"triangle_field_voxels_{resolution}" / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Triangle-field metadata not found for R{resolution}: {metadata_path}"
            )
        stage_instances: set[str] = set()
        with metadata_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "sha256" not in reader.fieldnames:
                raise ValueError(f"{metadata_path} must contain a sha256 column")
            for row in reader:
                if "triangle_field_voxelized" in row and not truthy(
                    row["triangle_field_voxelized"]
                ):
                    continue
                if "num_triangle_field_voxels" in row:
                    try:
                        if float(row["num_triangle_field_voxels"] or 0) <= 0:
                            continue
                    except ValueError:
                        continue
                sha256 = str(row["sha256"]).strip()
                if sha256:
                    stage_instances.add(sha256)
        if candidates is None:
            candidates = stage_instances
        else:
            candidates &= stage_instances

    if candidates is None:
        candidates = set()

    worklist = sorted(
        sha256
        for sha256 in candidates
        if (allowed is None or sha256 in allowed)
        and (pbr_root / f"{sha256}.pickle").is_file()
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        worklist = worklist[:limit]
    return worklist


def find_payload(root: Path, sha256: str) -> Path | None:
    for suffix in (".npz.zst", ".npz"):
        path = root / f"{sha256}{suffix}"
        if path.is_file():
            return path
    return None


def load_npz(path: Path):
    if path.name.endswith(".npz.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("Reading .npz.zst requires zstandard") from exc
        with path.open("rb") as handle:
            payload = zstd.ZstdDecompressor().decompress(handle.read())
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def load_support_coords(path: Path, resolution: int) -> np.ndarray:
    with load_npz(path) as payload:
        if "coords" not in payload:
            raise ValueError(f"{path} does not contain coords")
        coords = np.asarray(payload["coords"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} has invalid coords shape {coords.shape}")
    if not np.issubdtype(coords.dtype, np.integer):
        if not np.equal(coords, np.floor(coords)).all():
            raise ValueError(f"{path} contains non-integer coords")
    coords = coords.astype(np.int64, copy=False)
    if len(coords) == 0:
        raise ValueError(f"{path} contains empty sparse support")
    if (coords < 0).any() or (coords >= resolution).any():
        raise ValueError(f"{path} contains coords outside [0, {resolution})")
    keys = coord_keys(coords, resolution)
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"{path} contains duplicate coords")
    return coords


def load_float32_normalized_referenced_vertices(
    pbr_path: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Match voxelize_triangle_field.normalize_dump and QEM face filtering."""
    with pbr_path.open("rb") as handle:
        dump = pickle.load(handle)

    objects: list[tuple[np.ndarray, np.ndarray]] = []
    for object_index, obj in enumerate(dump.get("objects", [])):
        vertices = np.asarray(obj.get("vertices"))
        faces = np.asarray(obj.get("faces"))
        if vertices.size == 0 or faces.size == 0:
            continue
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"object[{object_index}] has invalid vertices shape {vertices.shape}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                f"object[{object_index}] has invalid faces shape {faces.shape}"
            )
        if not np.isfinite(vertices).all():
            raise ValueError(f"object[{object_index}] contains non-finite vertices")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError(f"object[{object_index}] faces are not integer typed")
        faces = faces.astype(np.int64, copy=False)
        if (faces < 0).any() or (faces >= len(vertices)).any():
            raise ValueError(f"object[{object_index}] has out-of-range face indices")
        objects.append((vertices.astype(np.float32, copy=False), faces))

    if not objects:
        raise ValueError("PBR dump contains no non-empty mesh objects")

    all_unnormalized = np.concatenate([vertices for vertices, _ in objects], axis=0)
    lower = all_unnormalized.min(axis=0)
    upper = all_unnormalized.max(axis=0)
    extent = np.max(upper - lower)
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("mesh bounding box is empty or invalid")
    center = (lower + upper) / np.float32(2.0)
    scale = np.float32(0.99999) / extent

    normalized_parts = []
    global_faces = []
    offset = 0
    for vertices, faces in objects:
        normalized = ((vertices - center) * scale).astype(np.float32, copy=False)
        normalized_parts.append(normalized)
        global_faces.append(faces + offset)
        offset += len(vertices)

    vertices = np.concatenate(normalized_parts, axis=0)
    faces = np.concatenate(global_faces, axis=0)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    area2 = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(area2) & (area2 > np.float32(1e-12))
    faces = faces[valid]
    if len(faces) == 0:
        raise ValueError("mesh has no finite non-degenerate triangles")

    referenced_global_indices = np.unique(faces.reshape(-1))
    return (
        vertices[referenced_global_indices],
        referenced_global_indices,
        int(len(faces)),
    )


def coord_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = coords.astype(np.int64, copy=False)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def empty_row(
    sha256: str,
    resolution: int,
    status: str,
    error: str,
    num_referenced_vertices: int = 0,
    num_valid_triangles: int = 0,
) -> dict[str, object]:
    return {
        "sha256": sha256,
        "resolution": resolution,
        "status": status,
        "error": error,
        "num_referenced_vertices": num_referenced_vertices,
        "num_valid_triangles": num_valid_triangles,
        "num_active_support_voxels": 0,
        "num_represented_vertices": 0,
        "num_missing_vertices": 0,
        "represented_vertex_fraction": "",
        "num_unique_vertex_voxels": 0,
        "num_unique_missing_vertex_voxels": 0,
        "represented_unique_voxel_fraction": "",
    }


def check_one(
    task: tuple[str, str, tuple[int, ...], int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root_string, sha256, resolutions, examples_per_mesh = task
    root = Path(root_string)
    try:
        vertices, global_indices, num_valid_triangles = (
            load_float32_normalized_referenced_vertices(
                root / "pbr_dumps" / f"{sha256}.pickle"
            )
        )
    except Exception as exc:
        rows = [
            empty_row(sha256, resolution, "mesh_error", repr(exc))
            for resolution in resolutions
        ]
        return rows, []

    rows = []
    examples = []
    for resolution in resolutions:
        support_root = root / f"triangle_field_voxels_{resolution}"
        payload_path = find_payload(support_root, sha256)
        if payload_path is None:
            rows.append(
                empty_row(
                    sha256,
                    resolution,
                    "missing_support",
                    f"No triangle-field payload under {support_root}",
                    len(vertices),
                    num_valid_triangles,
                )
            )
            continue
        try:
            support_coords = load_support_coords(payload_path, resolution)
        except Exception as exc:
            rows.append(
                empty_row(
                    sha256,
                    resolution,
                    "support_error",
                    repr(exc),
                    len(vertices),
                    num_valid_triangles,
                )
            )
            continue

        vertex_coords = np.floor(
            (vertices.astype(np.float64) + 0.5) * resolution
        ).astype(np.int64)
        vertex_coords = np.clip(vertex_coords, 0, resolution - 1)
        vertex_keys = coord_keys(vertex_coords, resolution)
        support_keys = coord_keys(support_coords, resolution)
        represented = np.isin(vertex_keys, support_keys, assume_unique=False)

        num_vertices = int(len(vertices))
        num_represented = int(represented.sum())
        num_missing = num_vertices - num_represented
        unique_vertex_keys = np.unique(vertex_keys)
        unique_missing_keys = np.unique(vertex_keys[~represented])
        rows.append(
            {
                "sha256": sha256,
                "resolution": resolution,
                "status": "ok",
                "error": "",
                "num_referenced_vertices": num_vertices,
                "num_valid_triangles": num_valid_triangles,
                "num_active_support_voxels": int(len(support_coords)),
                "num_represented_vertices": num_represented,
                "num_missing_vertices": num_missing,
                "represented_vertex_fraction": safe_ratio(
                    num_represented, num_vertices
                ),
                "num_unique_vertex_voxels": int(len(unique_vertex_keys)),
                "num_unique_missing_vertex_voxels": int(len(unique_missing_keys)),
                "represented_unique_voxel_fraction": safe_ratio(
                    len(unique_vertex_keys) - len(unique_missing_keys),
                    len(unique_vertex_keys),
                ),
            }
        )

        missing_indices = np.flatnonzero(~represented)[:examples_per_mesh]
        for referenced_index in missing_indices:
            coord = vertex_coords[referenced_index]
            vertex = vertices[referenced_index]
            examples.append(
                {
                    "sha256": sha256,
                    "resolution": resolution,
                    "referenced_vertex_index": int(referenced_index),
                    "global_vertex_index": int(global_indices[referenced_index]),
                    "vertex_x": float(vertex[0]),
                    "vertex_y": float(vertex[1]),
                    "vertex_z": float(vertex[2]),
                    "voxel_x": int(coord[0]),
                    "voxel_y": int(coord[1]),
                    "voxel_z": int(coord[2]),
                }
            )
    return rows, examples


def new_resolution_summary() -> dict[str, int]:
    return {
        "mesh_rows": 0,
        "audited_meshes": 0,
        "missing_support_meshes": 0,
        "mesh_error_rows": 0,
        "support_error_meshes": 0,
        "perfect_meshes": 0,
        "meshes_with_missing_vertices": 0,
        "total_referenced_vertices": 0,
        "represented_vertices": 0,
        "missing_vertices": 0,
        "summed_unique_vertex_voxels": 0,
        "summed_unique_missing_vertex_voxels": 0,
        "max_missing_vertices_in_one_mesh": 0,
    }


def add_row_to_summary(summary: dict[str, int], row: dict[str, object]) -> None:
    summary["mesh_rows"] += 1
    status = str(row["status"])
    if status == "missing_support":
        summary["missing_support_meshes"] += 1
        return
    if status == "mesh_error":
        summary["mesh_error_rows"] += 1
        return
    if status == "support_error":
        summary["support_error_meshes"] += 1
        return
    if status != "ok":
        raise ValueError(f"Unknown row status: {status}")

    summary["audited_meshes"] += 1
    num_vertices = int(row["num_referenced_vertices"])
    represented = int(row["num_represented_vertices"])
    missing = int(row["num_missing_vertices"])
    summary["total_referenced_vertices"] += num_vertices
    summary["represented_vertices"] += represented
    summary["missing_vertices"] += missing
    summary["summed_unique_vertex_voxels"] += int(row["num_unique_vertex_voxels"])
    summary["summed_unique_missing_vertex_voxels"] += int(
        row["num_unique_missing_vertex_voxels"]
    )
    summary["max_missing_vertices_in_one_mesh"] = max(
        summary["max_missing_vertices_in_one_mesh"], missing
    )
    if missing == 0:
        summary["perfect_meshes"] += 1
    else:
        summary["meshes_with_missing_vertices"] += 1


def finish_summary(summary: dict[str, int]) -> dict[str, int | float]:
    result: dict[str, int | float] = dict(summary)
    result["represented_vertex_fraction"] = safe_ratio(
        summary["represented_vertices"], summary["total_referenced_vertices"]
    )
    represented_unique = (
        summary["summed_unique_vertex_voxels"]
        - summary["summed_unique_missing_vertex_voxels"]
    )
    result["represented_unique_voxel_fraction"] = safe_ratio(
        represented_unique, summary["summed_unique_vertex_voxels"]
    )
    return result


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def audit(args: argparse.Namespace) -> None:
    if args.world_size <= 0:
        raise ValueError("--world_size must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world_size")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")
    if args.examples_per_mesh < 0:
        raise ValueError("--examples_per_mesh must be non-negative")
    resolutions = tuple(args.resolutions)
    if not resolutions or any(resolution <= 0 for resolution in resolutions):
        raise ValueError("--resolutions must contain positive integers")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("--resolutions contains duplicates")

    output_dir = args.output_dir or (
        args.root / "outputs" / "triangle_field_vertex_support_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    worklist = load_worklist(
        args.root, resolutions, args.instances, args.limit
    )
    shard = worklist[args.rank :: args.world_size]
    suffix = f"rank_{args.rank:05d}_of_{args.world_size:05d}"
    per_mesh_path = output_dir / f"per_mesh_{suffix}.csv"
    examples_path = output_dir / f"missing_vertex_examples_{suffix}.csv"
    summary_path = output_dir / f"summary_{suffix}.json"

    print(f"Root: {args.root}")
    print(f"Resolutions: {list(resolutions)}")
    print(f"Full worklist: {len(worklist)}")
    print(f"Shard: {args.rank}/{args.world_size} ({len(shard)} meshes)")
    print(f"Output: {output_dir}")

    tasks = [
        (str(args.root), sha256, resolutions, args.examples_per_mesh)
        for sha256 in shard
    ]
    executor = None
    if args.num_workers == 1:
        results = map(check_one, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers
        )
        results = executor.map(check_one, tasks, chunksize=args.chunksize)

    started = time.time()
    summaries = {resolution: new_resolution_summary() for resolution in resolutions}
    per_mesh_tmp = per_mesh_path.with_name(per_mesh_path.name + ".tmp")
    examples_tmp = examples_path.with_name(examples_path.name + ".tmp")
    try:
        with per_mesh_tmp.open("w", newline="") as per_mesh_handle, examples_tmp.open(
            "w", newline=""
        ) as examples_handle:
            per_mesh_writer = csv.DictWriter(
                per_mesh_handle, fieldnames=list(PER_MESH_FIELDS)
            )
            examples_writer = csv.DictWriter(
                examples_handle, fieldnames=list(EXAMPLE_FIELDS)
            )
            per_mesh_writer.writeheader()
            examples_writer.writeheader()
            for rows, examples in tqdm(
                results, total=len(tasks), desc=f"Auditing shard {args.rank}"
            ):
                for row in rows:
                    per_mesh_writer.writerow(row)
                    add_row_to_summary(summaries[int(row["resolution"])], row)
                examples_writer.writerows(examples)
    finally:
        if executor is not None:
            executor.shutdown()

    os.replace(per_mesh_tmp, per_mesh_path)
    os.replace(examples_tmp, examples_path)
    summary = {
        "mode": "triangle_field_vertex_floor_voxel_membership",
        "normalization": "voxelizer-compatible float32",
        "vertex_population": "vertices referenced by finite non-degenerate triangles",
        "root": str(args.root),
        "resolutions": list(resolutions),
        "rank": args.rank,
        "world_size": args.world_size,
        "full_worklist_meshes": len(worklist),
        "shard_meshes": len(shard),
        "examples_per_mesh_per_resolution": args.examples_per_mesh,
        "elapsed_seconds": time.time() - started,
        "per_resolution": {
            str(resolution): finish_summary(summaries[resolution])
            for resolution in resolutions
        },
    }
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2))


def parse_int_row(row: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = dict(row)
    integer_fields = (
        "resolution",
        "num_referenced_vertices",
        "num_valid_triangles",
        "num_active_support_voxels",
        "num_represented_vertices",
        "num_missing_vertices",
        "num_unique_vertex_voxels",
        "num_unique_missing_vertex_voxels",
    )
    for field in integer_fields:
        parsed[field] = int(row[field])
    for field in ("represented_vertex_fraction", "represented_unique_voxel_fraction"):
        parsed[field] = float(row[field]) if row[field] else ""
    return parsed


def finalize(args: argparse.Namespace) -> None:
    if args.world_size <= 0:
        raise ValueError("--world_size must be positive")
    output_dir = args.output_dir
    suffixes = [
        f"rank_{rank:05d}_of_{args.world_size:05d}"
        for rank in range(args.world_size)
    ]
    per_mesh_paths = [output_dir / f"per_mesh_{suffix}.csv" for suffix in suffixes]
    example_paths = [
        output_dir / f"missing_vertex_examples_{suffix}.csv"
        for suffix in suffixes
    ]
    missing = [str(path) for path in per_mesh_paths + example_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot finalize; missing shard output(s):\n" + "\n".join(missing[:20])
        )

    per_mesh_rows: list[dict[str, object]] = []
    for path in per_mesh_paths:
        with path.open(newline="") as handle:
            per_mesh_rows.extend(parse_int_row(row) for row in csv.DictReader(handle))
    per_mesh_rows.sort(key=lambda row: (str(row["sha256"]), int(row["resolution"])))

    seen = set()
    duplicates = []
    for row in per_mesh_rows:
        key = (str(row["sha256"]), int(row["resolution"]))
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate mesh/resolution rows found: {duplicates[:10]}")

    examples: list[dict[str, object]] = []
    for path in example_paths:
        with path.open(newline="") as handle:
            examples.extend(dict(row) for row in csv.DictReader(handle))
    examples.sort(
        key=lambda row: (
            str(row["sha256"]),
            int(row["resolution"]),
            int(row["referenced_vertex_index"]),
        )
    )

    summaries: dict[int, dict[str, int]] = defaultdict(new_resolution_summary)
    for row in per_mesh_rows:
        add_row_to_summary(summaries[int(row["resolution"])], row)

    resolutions = sorted(summaries)
    unique_meshes = sorted({str(row["sha256"]) for row in per_mesh_rows})
    summary = {
        "mode": "triangle_field_vertex_floor_voxel_membership",
        "normalization": "voxelizer-compatible float32",
        "vertex_population": "vertices referenced by finite non-degenerate triangles",
        "world_size": args.world_size,
        "mesh_count_in_merged_rows": len(unique_meshes),
        "resolutions": resolutions,
        "per_resolution": {
            str(resolution): finish_summary(summaries[resolution])
            for resolution in resolutions
        },
    }

    write_csv(output_dir / "per_mesh.csv", PER_MESH_FIELDS, per_mesh_rows)
    write_csv(
        output_dir / "missing_vertex_examples.csv", EXAMPLE_FIELDS, examples
    )
    failures = [row for row in per_mesh_rows if row["status"] != "ok"]
    write_csv(output_dir / "failures.csv", PER_MESH_FIELDS, failures)
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        audit(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
