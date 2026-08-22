#!/usr/bin/env python3
"""Validate a finalized QEM edge-collapsed mesh dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qem_root", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Defaults to <qem_root>/metadata.csv.",
    )
    parser.add_argument("--triangle_field_voxel_root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument(
        "--invalid_instances",
        type=Path,
        default=None,
        help="Write every bad or missing SHA256, one per line.",
    )
    parser.add_argument(
        "--failures_csv",
        type=Path,
        default=None,
        help="Write every validation failure with its status and error.",
    )
    return parser.parse_args()


def load_payload(path: Path):
    import zstandard as zstd

    payload = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return np.load(io.BytesIO(payload), allow_pickle=False)


def find_triangle_field_path(root: Path, sha256: str) -> Path:
    for suffix in (".npz.zst", ".npz"):
        path = root / f"{sha256}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing triangle-field support for {sha256}")


def load_triangle_field_coords(root: Path, sha256: str) -> np.ndarray:
    path = find_triangle_field_path(root, sha256)
    if path.name.endswith(".npz.zst"):
        with load_payload(path) as data:
            return np.asarray(data["coords"], dtype=np.int32)
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["coords"], dtype=np.int32)


def coords_to_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def check_one(
    root: str,
    support_root: str,
    sha256: str,
    resolution: int,
) -> dict[str, object]:
    path = Path(root) / f"{sha256}.npz.zst"
    if not path.is_file():
        return {"status": "missing", "sha256": sha256}
    try:
        with load_payload(path) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        required = {
            "vertices",
            "faces",
            "edges",
            "vertex_normals",
            "face_normals",
            "face_areas",
            "vertex_voxel_coords",
            "vertex_local_offsets",
            "metadata_json",
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        vertices = arrays["vertices"]
        faces = arrays["faces"]
        edges = arrays["edges"]
        vertex_normals = arrays["vertex_normals"]
        face_normals = arrays["face_normals"]
        face_areas = arrays["face_areas"]
        voxel_coords = arrays["vertex_voxel_coords"]
        local_offsets = arrays["vertex_local_offsets"]
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"invalid vertices {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f"invalid faces {faces.shape}")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"invalid edges {edges.shape}")
        if vertex_normals.shape != vertices.shape:
            raise ValueError(f"invalid vertex_normals {vertex_normals.shape}")
        if face_normals.shape != faces.shape:
            raise ValueError(f"invalid face_normals {face_normals.shape}")
        if face_areas.shape != (len(faces),):
            raise ValueError(f"invalid face_areas {face_areas.shape}")
        if voxel_coords.shape != vertices.shape or local_offsets.shape != vertices.shape:
            raise ValueError("vertex voxel/local-offset shapes do not match vertices")
        for name in ("vertices", "vertex_normals", "face_normals", "face_areas", "vertex_local_offsets"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} contains non-finite values")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("face index out of range")
        if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
            raise ValueError("edge index out of range")
        if (
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ).any():
            raise ValueError("degenerate face indices")
        if (voxel_coords < 0).any() or (voxel_coords >= resolution).any():
            raise ValueError("vertex_voxel_coords outside resolution")
        expected_offsets = (vertices.astype(np.float64) + 0.5) * resolution - voxel_coords
        if not np.allclose(expected_offsets, local_offsets, atol=2e-4):
            raise ValueError("vertex_local_offsets disagree with vertices/voxel coords")
        tolerance = 2e-5
        if (local_offsets < -tolerance).any() or (local_offsets > 1.0 + tolerance).any():
            raise ValueError("a collapsed vertex lies outside its assigned voxel")
        recovered_voxels = np.floor(
            (vertices.astype(np.float64) + 0.5) * resolution
        ).astype(np.int32)
        if not np.array_equal(recovered_voxels, voxel_coords):
            mismatch = int(np.any(recovered_voxels != voxel_coords, axis=1).sum())
            raise ValueError(
                f"{mismatch} physical vertex voxel(s) disagree with saved assignments"
            )
        support_coords = load_triangle_field_coords(Path(support_root), sha256)
        support_keys = set(int(key) for key in coords_to_keys(support_coords, resolution))
        output_keys = coords_to_keys(voxel_coords, resolution)
        outside_support = sum(int(key) not in support_keys for key in output_keys)
        if outside_support:
            raise ValueError(
                f"{outside_support} collapsed vertex voxel(s) are outside sparse support"
            )
        _, vertices_per_voxel = np.unique(
            voxel_coords, axis=0, return_counts=True
        )
        max_vertices_per_voxel = int(vertices_per_voxel.max(initial=0))
        if max_vertices_per_voxel > 1:
            raise ValueError(
                f"maximum vertices per voxel is {max_vertices_per_voxel}, expected 1"
            )
        if (face_areas <= 0).any():
            raise ValueError("non-positive face areas")
        if not np.allclose(np.linalg.norm(face_normals, axis=1), 1, atol=2e-4):
            raise ValueError("face normals are not unit length")
        return {
            "status": "ok",
            "sha256": sha256,
            "vertices": len(vertices),
            "faces": len(faces),
            "edges": len(edges),
            "max_vertices_per_voxel": max_vertices_per_voxel,
            "vertices_outside_sparse_support": 0,
            "physical_saved_voxel_mismatches": 0,
        }
    except Exception as exc:
        return {"status": "bad", "sha256": sha256, "error": repr(exc)}


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata or args.qem_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    sha256s = metadata["sha256"].astype(str).tolist()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        results = list(
            tqdm(
                executor.map(
                    check_one,
                    [str(args.qem_root)] * len(sha256s),
                    [str(args.triangle_field_voxel_root)] * len(sha256s),
                    sha256s,
                    [args.resolution] * len(sha256s),
                    chunksize=16,
                ),
                total=len(sha256s),
                desc="Checking QEM meshes",
            )
        )
    ok = [item for item in results if item["status"] == "ok"]
    bad = [item for item in results if item["status"] == "bad"]
    missing = [item for item in results if item["status"] == "missing"]
    failures = bad + missing
    invalid_instances_path = (
        args.invalid_instances or args.qem_root / "invalid_instances.txt"
    )
    failures_csv_path = (
        args.failures_csv or args.qem_root / "validation_failures.csv"
    )
    invalid_instances_path.parent.mkdir(parents=True, exist_ok=True)
    failures_csv_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_instances_path.write_text(
        "".join(f'{item["sha256"]}\n' for item in sorted(failures, key=lambda x: x["sha256"]))
    )
    pd.DataFrame(
        [
            {
                "sha256": item["sha256"],
                "status": item["status"],
                "error": item.get("error", ""),
            }
            for item in failures
        ],
        columns=["sha256", "status", "error"],
    ).to_csv(failures_csv_path, index=False)
    summary = {
        "resolution": args.resolution,
        "metadata_rows": len(sha256s),
        "valid": len(ok),
        "bad": len(bad),
        "missing": len(missing),
        "max_vertices_per_voxel": max(
            (item["max_vertices_per_voxel"] for item in ok), default=0
        ),
        "vertices_outside_sparse_support": 0,
        "physical_saved_voxel_mismatches": 0,
        "invalid_instances": str(invalid_instances_path),
        "failures_csv": str(failures_csv_path),
        "vertices": {
            "min": min((item["vertices"] for item in ok), default=0),
            "max": max((item["vertices"] for item in ok), default=0),
            "mean": float(np.mean([item["vertices"] for item in ok])) if ok else 0,
        },
        "faces": {
            "min": min((item["faces"] for item in ok), default=0),
            "max": max((item["faces"] for item in ok), default=0),
            "mean": float(np.mean([item["faces"] for item in ok])) if ok else 0,
        },
        "bad_examples": bad[:100],
        "missing_examples": missing[:100],
    }
    output_path = args.output_json or args.qem_root / "validation_stats.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if bad or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
