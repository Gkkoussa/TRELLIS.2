#!/usr/bin/env python3
"""Score meshes by tiny-triangle area imbalance and write TRELLIS filter metadata.

This is a metadata-only filter.  It does not create split-local symlink trees.
Use the filtered CSV with the dataset loader's `_metadata_filter_csv` key.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh


SCORE_COLUMNS = [
    "triangle_area_analyzed",
    "triangle_area_error",
    "triangle_area_mesh_path",
    "triangle_area_num_vertices",
    "triangle_area_num_faces",
    "triangle_area_total",
    "triangle_area_A0",
    "triangle_area_min",
    "triangle_area_p01",
    "triangle_area_p05",
    "triangle_area_p50",
    "triangle_area_p95",
    "triangle_area_max",
    "triangle_area_score",
    "triangle_area_reject",
    "triangle_area_filter_keep",
    # Compatibility columns consumed by StandardDatasetBase metadata filters.
    "has_local_dense_region",
    "local_density_filter_keep",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect meshes with locally tiny triangles using an area-weighted "
            "triangle-area imbalance score."
        )
    )
    parser.add_argument("--root", type=Path, default=None, help="Processed dataset root.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata CSV to score. Defaults to <root>/metadata.csv.",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root used to resolve relative mesh paths. Can be passed multiple times. "
            "Defaults to <root>/meshes and <root>."
        ),
    )
    parser.add_argument(
        "--path-columns",
        default="local_path,file_identifier",
        help="Comma-separated metadata columns to try as mesh paths.",
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        default=None,
        help="Scored metadata output. Defaults to <root>/metadata_triangle_area_scores.csv.",
    )
    parser.add_argument(
        "--filtered-metadata-output",
        type=Path,
        default=None,
        help="Optional CSV containing only rows kept by the triangle-area filter.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Reject meshes with triangle_area_score greater than this threshold.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-30,
        help="Minimum triangle area denominator used in the score.",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=1,
        help="Meshes with fewer faces are marked analyzed but kept.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel mesh scoring workers. Use 1 for easiest debugging.",
    )
    parser.add_argument(
        "--drop-score-errors",
        action="store_true",
        help="Drop meshes that could not be loaded/scored. Defaults to keeping them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output metadata files.",
    )
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def resolve_mesh_path(row: dict[str, Any], path_columns: list[str], mesh_roots: list[str]) -> str | None:
    for column in path_columns:
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        raw = str(value).strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_absolute() and path.exists():
            return str(path)
        for root in mesh_roots:
            candidate = Path(root) / raw
            if candidate.exists():
                return str(candidate)
    return None


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("empty scene")
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError("loaded object is not a triangle mesh")
    return mesh


def compute_triangle_area_score(
    mesh: trimesh.Trimesh,
    *,
    threshold: float,
    eps: float,
    min_faces: int,
) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    num_vertices = int(vertices.shape[0]) if vertices.ndim == 2 else 0
    num_faces = int(faces.shape[0]) if faces.ndim == 2 else 0

    if num_faces < min_faces:
        return {
            "triangle_area_analyzed": True,
            "triangle_area_error": "",
            "triangle_area_num_vertices": num_vertices,
            "triangle_area_num_faces": num_faces,
            "triangle_area_total": 0.0,
            "triangle_area_A0": np.nan,
            "triangle_area_min": np.nan,
            "triangle_area_p01": np.nan,
            "triangle_area_p05": np.nan,
            "triangle_area_p50": np.nan,
            "triangle_area_p95": np.nan,
            "triangle_area_max": np.nan,
            "triangle_area_score": 0.0,
            "triangle_area_reject": False,
        }

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    areas = areas[np.isfinite(areas) & (areas >= 0)]
    if areas.size == 0:
        raise ValueError("mesh has no finite triangle areas")

    area_total = float(areas.sum())
    if not np.isfinite(area_total) or area_total <= 0:
        raise ValueError("mesh total triangle area is zero or invalid")

    a0 = float(np.sum(np.square(areas)) / area_total)
    ratios = a0 / np.maximum(areas, eps)
    penalties = np.maximum(0.0, ratios - 1.0)
    score = float(np.sum(areas * np.square(penalties)) / area_total)
    reject = bool(score > threshold)
    p01, p05, p50, p95 = np.percentile(areas, [1, 5, 50, 95])

    return {
        "triangle_area_analyzed": True,
        "triangle_area_error": "",
        "triangle_area_num_vertices": num_vertices,
        "triangle_area_num_faces": num_faces,
        "triangle_area_total": area_total,
        "triangle_area_A0": a0,
        "triangle_area_min": float(areas.min()),
        "triangle_area_p01": float(p01),
        "triangle_area_p05": float(p05),
        "triangle_area_p50": float(p50),
        "triangle_area_p95": float(p95),
        "triangle_area_max": float(areas.max()),
        "triangle_area_score": score,
        "triangle_area_reject": reject,
    }


def score_one(payload: tuple[dict[str, Any], list[str], list[str], argparse.Namespace]) -> dict[str, Any]:
    row, path_columns, mesh_roots, args = payload
    sha256 = str(row["sha256"])
    result: dict[str, Any] = {"sha256": sha256}
    mesh_path = resolve_mesh_path(row, path_columns, mesh_roots)
    result["triangle_area_mesh_path"] = mesh_path or ""

    if mesh_path is None:
        result.update(
            {
                "triangle_area_analyzed": False,
                "triangle_area_error": "mesh path not found",
                "triangle_area_reject": False,
            }
        )
        return result

    try:
        mesh = load_mesh(mesh_path)
        result.update(
            compute_triangle_area_score(
                mesh,
                threshold=args.threshold,
                eps=args.eps,
                min_faces=args.min_faces,
            )
        )
    except Exception as exc:
        result.update(
            {
                "triangle_area_analyzed": False,
                "triangle_area_error": str(exc),
                "triangle_area_reject": False,
            }
        )
    return result


def add_filter_columns(scores: pd.DataFrame, drop_score_errors: bool) -> pd.DataFrame:
    scores = scores.copy()
    analyzed = scores["triangle_area_analyzed"].map(truthy)
    reject = scores["triangle_area_reject"].map(truthy)
    keep = (~reject) & (analyzed | (not drop_score_errors))
    scores["triangle_area_filter_keep"] = keep
    scores["has_local_dense_region"] = reject
    scores["local_density_filter_keep"] = keep
    return scores


def merge_scores(metadata: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["sha256"] = metadata["sha256"].astype(str)
    scores = scores.copy()
    scores["sha256"] = scores["sha256"].astype(str)
    metadata = metadata.drop(columns=[c for c in SCORE_COLUMNS if c in metadata.columns], errors="ignore")
    merge_cols = ["sha256"] + [c for c in SCORE_COLUMNS if c in scores.columns]
    return metadata.merge(scores[merge_cols], on="sha256", how="left")


def kept_mask(scored_metadata: pd.DataFrame) -> pd.Series:
    return scored_metadata["triangle_area_filter_keep"].map(
        lambda value: truthy(value) if pd.notna(value) else False
    )


def print_quantiles(scores: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in scores:
            continue
        values = pd.to_numeric(scores[column], errors="coerce").dropna()
        if len(values) == 0:
            continue
        q = values.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
        print(
            f"{column} quantiles: "
            f"p50={q[0.5]:.6g}, p90={q[0.9]:.6g}, "
            f"p95={q[0.95]:.6g}, p99={q[0.99]:.6g}"
        )


def main() -> None:
    args = parse_args()
    if args.root is None and args.metadata is None:
        raise ValueError("Provide --root or --metadata.")
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative.")
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")

    root = args.root.resolve() if args.root is not None else None
    metadata_path = (args.metadata or root / "metadata.csv").resolve()
    if args.score_output is None:
        if root is None:
            raise ValueError("--score-output is required when --root is not provided.")
        args.score_output = root / "metadata_triangle_area_scores.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    mesh_roots = args.mesh_root
    if mesh_roots is None:
        if root is None:
            mesh_roots = [metadata_path.parent]
        else:
            mesh_roots = [root / "meshes", root]
    mesh_roots_str = [str(path.resolve()) for path in mesh_roots]
    path_columns = [column.strip() for column in args.path_columns.split(",") if column.strip()]

    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain a sha256 column.")
    metadata["sha256"] = metadata["sha256"].astype(str)
    metadata = metadata.drop_duplicates("sha256", keep="first").reset_index(drop=True)

    rows = metadata.to_dict("records")
    payloads = [(row, path_columns, mesh_roots_str, args) for row in rows]
    print(f"Scoring {len(payloads)} meshes from {metadata_path}")
    print(f"Mesh roots: {mesh_roots_str}")
    print(f"Rejecting meshes with triangle_area_score > {args.threshold}")

    if args.num_workers == 1:
        score_rows = []
        for idx, payload in enumerate(payloads, 1):
            score_rows.append(score_one(payload))
            if idx % 100 == 0 or idx == len(payloads):
                print(f"  scored {idx}/{len(payloads)}")
    else:
        score_rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(score_one, payload) for payload in payloads]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                score_rows.append(future.result())
                if idx % 100 == 0 or idx == len(futures):
                    print(f"  scored {idx}/{len(futures)}")

    scores = pd.DataFrame(score_rows)
    scores = add_filter_columns(scores, args.drop_score_errors)
    scored_metadata = merge_scores(metadata, scores)

    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    if args.score_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.score_output} exists. Use --overwrite.")
    scored_metadata.to_csv(args.score_output, index=False)

    analyzed = int(scores["triangle_area_analyzed"].map(truthy).sum())
    rejected = int(scores["triangle_area_reject"].map(truthy).sum())
    kept = int(scores["triangle_area_filter_keep"].map(truthy).sum())
    print(f"Wrote scored metadata: {args.score_output}")
    print(f"Analyzed: {analyzed} / {len(scores)}")
    print(f"Rejected by triangle-area filter: {rejected}")
    print(f"Kept by filter: {kept}")
    print_quantiles(
        scores,
        [
            "triangle_area_score",
            "triangle_area_A0",
            "triangle_area_min",
            "triangle_area_p01",
            "triangle_area_p50",
        ],
    )

    if args.filtered_metadata_output is not None:
        filtered = scored_metadata[kept_mask(scored_metadata)].copy()
        args.filtered_metadata_output.parent.mkdir(parents=True, exist_ok=True)
        if args.filtered_metadata_output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.filtered_metadata_output} exists. Use --overwrite.")
        filtered.to_csv(args.filtered_metadata_output, index=False)
        print(f"Wrote filtered metadata: {args.filtered_metadata_output}")


if __name__ == "__main__":
    main()
