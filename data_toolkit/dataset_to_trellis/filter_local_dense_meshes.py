#!/usr/bin/env python3
"""Score meshes for local over-density and build filtered TRELLIS split views.

The filter is intentionally metadata-driven: it does not modify the training
datasets.  It writes scored metadata and, optionally, replacement split views
whose metadata files only contain meshes that pass the local-density test.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


SCORE_COLUMNS = [
    "local_density_analyzed",
    "local_density_error",
    "local_density_mesh_path",
    "local_density_knn_k",
    "local_density_num_vertices",
    "local_density_num_faces",
    "local_density_bbox_diag",
    "local_density_spacing_p01",
    "local_density_spacing_p05",
    "local_density_spacing_p50",
    "local_density_ratio_p01",
    "local_density_ratio_p05",
    "local_density_dense_fraction",
    "local_density_score",
    "has_local_dense_region",
    "local_density_filter_keep",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect meshes with locally dense vertex regions, annotate metadata, "
            "and optionally create filtered TRELLIS split views."
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
        help="Scored metadata output. Defaults to <root>/metadata_local_density_scores.csv.",
    )
    parser.add_argument(
        "--filtered-metadata-output",
        type=Path,
        default=None,
        help="Optional CSV containing only rows kept by the density filter.",
    )
    parser.add_argument("--k-neighbors", type=int, default=16, help="k for kNN local spacing.")
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=200_000,
        help="Deterministically subsample larger meshes to this many vertices.",
    )
    parser.add_argument(
        "--min-vertices",
        type=int,
        default=128,
        help="Meshes with fewer vertices are marked analyzed but not dense.",
    )
    parser.add_argument(
        "--dense-ratio-threshold",
        type=float,
        default=8.0,
        help="A vertex is locally dense if its kNN radius is at least this many times smaller than the median.",
    )
    parser.add_argument(
        "--min-dense-fraction",
        type=float,
        default=0.005,
        help="Minimum fraction of vertices that must be locally dense to filter the mesh.",
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
        "--input-split-root",
        type=Path,
        default=None,
        help="Existing split root to filter. Defaults to <root>/splits when --output-split-root is set.",
    )
    parser.add_argument(
        "--output-split-root",
        type=Path,
        default=None,
        help="Write replacement split views here, filtering every metadata.csv under each split.",
    )
    parser.add_argument(
        "--split-names",
        default=None,
        help="Comma-separated split names to filter. Defaults to all directories under input split root.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["relative_symlink", "absolute_symlink", "copy", "none"],
        default="relative_symlink",
        help="How kept split files point back to the original split files.",
    )
    parser.add_argument(
        "--drop-unscored-in-splits",
        action="store_true",
        help="Drop split rows with no matching density score. Defaults to keeping them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output metadata files and existing links.",
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


def load_mesh_arrays(mesh_path: str) -> tuple[np.ndarray, int]:
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
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh has no 3D vertices")
    faces = getattr(mesh, "faces", None)
    num_faces = 0 if faces is None else int(len(faces))
    finite = np.isfinite(vertices).all(axis=1)
    vertices = vertices[finite]
    return vertices, num_faces


def score_vertices(
    vertices: np.ndarray,
    *,
    num_faces: int,
    k_neighbors: int,
    max_vertices: int,
    min_vertices: int,
    dense_ratio_threshold: float,
    min_dense_fraction: float,
) -> dict[str, Any]:
    num_vertices_original = int(vertices.shape[0])
    if num_vertices_original == 0:
        raise ValueError("mesh has no finite vertices")

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if not np.isfinite(bbox_diag) or bbox_diag <= 0:
        raise ValueError("degenerate mesh bounds")

    if num_vertices_original > max_vertices:
        rng = np.random.default_rng(0)
        indices = rng.choice(num_vertices_original, size=max_vertices, replace=False)
        vertices = vertices[indices]

    num_vertices = int(vertices.shape[0])
    if num_vertices < min_vertices:
        return {
            "local_density_analyzed": True,
            "local_density_error": "",
            "local_density_knn_k": min(k_neighbors, max(num_vertices - 1, 1)),
            "local_density_num_vertices": num_vertices_original,
            "local_density_num_faces": num_faces,
            "local_density_bbox_diag": bbox_diag,
            "local_density_spacing_p01": np.nan,
            "local_density_spacing_p05": np.nan,
            "local_density_spacing_p50": np.nan,
            "local_density_ratio_p01": 1.0,
            "local_density_ratio_p05": 1.0,
            "local_density_dense_fraction": 0.0,
            "local_density_score": 0.0,
            "has_local_dense_region": False,
        }

    normalized = (vertices - bbox_min) / bbox_diag
    unique_vertices = np.unique(normalized, axis=0)
    if unique_vertices.shape[0] <= k_neighbors:
        raise ValueError("not enough unique vertices for kNN density")

    k = min(k_neighbors, unique_vertices.shape[0] - 1)
    tree = cKDTree(unique_vertices)
    distances, _ = tree.query(unique_vertices, k=k + 1, workers=-1)
    kth_radius = distances[:, k]
    kth_radius = kth_radius[np.isfinite(kth_radius) & (kth_radius > 0)]
    if kth_radius.size == 0:
        raise ValueError("all kNN radii are zero or invalid")

    p01, p05, p50 = np.percentile(kth_radius, [1, 5, 50])
    eps = np.finfo(np.float64).eps
    ratio_p01 = float(p50 / max(p01, eps))
    ratio_p05 = float(p50 / max(p05, eps))
    dense_cutoff = float(p50 / dense_ratio_threshold)
    dense_fraction = float(np.mean(kth_radius <= dense_cutoff))
    score = float(ratio_p01 * min(1.0, dense_fraction / max(min_dense_fraction, eps)))
    has_dense_region = bool(ratio_p01 >= dense_ratio_threshold and dense_fraction >= min_dense_fraction)

    return {
        "local_density_analyzed": True,
        "local_density_error": "",
        "local_density_knn_k": k,
        "local_density_num_vertices": num_vertices_original,
        "local_density_num_faces": num_faces,
        "local_density_bbox_diag": bbox_diag,
        "local_density_spacing_p01": float(p01),
        "local_density_spacing_p05": float(p05),
        "local_density_spacing_p50": float(p50),
        "local_density_ratio_p01": ratio_p01,
        "local_density_ratio_p05": ratio_p05,
        "local_density_dense_fraction": dense_fraction,
        "local_density_score": score,
        "has_local_dense_region": has_dense_region,
    }


def score_one(payload: tuple[dict[str, Any], list[str], list[str], argparse.Namespace]) -> dict[str, Any]:
    row, path_columns, mesh_roots, args = payload
    sha256 = str(row["sha256"])
    result: dict[str, Any] = {"sha256": sha256}
    mesh_path = resolve_mesh_path(row, path_columns, mesh_roots)
    result["local_density_mesh_path"] = mesh_path or ""
    if mesh_path is None:
        result.update(
            {
                "local_density_analyzed": False,
                "local_density_error": "mesh path not found",
                "has_local_dense_region": False,
            }
        )
        return result

    try:
        vertices, num_faces = load_mesh_arrays(mesh_path)
        result.update(
            score_vertices(
                vertices,
                num_faces=num_faces,
                k_neighbors=args.k_neighbors,
                max_vertices=args.max_vertices,
                min_vertices=args.min_vertices,
                dense_ratio_threshold=args.dense_ratio_threshold,
                min_dense_fraction=args.min_dense_fraction,
            )
        )
    except Exception as exc:
        result.update(
            {
                "local_density_analyzed": False,
                "local_density_error": str(exc),
                "has_local_dense_region": False,
            }
        )
    return result


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    if not src.exists() and not src.is_symlink():
        return False
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return True
    if mode == "none":
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "absolute_symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "relative_symlink":
        os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")
    return True


def merge_scores(metadata: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["sha256"] = metadata["sha256"].astype(str)
    scores = scores.copy()
    scores["sha256"] = scores["sha256"].astype(str)
    metadata = metadata.drop(columns=[c for c in SCORE_COLUMNS if c in metadata.columns], errors="ignore")
    return metadata.merge(scores[["sha256"] + [c for c in SCORE_COLUMNS if c in scores.columns]], on="sha256", how="left")


def add_filter_keep(scores: pd.DataFrame, drop_score_errors: bool) -> pd.DataFrame:
    scores = scores.copy()
    analyzed = scores["local_density_analyzed"].map(truthy)
    dense = scores["has_local_dense_region"].map(truthy)
    scores["local_density_filter_keep"] = (~dense) & (analyzed | (not drop_score_errors))
    return scores


def kept_mask(scored_metadata: pd.DataFrame, drop_unscored: bool) -> pd.Series:
    keep = scored_metadata["local_density_filter_keep"]
    keep_bool = keep.map(lambda value: truthy(value) if pd.notna(value) else not drop_unscored)
    return keep_bool.astype(bool)


def write_instances(path: Path, sha256s: list[str]) -> None:
    path.write_text("\n".join(sha256s) + ("\n" if sha256s else ""))


def link_kept_stage_files(stage_src_dir: Path, stage_dst_dir: Path, sha256s: list[str], args: argparse.Namespace) -> int:
    linked = 0
    skip_names = {"metadata.csv", "instances.txt", "statistics.txt", "normalization.json"}
    for sha256 in sha256s:
        for src in stage_src_dir.glob(f"{sha256}*"):
            if src.name in skip_names or src.is_dir():
                continue
            dst = stage_dst_dir / src.name
            linked += int(place_file(src, dst, args.link_mode, args.overwrite))
    return linked


def filter_split_views(args: argparse.Namespace, scores: pd.DataFrame) -> None:
    if args.output_split_root is None:
        return
    if args.root is None and args.input_split_root is None:
        raise ValueError("--root or --input-split-root is required with --output-split-root")

    input_split_root = (args.input_split_root or args.root / "splits").resolve()
    output_split_root = args.output_split_root.resolve()
    if not input_split_root.exists():
        raise FileNotFoundError(f"Input split root not found: {input_split_root}")

    if args.split_names:
        split_dirs = [input_split_root / name.strip() for name in args.split_names.split(",") if name.strip()]
    else:
        split_dirs = sorted(path for path in input_split_root.iterdir() if path.is_dir())
    if not split_dirs:
        raise ValueError(f"No split directories found under {input_split_root}")

    output_split_root.mkdir(parents=True, exist_ok=True)
    for split_src in split_dirs:
        split_meta_path = split_src / "metadata.csv"
        if not split_meta_path.exists():
            print(f"Skipping {split_src}: no metadata.csv")
            continue

        split_dst = output_split_root / split_src.name
        split_dst.mkdir(parents=True, exist_ok=True)
        split_meta_out = split_dst / "metadata.csv"
        if split_meta_out.exists() and not args.overwrite:
            raise FileExistsError(f"{split_meta_out} exists. Use --overwrite.")

        split_meta = pd.read_csv(split_meta_path)
        scored_split_meta = merge_scores(split_meta, scores)
        filtered_split_meta = scored_split_meta[kept_mask(scored_split_meta, args.drop_unscored_in_splits)].copy()
        filtered_split_meta = filtered_split_meta.drop_duplicates("sha256", keep="first").sort_values("sha256")
        split_sha = filtered_split_meta["sha256"].astype(str).tolist()
        split_sha_set = set(split_sha)

        filtered_split_meta.to_csv(split_meta_out, index=False)
        write_instances(split_dst / "instances.txt", split_sha)

        print(
            f"{split_src.name}: kept {len(filtered_split_meta)} / {len(split_meta)} rows "
            f"after local-density filtering"
        )

        stage_meta_paths = sorted(
            path for path in split_src.rglob("metadata.csv")
            if path != split_meta_path
        )
        for stage_meta_path in stage_meta_paths:
            rel_meta = stage_meta_path.relative_to(split_src)
            stage_src_dir = stage_meta_path.parent
            stage_dst_dir = split_dst / rel_meta.parent
            stage_dst_dir.mkdir(parents=True, exist_ok=True)
            stage_meta_out = stage_dst_dir / "metadata.csv"
            if stage_meta_out.exists() and not args.overwrite:
                raise FileExistsError(f"{stage_meta_out} exists. Use --overwrite.")

            stage_meta = pd.read_csv(stage_meta_path)
            stage_meta["sha256"] = stage_meta["sha256"].astype(str)
            stage_meta = stage_meta[stage_meta["sha256"].isin(split_sha_set)]
            stage_meta = merge_scores(stage_meta, scores)
            stage_meta = stage_meta.drop_duplicates("sha256", keep="first").sort_values("sha256")
            stage_sha = stage_meta["sha256"].astype(str).tolist()
            stage_meta.to_csv(stage_meta_out, index=False)
            if (stage_src_dir / "instances.txt").exists():
                write_instances(stage_dst_dir / "instances.txt", stage_sha)
            linked = link_kept_stage_files(stage_src_dir, stage_dst_dir, stage_sha, args)
            print(f"  {rel_meta.parent}: {len(stage_meta)} rows, {linked} linked/copied files")

            if (stage_src_dir / "normalization.json").exists():
                print(
                    f"  {rel_meta.parent}: skipped normalization.json; recompute it from the filtered train split"
                )


def main() -> None:
    args = parse_args()
    if args.root is None and args.metadata is None:
        raise ValueError("Provide --root or --metadata.")
    root = args.root.resolve() if args.root is not None else None
    metadata_path = (args.metadata or root / "metadata.csv").resolve()
    if args.score_output is None:
        if root is None:
            raise ValueError("--score-output is required when --root is not provided.")
        args.score_output = root / "metadata_local_density_scores.csv"

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
    scores = add_filter_keep(scores, args.drop_score_errors)
    scored_metadata = merge_scores(metadata, scores)
    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    if args.score_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.score_output} exists. Use --overwrite.")
    scored_metadata.to_csv(args.score_output, index=False)

    analyzed = int(scores["local_density_analyzed"].map(truthy).sum())
    dense = int(scores["has_local_dense_region"].map(truthy).sum())
    kept = int(scores["local_density_filter_keep"].map(truthy).sum())
    print(f"Wrote scored metadata: {args.score_output}")
    print(f"Analyzed: {analyzed} / {len(scores)}")
    print(f"Local dense region: {dense}")
    print(f"Kept by filter: {kept}")
    for column in ["local_density_ratio_p01", "local_density_dense_fraction", "local_density_score"]:
        values = pd.to_numeric(scores[column], errors="coerce").dropna()
        if len(values):
            q = values.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
            print(
                f"{column} quantiles: "
                f"p50={q[0.5]:.4g}, p90={q[0.9]:.4g}, p95={q[0.95]:.4g}, p99={q[0.99]:.4g}"
            )

    if args.filtered_metadata_output is not None:
        filtered = scored_metadata[kept_mask(scored_metadata, drop_unscored=False)].copy()
        args.filtered_metadata_output.parent.mkdir(parents=True, exist_ok=True)
        if args.filtered_metadata_output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.filtered_metadata_output} exists. Use --overwrite.")
        filtered.to_csv(args.filtered_metadata_output, index=False)
        print(f"Wrote filtered metadata: {args.filtered_metadata_output}")

    filter_split_views(args, scores)


if __name__ == "__main__":
    main()
