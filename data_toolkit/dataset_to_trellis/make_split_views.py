#!/usr/bin/env python3
"""Create train/test split views for TRELLIS voxel datasets without duplicating voxels."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


VOXEL_KINDS = {
    "edge_distance": {
        "dirname_prefix": "edge_distance_voxels",
        "flag": "edge_distance_voxelized",
        "count": "num_edge_distance_voxels",
        "data_dir_key": "edge_distance_voxel",
    },
    "vertex_distance": {
        "dirname_prefix": "vertex_distance_voxels",
        "flag": "vertex_distance_voxelized",
        "count": "num_vertex_distance_voxels",
        "data_dir_key": "vertex_distance_voxel",
    },
    "pbr": {
        "dirname_prefix": "pbr_voxels",
        "flag": "pbr_voxelized",
        "count": "num_pbr_voxels",
        "data_dir_key": "pbr_voxel",
    },
}


def truthy_series(series):
    true_values = {"1", "true", "t", "yes", "y"}
    return series.fillna(False).map(lambda value: str(value).strip().lower() in true_values)


def split_count(total: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(total * fraction)))


def write_text_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def place_voxel(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    if not src.exists():
        print(f"Missing voxel file: {src}")
        return False
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return True

    if mode == "none":
        return True
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "absolute_symlink":
        os.symlink(src, dst)
    elif mode == "relative_symlink":
        os.symlink(os.path.relpath(src, dst.parent), dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create split-specific metadata and voxel roots for TRELLIS training."
    )
    parser.add_argument("--root", type=Path, required=True, help="Processed dataset root.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Merged metadata CSV. Defaults to <root>/metadata.csv.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Output split directory. Defaults to <root>/splits.",
    )
    parser.add_argument("--resolution", type=int, default=256, help="Voxel resolution to split.")
    parser.add_argument(
        "--voxel-kind",
        choices=sorted(VOXEL_KINDS),
        default="edge_distance",
        help="Voxel directory/metadata schema to split.",
    )
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction assigned to test.")
    parser.add_argument("--val-frac", type=float, default=0.0, help="Optional validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    parser.add_argument(
        "--train-name",
        default="train",
        help="Name of the train split directory.",
    )
    parser.add_argument("--test-name", default="test", help="Name of the test split directory.")
    parser.add_argument("--val-name", default="val", help="Name of the validation split directory.")
    parser.add_argument(
        "--allow-unvoxelized",
        action="store_true",
        help="Do not filter to rows where the voxelized flag is true.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["relative_symlink", "absolute_symlink", "copy", "none"],
        default="relative_symlink",
        help="How split voxel files point to the canonical voxel directory.",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="Replace existing symlinks/files inside split voxel directories.",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Replace existing split metadata.csv files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd

    root = args.root.resolve()
    metadata_path = args.metadata or root / "metadata.csv"
    split_root = args.split_root or root / "splits"
    schema = VOXEL_KINDS[args.voxel_kind]
    voxel_dirname = f"{schema['dirname_prefix']}_{args.resolution}"
    canonical_voxel_root = root / voxel_dirname

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if args.link_mode != "none" and not canonical_voxel_root.exists():
        raise FileNotFoundError(f"Voxel directory not found: {canonical_voxel_root}")

    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain a sha256 column.")
    metadata["sha256"] = metadata["sha256"].astype(str)

    flag_col = schema["flag"]
    count_col = schema["count"]
    if not args.allow_unvoxelized:
        if flag_col not in metadata.columns:
            raise ValueError(f"{metadata_path} is missing required column: {flag_col}")
        metadata = metadata[truthy_series(metadata[flag_col])].copy()

    metadata = metadata.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No eligible rows remain after filtering.")

    n_total = len(metadata)
    n_test = split_count(n_total, args.test_frac)
    n_val = split_count(n_total, args.val_frac)
    if n_test + n_val >= n_total:
        raise ValueError("test-frac + val-frac leaves no training data.")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    split_indices = {
        args.test_name: indices[:n_test],
        args.train_name: indices[n_test + n_val :],
    }
    if n_val:
        split_indices[args.val_name] = indices[n_test : n_test + n_val]

    split_root.mkdir(parents=True, exist_ok=True)
    for split_name, idx in split_indices.items():
        split_dir = split_root / split_name
        split_voxel_root = split_dir / voxel_dirname
        split_meta_path = split_dir / "metadata.csv"
        voxel_meta_path = split_voxel_root / "metadata.csv"

        if split_meta_path.exists() and not args.overwrite_metadata:
            raise FileExistsError(f"{split_meta_path} already exists. Use --overwrite-metadata.")
        if voxel_meta_path.exists() and not args.overwrite_metadata:
            raise FileExistsError(f"{voxel_meta_path} already exists. Use --overwrite-metadata.")

        split_dir.mkdir(parents=True, exist_ok=True)
        split_voxel_root.mkdir(parents=True, exist_ok=True)

        split_meta = metadata.iloc[idx].sort_values("sha256").reset_index(drop=True)
        split_meta.to_csv(split_meta_path, index=False)
        write_text_lines(split_dir / "instances.txt", split_meta["sha256"].tolist())

        voxel_cols = ["sha256"]
        if flag_col in split_meta.columns:
            voxel_cols.append(flag_col)
        if count_col in split_meta.columns:
            voxel_cols.append(count_col)
        split_meta[voxel_cols].to_csv(voxel_meta_path, index=False)

        linked = 0
        for sha256 in split_meta["sha256"]:
            src = canonical_voxel_root / f"{sha256}.vxz"
            dst = split_voxel_root / f"{sha256}.vxz"
            linked += int(place_voxel(src, dst, args.link_mode, args.overwrite_links))

        print(
            f"{split_name}: {len(split_meta)} rows, {linked} voxel references, "
            f"data_dir key '{schema['data_dir_key']}' -> {split_voxel_root}"
        )


if __name__ == "__main__":
    main()
