#!/usr/bin/env python3
"""Create split-specific latent views for existing TRELLIS splits without changing split membership."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


LATENT_KINDS = {
    "shape_latent": {
        "dirname_prefix": "shape_latents",
        "flag": "shape_latent_encoded",
        "count": "shape_latent_tokens",
        "data_dir_key": "shape_latent",
        "suffix": ".npz",
    },
    "gaussian_distance_latent": {
        "dirname_prefix": "gaussian_distance_latents",
        "flag": "gaussian_distance_latent_encoded",
        "count": "gaussian_distance_latent_tokens",
        "data_dir_key": "gaussian_distance_latent",
        "suffix": ".npz",
    },
}


def truthy_series(series):
    true_values = {"1", "true", "t", "yes", "y"}
    return series.fillna(False).map(lambda value: str(value).strip().lower() in true_values)


def write_text_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    if not src.exists():
        print(f"Missing latent file: {src}")
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
        description="Create split-specific latent roots using the existing split instance files."
    )
    parser.add_argument("--root", type=Path, required=True, help="Processed dataset root.")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Existing split directory. Defaults to <root>/splits.",
    )
    parser.add_argument(
        "--latent-kind",
        choices=sorted(LATENT_KINDS),
        required=True,
        help="Latent directory/metadata schema to mirror into split-local views.",
    )
    parser.add_argument(
        "--latent-name",
        required=True,
        help="Name of the latent model directory under the canonical latent root.",
    )
    parser.add_argument(
        "--instance-file-name",
        default="instances.txt",
        help="Name of the instance list inside each split directory.",
    )
    parser.add_argument(
        "--allow-unencoded",
        action="store_true",
        help="Do not require the latent encoded flag to be true.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["relative_symlink", "absolute_symlink", "copy", "none"],
        default="relative_symlink",
        help="How split latent files point to the canonical latent directory.",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="Replace existing symlinks/files inside split latent directories.",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Replace existing split latent metadata.csv files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd

    root = args.root.resolve()
    split_root = (args.split_root or root / "splits").resolve()
    schema = LATENT_KINDS[args.latent_kind]
    canonical_parent = root / schema["dirname_prefix"]
    canonical_latent_root = canonical_parent / args.latent_name
    canonical_meta_path = canonical_latent_root / "metadata.csv"

    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")
    if not canonical_latent_root.exists():
        raise FileNotFoundError(f"Canonical latent directory not found: {canonical_latent_root}")
    if not canonical_meta_path.exists():
        raise FileNotFoundError(
            f"Latent metadata not found: {canonical_meta_path}. "
            f"Run the latent encoder and build_metadata.py first."
        )

    metadata = pd.read_csv(canonical_meta_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"{canonical_meta_path} must contain a sha256 column.")
    metadata["sha256"] = metadata["sha256"].astype(str)

    flag_col = schema["flag"]
    count_col = schema["count"]
    required_cols = [flag_col, count_col]
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"{canonical_meta_path} is missing required columns: {missing_cols}")

    if not args.allow_unencoded:
        metadata = metadata[truthy_series(metadata[flag_col])].copy()

    if metadata.empty:
        raise ValueError("No eligible latent rows remain after filtering.")

    split_dirs = sorted(
        p for p in split_root.iterdir()
        if p.is_dir() and (p / args.instance_file_name).exists()
    )
    if not split_dirs:
        raise ValueError(f"No split directories with {args.instance_file_name} found under {split_root}")

    for split_dir in split_dirs:
        with open(split_dir / args.instance_file_name, "r") as f:
            instances = [line.strip() for line in f if line.strip()]
        if not instances:
            raise ValueError(f"{split_dir / args.instance_file_name} is empty.")

        split_latent_root = split_dir / schema["dirname_prefix"] / args.latent_name
        split_meta_path = split_latent_root / "metadata.csv"

        if split_meta_path.exists() and not args.overwrite_metadata:
            raise FileExistsError(f"{split_meta_path} already exists. Use --overwrite-metadata.")

        split_latent_root.mkdir(parents=True, exist_ok=True)

        split_meta = metadata[metadata["sha256"].isin(instances)].drop_duplicates("sha256", keep="first")
        missing_instances = sorted(set(instances) - set(split_meta["sha256"].tolist()))
        if missing_instances:
            print(
                f"{split_dir.name}: {len(missing_instances)} instances missing from canonical latent metadata. "
                f"First few: {missing_instances[:10]}"
            )

        split_meta = split_meta.sort_values("sha256").reset_index(drop=True)
        split_meta[["sha256", flag_col, count_col]].to_csv(split_meta_path, index=False)
        write_text_lines(split_latent_root / "instances.txt", split_meta["sha256"].tolist())

        linked = 0
        for sha256 in split_meta["sha256"]:
            src = canonical_latent_root / f"{sha256}{schema['suffix']}"
            dst = split_latent_root / f"{sha256}{schema['suffix']}"
            linked += int(place_file(src, dst, args.link_mode, args.overwrite_links))

        print(
            f"{split_dir.name}: {len(split_meta)} rows, {linked} latent references, "
            f"data_dir key '{schema['data_dir_key']}' -> {split_latent_root}"
        )


if __name__ == "__main__":
    main()
