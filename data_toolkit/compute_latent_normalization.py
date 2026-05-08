#!/usr/bin/env python3
"""Compute per-channel latent normalization statistics from a latent root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LATENT_KINDS = {
    "shape_latent": {
        "flag": "shape_latent_encoded",
        "count": "shape_latent_tokens",
        "suffix": ".npz",
    },
    "gaussian_distance_latent": {
        "flag": "gaussian_distance_latent_encoded",
        "count": "gaussian_distance_latent_tokens",
        "suffix": ".npz",
    },
    "michelangelo_latent": {
        "flag": "michelangelo_latent_encoded",
        "count": "michelangelo_latent_tokens",
        "suffix": ".npz",
    },
    "pbr_latent": {
        "flag": "pbr_latent_encoded",
        "count": "pbr_latent_tokens",
        "suffix": ".npz",
    },
}


def truthy_series(series):
    true_values = {"1", "true", "t", "yes", "y"}
    return series.fillna(False).map(lambda value: str(value).strip().lower() in true_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std for latent features using token-weighted statistics."
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        required=True,
        help="Directory containing latent .npz files and metadata.csv.",
    )
    parser.add_argument(
        "--latent-kind",
        choices=sorted(LATENT_KINDS),
        required=True,
        help="Metadata schema to use for validation/filtering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <latent-root>/normalization.json.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of latent files to use.",
    )
    parser.add_argument(
        "--allow-unencoded",
        action="store_true",
        help="Do not require the latent encoded flag to be true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = LATENT_KINDS[args.latent_kind]
    latent_root = args.latent_root.resolve()
    metadata_path = latent_root / "metadata.csv"
    output_path = (args.output or (latent_root / "normalization.json")).resolve()

    if not latent_root.exists():
        raise FileNotFoundError(f"Latent root not found: {latent_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain a sha256 column.")

    flag_col = schema["flag"]
    count_col = schema["count"]
    if flag_col not in metadata.columns:
        raise ValueError(f"{metadata_path} is missing required column: {flag_col}")
    if count_col not in metadata.columns:
        raise ValueError(f"{metadata_path} is missing required column: {count_col}")

    metadata["sha256"] = metadata["sha256"].astype(str)
    if not args.allow_unencoded:
        metadata = metadata[truthy_series(metadata[flag_col])].copy()
    metadata = metadata.drop_duplicates("sha256", keep="first")
    if args.max_files is not None:
        metadata = metadata.head(args.max_files)
    if metadata.empty:
        raise ValueError("No eligible latent rows remain after filtering.")

    sum_feats = None
    sum_sq_feats = None
    total_tokens = 0
    files_used = 0

    for sha256 in metadata["sha256"]:
        npz_path = latent_root / f"{sha256}{schema['suffix']}"
        if not npz_path.exists():
            print(f"Missing latent file: {npz_path}")
            continue
        data = np.load(npz_path)
        if "feats" not in data:
            print(f"Missing feats array in {npz_path}")
            continue

        feats = data["feats"].astype(np.float64, copy=False)
        if feats.ndim != 2:
            print(f"Unexpected feats shape in {npz_path}: {feats.shape}")
            continue
        if feats.shape[0] == 0:
            print(f"Empty feats in {npz_path}")
            continue
        if not np.isfinite(feats).all():
            print(f"Non-finite feats in {npz_path}")
            continue

        if sum_feats is None:
            sum_feats = np.zeros(feats.shape[1], dtype=np.float64)
            sum_sq_feats = np.zeros(feats.shape[1], dtype=np.float64)

        sum_feats += feats.sum(axis=0)
        sum_sq_feats += np.square(feats).sum(axis=0)
        total_tokens += feats.shape[0]
        files_used += 1

    if files_used == 0 or total_tokens == 0:
        raise ValueError("No usable latent files were found.")

    mean = sum_feats / total_tokens
    var = np.maximum(sum_sq_feats / total_tokens - np.square(mean), 0.0)
    std = np.sqrt(var)

    payload = {
        "latent_kind": args.latent_kind,
        "latent_root": str(latent_root),
        "files_used": int(files_used),
        "total_tokens": int(total_tokens),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
