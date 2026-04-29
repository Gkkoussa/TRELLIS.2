#!/usr/bin/env python3
"""Create TRELLIS-style metadata.csv for a local tree of OBJ assets."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = (
    "asset_stats",
    "edge_distance_voxels_256",
    "edge_distance_voxels_512",
    "metadata_backups",
    "outputs",
    "pbr_dumps",
    "splits",
    "vertex_distance_voxels_256",
    "vertex_distance_voxels_512",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_obj_paths(root: Path, includes: list[str], exclude_dirs: set[str]):
    seen = set()
    for pattern in includes:
        for path in root.rglob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            rel_parts = path.relative_to(root).parts
            if any(part in exclude_dirs for part in rel_parts):
                continue
            yield path


def build_metadata(args: argparse.Namespace):
    import pandas as pd

    root = args.root.resolve()
    rows = []
    for path in sorted(iter_obj_paths(root, args.include, set(args.exclude_dir))):
        rel = as_posix_relative(path, root)
        if args.id_source == "file_contents":
            sha256 = sha256_file(path)
        else:
            sha256 = sha256_text(f"local_obj:{rel}")

        if args.path_mode == "absolute":
            local_path = str(path.resolve())
        else:
            local_path = rel

        rows.append(
            {
                "sha256": sha256,
                "file_identifier": rel,
                "local_path": local_path,
                "aesthetic_score": args.aesthetic_score,
            }
        )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        return metadata

    duplicate_count = int(metadata.duplicated("sha256").sum())
    if duplicate_count:
        print(f"Dropping {duplicate_count} duplicate sha256 rows.")
        metadata = metadata.drop_duplicates("sha256", keep="first")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create metadata.csv for local OBJ assets arranged in subdirectories."
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing OBJ files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <root>/metadata.csv.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["*.obj"],
        help="Glob pattern(s) to include, relative to --root. Default: *.obj",
    )
    parser.add_argument(
        "--exclude-dir",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_DIRS),
        help="Directory names to skip while recursively searching.",
    )
    parser.add_argument(
        "--id-source",
        choices=["relative_path", "file_contents"],
        default="relative_path",
        help="How to populate the sha256 column. relative_path avoids collisions across duplicate OBJ files.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Store local_path relative to --root or as an absolute path.",
    )
    parser.add_argument(
        "--aesthetic-score",
        type=float,
        default=5.0,
        help="Default score used by TRELLIS filters when no real aesthetic scores exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output metadata.csv if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    output = args.output or root / "metadata.csv"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists. Re-run with --overwrite to replace it.")

    metadata = build_metadata(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False)
    print(f"Wrote {len(metadata)} assets to {output}")


if __name__ == "__main__":
    main()
