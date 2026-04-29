#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_keep_paths(merged_records_dir: Path) -> set[str]:
    keep_paths: set[str] = set()
    for csv_path in sorted(merged_records_dir.glob("*.csv")):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                local_path = row.get("local_path")
                if local_path:
                    keep_paths.add(local_path)
    return keep_paths


def prune_empty_dirs(root: Path, dry_run: bool) -> int:
    removed = 0
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            next(path.iterdir())
        except StopIteration:
            print(f"Removing empty dir: {path}")
            if not dry_run:
                path.rmdir()
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing raw/merged_records and raw/hf-objaverse-v1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without removing anything",
    )
    args = parser.parse_args()

    raw_root = args.dataset_root / "raw"
    merged_records_dir = raw_root / "merged_records"
    hf_root = raw_root / "hf-objaverse-v1"

    if not merged_records_dir.exists():
        raise FileNotFoundError(f"Merged records directory not found: {merged_records_dir}")
    if not hf_root.exists():
        raise FileNotFoundError(f"Downloaded raw directory not found: {hf_root}")

    keep_paths = load_keep_paths(merged_records_dir)
    if not keep_paths:
        raise ValueError(f"No local_path entries found in merged records: {merged_records_dir}")

    deleted_files = 0
    kept_files = 0
    for path in sorted(p for p in hf_root.rglob("*") if p.is_file()):
        rel_to_dataset = path.relative_to(args.dataset_root).as_posix()
        if rel_to_dataset in keep_paths:
            kept_files += 1
            continue
        print(f"Deleting file: {path}")
        if not args.dry_run:
            path.unlink()
        deleted_files += 1

    removed_dirs = prune_empty_dirs(hf_root, dry_run=args.dry_run)

    print(f"Kept files: {kept_files}")
    print(f"Deleted files: {deleted_files}")
    print(f"Removed empty dirs: {removed_dirs}")
    if args.dry_run:
        print("Dry run only; no files were deleted.")


if __name__ == "__main__":
    main()
