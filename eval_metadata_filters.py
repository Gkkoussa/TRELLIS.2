import os
from pathlib import Path


def add_eval_metadata_filter_args(parser):
    parser.add_argument(
        "--metadata_filter_csv",
        type=str,
        default=os.environ.get("EVAL_METADATA_FILTER_CSV"),
        help="Comma-separated metadata CSV filters to intersect by sha256.",
    )
    parser.add_argument(
        "--no_train_duplicate_csv",
        type=str,
        default=os.environ.get("NO_TRAIN_DUPLICATE_CSV"),
        help="Metadata CSV containing test samples with train duplicates removed.",
    )
    parser.add_argument(
        "--triangle_filter_csv",
        type=str,
        default=os.environ.get("TRIANGLE_FILTER_CSV"),
        help="Metadata CSV containing samples that pass the triangle-density filter.",
    )
    parser.add_argument(
        "--disable_default_eval_filters",
        action="store_true",
        help="Disable automatic duplicate/triangle filters for split=test.",
    )


def resolve_eval_metadata_filter_csv(root, split, args=None):
    if args is not None and getattr(args, "metadata_filter_csv", None):
        return getattr(args, "metadata_filter_csv")
    if args is not None and getattr(args, "disable_default_eval_filters", False):
        return None
    if split != "test":
        return None

    root = Path(root)
    no_train_duplicate_csv = (
        getattr(args, "no_train_duplicate_csv", None)
        if args is not None else None
    ) or root / "metadata_test_no_train_duplicates" / "metadata_test_no_train_duplicates.csv"
    triangle_filter_csv = (
        getattr(args, "triangle_filter_csv", None)
        if args is not None else None
    ) or root / "metadata_no_triangle_dense.csv"

    filters = [
        root / "splits" / split / "metadata.csv",
        no_train_duplicate_csv,
        triangle_filter_csv,
    ]
    filters = [str(Path(path)) for path in filters if Path(path).exists()]
    return ",".join(filters) if filters else None


def attach_eval_metadata_filter(data_dir, root, split, args=None):
    metadata_filter_csv = resolve_eval_metadata_filter_csv(root, split, args)
    if metadata_filter_csv is None:
        return data_dir
    for source in data_dir.values():
        source["_metadata_filter_csv"] = metadata_filter_csv
    return data_dir
