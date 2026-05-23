#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def read_split_metadata(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "sha256": row["sha256"],
                    "file_identifier": row.get("file_identifier", ""),
                    "local_path": row["local_path"],
                }
            )
    return rows


def count_obj_topology(args):
    root, split, row = args
    obj_path = os.path.join(root, row["local_path"])
    vertices = 0
    faces = 0
    try:
        with open(obj_path, "rb") as f:
            for line in f:
                if line.startswith(b"v "):
                    vertices += 1
                elif line.startswith(b"f "):
                    faces += 1
        error = ""
    except Exception as exc:
        error = repr(exc)
    return {
        "split": split,
        "sha256": row["sha256"],
        "file_identifier": row["file_identifier"],
        "local_path": row["local_path"],
        "num_vertices": vertices if not error else "",
        "num_faces": faces if not error else "",
        "error": error,
    }


def count_split(root, split, rows, workers):
    tasks = [(root, split, row) for row in rows]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(count_obj_topology, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"count {split}",
            unit="obj",
        ):
            results.append(future.result())
    return results


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k",
        help="OBJXL root directory",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--out-prefix",
        default="objxl_train_test_topology",
        help="Output prefix in the current directory",
    )
    args = parser.parse_args()

    train_csv = os.path.join(args.root, "splits", "train", "metadata.csv")
    test_csv = os.path.join(args.root, "splits", "test", "metadata.csv")
    train_rows = read_split_metadata(train_csv)
    test_rows = read_split_metadata(test_csv)

    print(f"Loaded train={len(train_rows)} test={len(test_rows)}")
    train_counts = count_split(args.root, "train", train_rows, args.workers)
    test_counts = count_split(args.root, "test", test_rows, args.workers)

    count_fields = [
        "split",
        "sha256",
        "file_identifier",
        "local_path",
        "num_vertices",
        "num_faces",
        "error",
    ]
    counts_path = f"{args.out_prefix}_counts.csv"
    write_csv(counts_path, train_counts + test_counts, count_fields)

    train_by_topology = defaultdict(list)
    for row in train_counts:
        if row["error"]:
            continue
        train_by_topology[(row["num_vertices"], row["num_faces"])].append(row)

    matches = []
    for test in test_counts:
        if test["error"]:
            continue
        key = (test["num_vertices"], test["num_faces"])
        for train in train_by_topology.get(key, []):
            matches.append(
                {
                    "num_vertices": key[0],
                    "num_faces": key[1],
                    "test_sha256": test["sha256"],
                    "test_file_identifier": test["file_identifier"],
                    "test_local_path": test["local_path"],
                    "train_sha256": train["sha256"],
                    "train_file_identifier": train["file_identifier"],
                    "train_local_path": train["local_path"],
                    "same_sha256": str(test["sha256"] == train["sha256"]),
                }
            )

    matches_path = f"{args.out_prefix}_test_train_matches.csv"
    write_csv(
        matches_path,
        matches,
        [
            "num_vertices",
            "num_faces",
            "test_sha256",
            "test_file_identifier",
            "test_local_path",
            "train_sha256",
            "train_file_identifier",
            "train_local_path",
            "same_sha256",
        ],
    )

    train_errors = sum(1 for row in train_counts if row["error"])
    test_errors = sum(1 for row in test_counts if row["error"])
    matched_tests = len({row["test_sha256"] for row in matches})
    print(f"Wrote {counts_path}")
    print(f"Wrote {matches_path}")
    print(f"Errors train={train_errors} test={test_errors}")
    print(f"Matched test meshes={matched_tests}; pair rows={len(matches)}")


if __name__ == "__main__":
    main()
