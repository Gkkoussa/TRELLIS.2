#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = [
    "sha256",
    "split",
    "collection",
    "source_dataset",
    "source_id",
    "metadata_root",
    "pbr_dump_root",
    "local_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic train/holdout manifests for the combined triangle-field dataset."
    )
    parser.add_argument("--old_root", type=Path, required=True)
    parser.add_argument("--old_instances", type=Path, required=True)
    parser.add_argument("--new_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--holdout_size", type=int, default=1000)
    parser.add_argument("--pilot_old", type=int, default=64)
    parser.add_argument("--pilot_new", type=int, default=64)
    parser.add_argument("--shard_size", type=int, default=128)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_instances(path: Path) -> list[str]:
    instances = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if instances and instances[0] == "sha256":
        instances = instances[1:]
    if len(instances) != len(set(instances)):
        raise ValueError(f"Duplicate SHA256 entries in {path}")
    return instances


def validate_sha(value: str, context: str) -> None:
    if len(value) != 64:
        raise ValueError(f"Invalid SHA256 in {context}: {value!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"Invalid SHA256 in {context}: {value!r}") from exc


def index_metadata(path: Path) -> dict[str, dict[str, str]]:
    indexed = {}
    for row in read_csv(path):
        sha256 = row.get("sha256", "")
        validate_sha(sha256, str(path))
        if sha256 in indexed:
            raise ValueError(f"Duplicate SHA256 {sha256} in {path}")
        indexed[sha256] = row
    return indexed


def stable_rank(namespace: str, source: str, sha256: str) -> bytes:
    return hashlib.sha256(f"{namespace}\0{source}\0{sha256}".encode("utf-8")).digest()


def allocate_proportional(counts: dict[str, int], total: int) -> dict[str, int]:
    available = sum(counts.values())
    if total < 0 or total > available:
        raise ValueError(f"Requested {total} selections from only {available} rows")
    exact = {key: total * count / available for key, count in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def stratified_select(rows: list[dict], count: int, namespace: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_dataset"]].append(row)
    quotas = allocate_proportional({key: len(value) for key, value in grouped.items()}, count)
    selected = []
    for source, source_rows in grouped.items():
        source_rows = sorted(
            source_rows,
            key=lambda row: stable_rank(namespace, source, row["sha256"]),
        )
        selected.extend(source_rows[:quotas[source]])
    return sorted(selected, key=lambda row: row["sha256"])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)


def write_instances(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row['sha256']}\n" for row in rows))


def digest_rows(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_shards(output_dir: Path, rows: list[dict], shard_size: int) -> int:
    if shard_size <= 0:
        raise ValueError("--shard_size must be positive")
    shards_dir = output_dir / "shards"
    index_rows = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start:start + shard_size]
        shard_dir = shards_dir / f"shard_{shard_index:05d}"
        old_rows = [row for row in shard_rows if row["collection"] == "legacy_train_no_test"]
        new_rows = [row for row in shard_rows if row["collection"] == "new_meshes"]
        write_csv(shard_dir / "metadata.csv", shard_rows)
        write_instances(shard_dir / "instances.txt", shard_rows)
        write_instances(shard_dir / "old_instances.txt", old_rows)
        write_instances(shard_dir / "new_instances.txt", new_rows)
        index_rows.append({
            "shard_index": shard_index,
            "shard_id": f"shard_{shard_index:05d}",
            "start": start,
            "stop": start + len(shard_rows),
            "instances": len(shard_rows),
            "old_instances": len(old_rows),
            "new_instances": len(new_rows),
        })
    shards_dir.mkdir(parents=True, exist_ok=True)
    with (shards_dir / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)
    return len(index_rows)


def main():
    args = parse_args()
    old_metadata = index_metadata(args.old_root / "metadata.csv")
    new_metadata = index_metadata(args.new_root / "metadata.csv")
    old_instances = read_instances(args.old_instances)

    missing_old = [sha256 for sha256 in old_instances if sha256 not in old_metadata]
    if missing_old:
        raise ValueError(f"{len(missing_old)} old instances are absent from {args.old_root / 'metadata.csv'}")
    overlap = set(old_instances).intersection(new_metadata)
    if overlap:
        raise ValueError(f"Old/new SHA256 overlap contains {len(overlap)} objects")

    old_rows = []
    for sha256 in old_instances:
        source = old_metadata[sha256]
        old_rows.append({
            "sha256": sha256,
            "split": "train",
            "collection": "legacy_train_no_test",
            "source_dataset": source.get("source_dataset") or "legacy_objxl4k",
            "source_id": source.get("source_id") or source.get("file_identifier", ""),
            "metadata_root": str(args.old_root),
            "pbr_dump_root": str(args.old_root),
            "local_path": source.get("local_path", ""),
        })

    new_rows = []
    for sha256, source in sorted(new_metadata.items()):
        new_rows.append({
            "sha256": sha256,
            "split": "train",
            "collection": "new_meshes",
            "source_dataset": source.get("source_dataset") or "unknown",
            "source_id": source.get("source_id", ""),
            "metadata_root": str(args.new_root),
            "pbr_dump_root": str(args.new_root),
            "local_path": source.get("local_path", ""),
        })

    holdout_rows = stratified_select(new_rows, args.holdout_size, "triangle-field-holdout-v1")
    holdout_ids = {row["sha256"] for row in holdout_rows}
    for row in holdout_rows:
        row["split"] = "holdout"
    train_new_rows = [row for row in new_rows if row["sha256"] not in holdout_ids]
    all_rows = old_rows + sorted(new_rows, key=lambda row: row["sha256"])
    train_rows = old_rows + train_new_rows

    pilot_old = sorted(
        old_rows,
        key=lambda row: stable_rank("triangle-field-pilot-v1", row["source_dataset"], row["sha256"]),
    )[:args.pilot_old]
    pilot_new = stratified_select(train_new_rows, args.pilot_new, "triangle-field-pilot-v1")
    pilot_rows = pilot_old + pilot_new

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("all_processed", all_rows), ("train", train_rows), ("holdout", holdout_rows)):
        write_csv(args.output_dir / f"{name}.csv", rows)
        write_instances(args.output_dir / f"{name}_instances.txt", rows)

    pilot_dir = args.output_dir / "pilot"
    write_csv(pilot_dir / "metadata.csv", pilot_rows)
    write_instances(pilot_dir / "instances.txt", pilot_rows)
    write_instances(pilot_dir / "old_instances.txt", pilot_old)
    write_instances(pilot_dir / "new_instances.txt", pilot_new)
    num_shards = write_shards(args.output_dir, all_rows, args.shard_size)

    report = {
        "format": "combined_triangle_field_manifests",
        "version": 1,
        "holdout_selection": "source-stratified stable SHA256 ranking, namespace triangle-field-holdout-v1",
        "pilot_selection": "stable SHA256 ranking, namespace triangle-field-pilot-v1",
        "counts": {
            "old_train_no_test": len(old_rows),
            "new_total": len(new_rows),
            "combined_processed": len(all_rows),
            "train": len(train_rows),
            "holdout": len(holdout_rows),
            "pilot_old": len(pilot_old),
            "pilot_new": len(pilot_new),
            "production_shards": num_shards,
            "production_shard_size": args.shard_size,
        },
        "new_source_counts": dict(sorted(Counter(row["source_dataset"] for row in new_rows).items())),
        "holdout_source_counts": dict(sorted(Counter(row["source_dataset"] for row in holdout_rows).items())),
        "digests": {
            "all_processed_instances_sha256": digest_rows(all_rows),
            "train_instances_sha256": digest_rows(train_rows),
            "holdout_instances_sha256": digest_rows(holdout_rows),
            "pilot_instances_sha256": digest_rows(pilot_rows),
        },
        "paths": {
            "old_root": str(args.old_root),
            "old_instances": str(args.old_instances),
            "new_root": str(args.new_root),
            "output_dir": str(args.output_dir),
        },
    }
    report_path = args.output_dir / "manifest_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
