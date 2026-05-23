#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


def resolve_default_matches(path):
    if os.path.exists(path):
        return path
    parent_path = os.path.join("..", path)
    if os.path.exists(parent_path):
        return parent_path
    return path


def read_candidate_meshes(matches_path, root, max_pairs):
    meshes = {}
    pair_count = 0
    with open(matches_path, newline="") as f:
        for row in csv.DictReader(f):
            pair_count += 1
            meshes[row["test_sha256"]] = {
                "split": "test",
                "sha256": row["test_sha256"],
                "file_identifier": row["test_file_identifier"],
                "local_path": row["test_local_path"],
                "path": os.path.join(root, row["test_local_path"]),
                "num_vertices": row["num_vertices"],
                "num_faces": row["num_faces"],
            }
            meshes[row["train_sha256"]] = {
                "split": "train",
                "sha256": row["train_sha256"],
                "file_identifier": row["train_file_identifier"],
                "local_path": row["train_local_path"],
                "path": os.path.join(root, row["train_local_path"]),
                "num_vertices": row["num_vertices"],
                "num_faces": row["num_faces"],
            }
            if max_pairs and pair_count >= max_pairs:
                break
    return list(meshes.values()), pair_count


def parse_obj(path):
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:]:
                    index_token = token.split("/", 1)[0]
                    if not index_token:
                        continue
                    index = int(index_token)
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    face.append(index)
                if len(face) >= 3:
                    faces.append(tuple(face))
    return vertices, faces


def quantize_vertices(vertices, tolerance, normalize):
    if not vertices:
        return []

    verts = [list(v) for v in vertices]
    if normalize:
        mins = [min(v[i] for v in verts) for i in range(3)]
        maxs = [max(v[i] for v in verts) for i in range(3)]
        center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
        diag = sum((maxs[i] - mins[i]) ** 2 for i in range(3)) ** 0.5
        scale = diag if diag > 0 else 1.0
        verts = [[(v[i] - center[i]) / scale for i in range(3)] for v in verts]

    if tolerance <= 0:
        return [tuple(repr(coord) for coord in v) for v in verts]
    return [tuple(int(round(coord / tolerance)) for coord in v) for v in verts]


def canonical_face(face, old_to_new, ignore_winding):
    mapped = tuple(old_to_new[i] for i in face)
    rotations = [mapped[i:] + mapped[:i] for i in range(len(mapped))]
    best = min(rotations)
    if ignore_winding:
        rev = tuple(reversed(mapped))
        rev_rotations = [rev[i:] + rev[:i] for i in range(len(rev))]
        best = min(best, min(rev_rotations))
    return best


def canonical_hash(vertices, faces, tolerance, normalize, ignore_winding):
    quantized = quantize_vertices(vertices, tolerance, normalize)
    order = sorted(range(len(quantized)), key=lambda i: (quantized[i], i))
    old_to_new = {old: new for new, old in enumerate(order)}
    sorted_vertices = [quantized[i] for i in order]
    canonical_faces = sorted(
        canonical_face(face, old_to_new, ignore_winding)
        for face in faces
        if all(0 <= i < len(vertices) for i in face)
    )

    digest = hashlib.sha256()
    digest.update(f"v={len(sorted_vertices)};f={len(canonical_faces)}\n".encode())
    for vertex in sorted_vertices:
        digest.update(("v " + " ".join(map(str, vertex)) + "\n").encode())
    for face in canonical_faces:
        digest.update(("f " + " ".join(map(str, face)) + "\n").encode())
    return digest.hexdigest(), len(sorted_vertices), len(canonical_faces)


def process_mesh(args):
    row, tolerance, normalize, ignore_winding = args
    try:
        vertices, faces = parse_obj(row["path"])
        signature, parsed_vertices, parsed_faces = canonical_hash(
            vertices, faces, tolerance, normalize, ignore_winding
        )
        error = ""
    except Exception as exc:
        signature = ""
        parsed_vertices = ""
        parsed_faces = ""
        error = repr(exc)

    out = dict(row)
    out.update(
        {
            "canonical_signature": signature,
            "parsed_vertices": parsed_vertices,
            "parsed_faces": parsed_faces,
            "error": error,
        }
    )
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def duplicate_rows(signature_rows, scope):
    grouped = defaultdict(list)
    for row in signature_rows:
        if row["error"]:
            continue
        grouped[row["canonical_signature"]].append(row)

    rows = []
    group_id = 0
    for signature, members in grouped.items():
        splits = {m["split"] for m in members}
        if scope == "train" and splits != {"train"}:
            continue
        if scope == "test" and splits != {"test"}:
            continue
        if scope == "cross" and not {"train", "test"}.issubset(splits):
            continue
        if len(members) < 2:
            continue
        group_id += 1
        for member in sorted(members, key=lambda r: (r["split"], r["sha256"])):
            row = dict(member)
            row.update(
                {
                    "duplicate_group": group_id,
                    "group_size": len(members),
                    "train_count": sum(m["split"] == "train" for m in members),
                    "test_count": sum(m["split"] == "test" for m in members),
                    "scope": scope,
                }
            )
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k",
        help="OBJXL root directory",
    )
    parser.add_argument(
        "--matches",
        default="objxl_train_test_topology_test_train_matches.csv",
        help="Topology match CSV from find_objxl_train_test_topology_matches.py",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--ignore-winding", action="store_true")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Center and scale meshes before hashing. More aggressive; use after strict mode.",
    )
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--out-prefix", default="objxl_canonical")
    args = parser.parse_args()

    matches_path = resolve_default_matches(args.matches)
    meshes, pair_count = read_candidate_meshes(matches_path, args.root, args.max_pairs)
    print(f"Loaded candidate pairs={pair_count} unique meshes={len(meshes)}")

    tasks = [
        (mesh, args.tolerance, args.normalize, args.ignore_winding)
        for mesh in meshes
    ]
    signature_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_mesh, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="canonical mesh signatures",
            unit="mesh",
        ):
            signature_rows.append(future.result())

    signature_fields = [
        "split",
        "sha256",
        "file_identifier",
        "local_path",
        "path",
        "num_vertices",
        "num_faces",
        "canonical_signature",
        "parsed_vertices",
        "parsed_faces",
        "error",
    ]
    signatures_path = f"{args.out_prefix}_mesh_signatures.csv"
    write_csv(signatures_path, signature_rows, signature_fields)

    duplicate_fields = [
        "duplicate_group",
        "group_size",
        "train_count",
        "test_count",
        "scope",
    ] + signature_fields

    outputs = {}
    for scope in ("train", "test", "cross"):
        rows = duplicate_rows(signature_rows, scope)
        path = f"{args.out_prefix}_duplicates_{scope}.csv"
        write_csv(path, rows, duplicate_fields)
        outputs[scope] = (path, rows)

    errors = sum(1 for row in signature_rows if row["error"])
    print(f"Wrote {signatures_path}")
    for scope, (path, rows) in outputs.items():
        groups = len({row["duplicate_group"] for row in rows})
        meshes_in_groups = len(rows)
        print(f"Wrote {path}: groups={groups} meshes_in_groups={meshes_in_groups}")
    print(f"Errors={errors}")


if __name__ == "__main__":
    main()
