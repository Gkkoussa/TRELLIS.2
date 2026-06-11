#!/usr/bin/env python3
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


def normalize_points(points):
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    points = points - center
    scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def stable_sample_mesh(path, n_points, seed):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if len(mesh.faces) == 0:
        points = np.asarray(mesh.vertices, dtype=np.float32)
        if len(points) == 0:
            raise ValueError("mesh has no vertices/faces")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=n_points, replace=len(points) < n_points)
        return normalize_points(points[idx])
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    finally:
        np.random.set_state(state)
    return normalize_points(np.asarray(points, dtype=np.float32))


def sample_one(args):
    sha, path, n_points, cache_dir = args
    out_path = os.path.join(cache_dir, f"{sha}.npy")
    if os.path.exists(out_path):
        return sha, out_path, ""
    try:
        seed = int(sha[:16], 16) % (2**32)
        points = stable_sample_mesh(path, n_points, seed)
        tmp_path = f"{out_path}.tmp.{os.getpid()}"
        np.save(tmp_path, points)
        os.replace(f"{tmp_path}.npy", out_path)
        return sha, out_path, ""
    except Exception as exc:
        return sha, out_path, repr(exc)


def chamfer(points_a, points_b):
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    d_ab, _ = tree_b.query(points_a, k=1, workers=1)
    d_ba, _ = tree_a.query(points_b, k=1, workers=1)
    return float((np.mean(d_ab ** 2) + np.mean(d_ba ** 2)) * 0.5)


def load_candidates(path, root, max_pairs):
    candidates = []
    meshes = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            test_sha = row["test_sha256"]
            train_sha = row["train_sha256"]
            test_path = os.path.join(root, row["test_local_path"])
            train_path = os.path.join(root, row["train_local_path"])
            meshes[test_sha] = test_path
            meshes[train_sha] = train_path
            candidates.append(row)
            if max_pairs and len(candidates) >= max_pairs:
                break
    return candidates, meshes


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
        help="Topology match CSV produced by find_objxl_train_test_topology_matches.py",
    )
    parser.add_argument(
        "--out",
        default="objxl_train_test_topology_chamfer.csv",
        help="Output CSV with Chamfer scores",
    )
    parser.add_argument(
        "--cache-dir",
        default="objxl_chamfer_point_cache",
        help="Directory for sampled point clouds",
    )
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Normalized squared symmetric Chamfer threshold to flag likely same geometry",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Debug limit. 0 means all pairs.",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    candidates, meshes = load_candidates(args.matches, args.root, args.max_pairs)
    print(f"Loaded candidate pairs={len(candidates)} unique meshes={len(meshes)}")

    sample_tasks = [
        (sha, path, args.points, args.cache_dir) for sha, path in sorted(meshes.items())
    ]
    errors = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sample_one, task) for task in sample_tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="sample/cache point clouds",
            unit="mesh",
        ):
            sha, _, error = future.result()
            if error:
                errors[sha] = error

    point_cache = {}

    def points_for(sha):
        points = point_cache.get(sha)
        if points is None:
            points = np.load(os.path.join(args.cache_dir, f"{sha}.npy"))
            point_cache[sha] = points
        return points

    fieldnames = list(candidates[0].keys()) + [
        "normalized_squared_chamfer",
        "likely_same_geometry",
        "error",
    ]
    likely = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        progress = tqdm(candidates, total=len(candidates), desc="chamfer pairs", unit="pair")
        for row in progress:
            test_sha = row["test_sha256"]
            train_sha = row["train_sha256"]
            error = errors.get(test_sha) or errors.get(train_sha) or ""
            score = ""
            same = "False"
            if not error:
                try:
                    score_value = chamfer(points_for(test_sha), points_for(train_sha))
                    score = f"{score_value:.10g}"
                    same = str(score_value <= args.threshold)
                    likely += score_value <= args.threshold
                except Exception as exc:
                    error = repr(exc)
            out = dict(row)
            out.update(
                {
                    "normalized_squared_chamfer": score,
                    "likely_same_geometry": same,
                    "error": error,
                }
            )
            writer.writerow(out)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(likely_same=likely)

    print(f"Wrote {args.out}")
    print(f"Likely same geometry at threshold {args.threshold}: {likely}")


if __name__ == "__main__":
    main()
