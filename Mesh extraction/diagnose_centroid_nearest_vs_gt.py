"""
Diagnose centroid-nearest reconstruction against a GT OBJ mesh.

This script answers the practical questions:
  - did we miss GT vertices?
  - did we miss face-centroid blobs?
  - did a centroid choose the wrong three nearest vertices?
  - did the reconstruction create faces that are not GT vertex triples?

It is a diagnostic only. It does not rebuild the mesh.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from extract_barycentric_ridge_graph import infer_resolution, load_triangle_field


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                raw = []
                for token in parts[1:]:
                    value = token.split("/")[0]
                    if not value:
                        continue
                    index = int(value)
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    raw.append(index)
                if len(raw) < 3:
                    continue
                for i in range(1, len(raw) - 1):
                    faces.append([raw[0], raw[i], raw[i + 1]])

    if not vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def bounds_text(points: np.ndarray) -> str:
    if points.size == 0:
        return "empty"
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return (
        f"[{lo[0]:.6f}, {lo[1]:.6f}, {lo[2]:.6f}] -> "
        f"[{hi[0]:.6f}, {hi[1]:.6f}, {hi[2]:.6f}]"
    )


def bbox_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.size == 0 or target.size == 0:
        return source.copy()
    source_center = (source.min(axis=0) + source.max(axis=0)) * 0.5
    target_center = (target.min(axis=0) + target.max(axis=0)) * 0.5
    source_scale = float(np.linalg.norm(source.max(axis=0) - source.min(axis=0)))
    target_scale = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    if source_scale <= 1e-12:
        return source.copy()
    return ((source - source_center[None, :]) * (target_scale / source_scale) + target_center[None, :]).astype(
        np.float32,
        copy=False,
    )


def nearest_indices(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    if queries.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    if references.shape[0] == 0:
        return np.full(queries.shape[0], -1, dtype=np.int32), np.full(queries.shape[0], np.inf, dtype=np.float32)

    nearest = np.zeros(queries.shape[0], dtype=np.int32)
    distances = np.zeros(queries.shape[0], dtype=np.float32)
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        diff = queries[start:stop, None, :] - references[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        local = np.argmin(dist2, axis=1)
        nearest[start:stop] = local.astype(np.int32, copy=False)
        distances[start:stop] = np.sqrt(dist2[np.arange(stop - start), local]).astype(np.float32, copy=False)
    return nearest, distances


def face_key(face: np.ndarray | list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sorted(int(v) for v in face))


def face_key_set(faces: np.ndarray) -> set[tuple[int, int, int]]:
    return {face_key(face) for face in faces}


def nearest_three_vertices(vertices: np.ndarray, point: np.ndarray) -> tuple[int, int, int] | None:
    if vertices.shape[0] < 3:
        return None
    dist2 = np.sum((vertices - point[None, :]) ** 2, axis=1)
    nearest = np.argpartition(dist2, 2)[:3]
    nearest = nearest[np.argsort(dist2[nearest], kind="stable")]
    return face_key(nearest)


def build_centroid_to_face(faces: np.ndarray, face_centroid_id: np.ndarray | None) -> dict[int, tuple[int, int, int]]:
    out = {}
    if face_centroid_id is None:
        return out
    count = min(int(faces.shape[0]), int(face_centroid_id.shape[0]))
    for face_id in range(count):
        out[int(face_centroid_id[face_id])] = face_key(faces[face_id])
    return out


def make_offsets(radius: int) -> list[tuple[int, int, int]]:
    offsets = []
    radius2 = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz <= radius2:
                    offsets.append((dx, dy, dz))
    return offsets


def world_to_voxel_coord(point: np.ndarray, resolution: int) -> tuple[int, int, int]:
    voxel = np.rint((point + 0.5) * float(resolution) - 0.5).astype(np.int32)
    return int(voxel[0]), int(voxel[1]), int(voxel[2])


def probe_field_at_points(
    points: np.ndarray,
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    *,
    radius_voxels: int,
    centroid_dtri_threshold: float,
    centroid_max_dvert: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_to_index = {tuple(c.tolist()): int(i) for i, c in enumerate(coords)}
    offsets = make_offsets(int(radius_voxels))
    local_max_dtri = np.full(points.shape[0], np.nan, dtype=np.float32)
    local_min_dvert = np.full(points.shape[0], np.nan, dtype=np.float32)
    local_centroid_voxels = np.zeros(points.shape[0], dtype=np.int32)

    for point_id, point in enumerate(points):
        x, y, z = world_to_voxel_coord(point, resolution)
        found_dtri = []
        found_dvert = []
        pass_count = 0
        for dx, dy, dz in offsets:
            index = coord_to_index.get((x + dx, y + dy, z + dz))
            if index is None:
                continue
            tri_value = float(d_tri[index])
            vert_value = float(d_vert[index])
            found_dtri.append(tri_value)
            found_dvert.append(vert_value)
            if tri_value >= float(centroid_dtri_threshold) and vert_value <= float(centroid_max_dvert):
                pass_count += 1
        if found_dtri:
            local_max_dtri[point_id] = float(max(found_dtri))
            local_min_dvert[point_id] = float(min(found_dvert))
        local_centroid_voxels[point_id] = int(pass_count)

    return local_max_dtri, local_min_dvert, local_centroid_voxels


def write_summary_csv(path: str | Path, rows: list[tuple[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        for key, value in rows:
            writer.writerow([key, value])


def write_gt_faces_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "gt_face_id",
        "status",
        "gt_v0",
        "gt_v1",
        "gt_v2",
        "mapped_recon_v0",
        "mapped_recon_v1",
        "mapped_recon_v2",
        "max_gt_vertex_match_distance_voxels",
        "nearest_centroid_id",
        "nearest_centroid_distance_voxels",
        "nearest3_from_gt_center",
        "nearest3_from_extracted_centroid",
        "generated_face_from_centroid",
        "mapped_face_exists_in_recon",
        "local_max_d_tri",
        "local_min_d_vert",
        "local_centroid_mask_voxels",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_recon_faces_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "recon_face_id",
        "status",
        "recon_v0",
        "recon_v1",
        "recon_v2",
        "mapped_gt_v0",
        "mapped_gt_v1",
        "mapped_gt_v2",
        "max_recon_vertex_match_distance_voxels",
        "centroid_id",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tuple_text(value: tuple[int, int, int] | None) -> str:
    if value is None:
        return ""
    return f"{value[0]} {value[1]} {value[2]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose centroid-nearest reconstruction against GT OBJ.")
    parser.add_argument("--gt_obj", type=str, default="Mesh extraction/input.obj")
    parser.add_argument(
        "--field",
        type=str,
        default="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst",
    )
    parser.add_argument("--recon_npz", type=str, default="Mesh extraction/results/heart_centroid_nearest.npz")
    parser.add_argument("--out_prefix", type=str, default="Mesh extraction/results/heart_centroid_nearest_gt_diagnosis")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--align_gt_bbox_to_recon", action="store_true")
    parser.add_argument("--vertex_match_radius_voxels", type=float, default=4.0)
    parser.add_argument("--centroid_match_radius_voxels", type=float, default=6.0)
    parser.add_argument("--field_probe_radius_voxels", type=int, default=3)
    parser.add_argument("--centroid_dtri_threshold", type=float, default=None)
    parser.add_argument("--centroid_max_dvert", type=float, default=None)
    args = parser.parse_args()

    gt_vertices, gt_faces = load_obj_mesh(args.gt_obj)
    coords, features = load_triangle_field(args.field)
    resolution = int(args.resolution) if args.resolution is not None else infer_resolution(coords)
    d_tri = np.asarray(features[:, 0], dtype=np.float32)
    d_vert = np.asarray(features[:, 1], dtype=np.float32)

    recon = np.load(args.recon_npz, allow_pickle=False)
    recon_vertices = np.asarray(recon["vertices"], dtype=np.float32)
    recon_faces = np.asarray(recon["faces"], dtype=np.int32)
    centroids = np.asarray(recon["centroid_positions"], dtype=np.float32) if "centroid_positions" in recon else np.zeros((0, 3), dtype=np.float32)
    face_centroid_id = np.asarray(recon["face_centroid_id"], dtype=np.int32) if "face_centroid_id" in recon else None

    centroid_dtri_threshold = (
        float(args.centroid_dtri_threshold)
        if args.centroid_dtri_threshold is not None
        else float(np.asarray(recon["centroid_dtri_threshold"]).reshape(()))
        if "centroid_dtri_threshold" in recon
        else 0.60
    )
    centroid_max_dvert = (
        float(args.centroid_max_dvert)
        if args.centroid_max_dvert is not None
        else float(np.asarray(recon["centroid_max_dvert"]).reshape(()))
        if "centroid_max_dvert" in recon
        else 0.55
    )

    original_gt_bounds = bounds_text(gt_vertices)
    if args.align_gt_bbox_to_recon:
        gt_vertices = bbox_align(gt_vertices, recon_vertices)

    gt_face_centers = gt_vertices[gt_faces].mean(axis=1) if gt_faces.size else np.zeros((0, 3), dtype=np.float32)

    gt_to_recon_vertex, gt_to_recon_dist = nearest_indices(gt_vertices, recon_vertices)
    recon_to_gt_vertex, recon_to_gt_dist = nearest_indices(recon_vertices, gt_vertices)
    gt_to_recon_dist_vox = gt_to_recon_dist * float(resolution)
    recon_to_gt_dist_vox = recon_to_gt_dist * float(resolution)

    gt_center_to_centroid, gt_center_centroid_dist = nearest_indices(gt_face_centers, centroids)
    gt_center_centroid_dist_vox = gt_center_centroid_dist * float(resolution)

    local_max_dtri, local_min_dvert, local_centroid_voxels = probe_field_at_points(
        gt_face_centers,
        coords,
        d_tri,
        d_vert,
        resolution,
        radius_voxels=args.field_probe_radius_voxels,
        centroid_dtri_threshold=centroid_dtri_threshold,
        centroid_max_dvert=centroid_max_dvert,
    )

    gt_keys = face_key_set(gt_faces)
    recon_keys = face_key_set(recon_faces)
    centroid_to_face = build_centroid_to_face(recon_faces, face_centroid_id)

    gt_rows: list[dict[str, object]] = []
    gt_status_counts: Counter[str] = Counter()
    for face_id, gt_face in enumerate(gt_faces):
        mapped_recon = tuple(int(gt_to_recon_vertex[int(v)]) for v in gt_face)
        mapped_recon_key = face_key(mapped_recon)
        mapped_face_exists = mapped_recon_key in recon_keys
        max_vertex_dist = float(gt_to_recon_dist_vox[gt_face].max())
        duplicate_mapped_vertices = len(set(mapped_recon)) < 3

        centroid_id = int(gt_center_to_centroid[face_id]) if gt_center_to_centroid.size else -1
        centroid_dist = float(gt_center_centroid_dist_vox[face_id]) if centroid_id >= 0 else float("inf")
        centroid_missing = centroid_id < 0 or centroid_dist > float(args.centroid_match_radius_voxels)
        centroid_point = centroids[centroid_id] if centroid_id >= 0 else gt_face_centers[face_id]

        nearest3_gt_center = nearest_three_vertices(recon_vertices, gt_face_centers[face_id])
        nearest3_centroid = nearest_three_vertices(recon_vertices, centroid_point) if centroid_id >= 0 else None
        generated_from_centroid = centroid_to_face.get(centroid_id)

        if max_vertex_dist > float(args.vertex_match_radius_voxels):
            status = "missing_or_shifted_vertex"
        elif duplicate_mapped_vertices:
            status = "gt_vertices_collapsed_to_same_recon_vertex"
        elif mapped_face_exists:
            status = "matched_face_exists"
        elif centroid_missing:
            status = "missing_centroid_blob"
        elif generated_from_centroid is None:
            status = "centroid_did_not_write_face"
        elif generated_from_centroid != mapped_recon_key:
            status = "centroid_generated_wrong_face"
        elif nearest3_centroid != mapped_recon_key:
            status = "wrong_nearest3_from_centroid"
        else:
            status = "missing_for_unknown_reason"

        gt_status_counts[status] += 1
        gt_rows.append({
            "gt_face_id": int(face_id),
            "status": status,
            "gt_v0": int(gt_face[0]),
            "gt_v1": int(gt_face[1]),
            "gt_v2": int(gt_face[2]),
            "mapped_recon_v0": int(mapped_recon[0]),
            "mapped_recon_v1": int(mapped_recon[1]),
            "mapped_recon_v2": int(mapped_recon[2]),
            "max_gt_vertex_match_distance_voxels": f"{max_vertex_dist:.6f}",
            "nearest_centroid_id": int(centroid_id),
            "nearest_centroid_distance_voxels": f"{centroid_dist:.6f}",
            "nearest3_from_gt_center": tuple_text(nearest3_gt_center),
            "nearest3_from_extracted_centroid": tuple_text(nearest3_centroid),
            "generated_face_from_centroid": tuple_text(generated_from_centroid),
            "mapped_face_exists_in_recon": int(mapped_face_exists),
            "local_max_d_tri": f"{float(local_max_dtri[face_id]):.6f}" if not np.isnan(local_max_dtri[face_id]) else "",
            "local_min_d_vert": f"{float(local_min_dvert[face_id]):.6f}" if not np.isnan(local_min_dvert[face_id]) else "",
            "local_centroid_mask_voxels": int(local_centroid_voxels[face_id]),
        })

    recon_rows: list[dict[str, object]] = []
    recon_status_counts: Counter[str] = Counter()
    for face_id, recon_face in enumerate(recon_faces):
        mapped_gt = tuple(int(recon_to_gt_vertex[int(v)]) for v in recon_face)
        mapped_gt_key = face_key(mapped_gt)
        max_vertex_dist = float(recon_to_gt_dist_vox[recon_face].max())
        if max_vertex_dist > float(args.vertex_match_radius_voxels):
            status = "uses_unmatched_or_shifted_vertex"
        elif len(set(mapped_gt)) < 3:
            status = "recon_vertices_collapse_to_same_gt_vertex"
        elif mapped_gt_key in gt_keys:
            status = "matches_gt"
        else:
            status = "non_gt_vertex_triple"
        recon_status_counts[status] += 1
        centroid_id = int(face_centroid_id[face_id]) if face_centroid_id is not None and face_id < face_centroid_id.shape[0] else -1
        recon_rows.append({
            "recon_face_id": int(face_id),
            "status": status,
            "recon_v0": int(recon_face[0]),
            "recon_v1": int(recon_face[1]),
            "recon_v2": int(recon_face[2]),
            "mapped_gt_v0": int(mapped_gt[0]),
            "mapped_gt_v1": int(mapped_gt[1]),
            "mapped_gt_v2": int(mapped_gt[2]),
            "max_recon_vertex_match_distance_voxels": f"{max_vertex_dist:.6f}",
            "centroid_id": centroid_id,
        })

    duplicate_recon_assignments = int(gt_vertices.shape[0] - len(set(int(v) for v in gt_to_recon_vertex.tolist())))
    missing_gt_vertices = int(np.sum(gt_to_recon_dist_vox > float(args.vertex_match_radius_voxels)))
    unmatched_recon_vertices = int(np.sum(recon_to_gt_dist_vox > float(args.vertex_match_radius_voxels)))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    gt_faces_path = out_prefix.with_name(out_prefix.name + "_gt_faces.csv")
    recon_faces_path = out_prefix.with_name(out_prefix.name + "_recon_faces.csv")

    summary_rows: list[tuple[str, object]] = [
        ("gt_obj", args.gt_obj),
        ("field", args.field),
        ("recon_npz", args.recon_npz),
        ("resolution", resolution),
        ("gt_vertices", int(gt_vertices.shape[0])),
        ("gt_faces", int(gt_faces.shape[0])),
        ("recon_vertices", int(recon_vertices.shape[0])),
        ("recon_faces", int(recon_faces.shape[0])),
        ("centroids", int(centroids.shape[0])),
        ("vertex_match_radius_voxels", float(args.vertex_match_radius_voxels)),
        ("centroid_match_radius_voxels", float(args.centroid_match_radius_voxels)),
        ("field_probe_radius_voxels", int(args.field_probe_radius_voxels)),
        ("centroid_dtri_threshold", centroid_dtri_threshold),
        ("centroid_max_dvert", centroid_max_dvert),
        ("gt_bounds_original", original_gt_bounds),
        ("gt_bounds_used", bounds_text(gt_vertices)),
        ("recon_bounds", bounds_text(recon_vertices)),
        ("missing_gt_vertices", missing_gt_vertices),
        ("unmatched_recon_vertices", unmatched_recon_vertices),
        ("duplicate_gt_to_recon_vertex_assignments", duplicate_recon_assignments),
    ]
    for status, count in sorted(gt_status_counts.items()):
        summary_rows.append((f"gt_face_status_{status}", int(count)))
    for status, count in sorted(recon_status_counts.items()):
        summary_rows.append((f"recon_face_status_{status}", int(count)))

    write_summary_csv(summary_path, summary_rows)
    write_gt_faces_csv(gt_faces_path, gt_rows)
    write_recon_faces_csv(recon_faces_path, recon_rows)

    print("Centroid-nearest GT diagnosis")
    print(f"Resolution: {resolution}")
    print(f"GT: {gt_vertices.shape[0]} vertices, {gt_faces.shape[0]} faces")
    print(f"Recon: {recon_vertices.shape[0]} vertices, {recon_faces.shape[0]} faces")
    print(f"Centroids: {centroids.shape[0]}")
    print(f"Missing GT vertices > {args.vertex_match_radius_voxels} voxels: {missing_gt_vertices}")
    print(f"Unmatched recon vertices > {args.vertex_match_radius_voxels} voxels: {unmatched_recon_vertices}")
    print("GT face statuses:")
    for status, count in sorted(gt_status_counts.items()):
        print(f"  {status}: {count}")
    print("Recon face statuses:")
    for status, count in sorted(recon_status_counts.items()):
        print(f"  {status}: {count}")
    print(f"Saved {summary_path}")
    print(f"Saved {gt_faces_path}")
    print(f"Saved {recon_faces_path}")


if __name__ == "__main__":
    main()
