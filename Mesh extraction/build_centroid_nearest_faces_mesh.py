"""
Build a mesh by connecting each face-centroid blob to its three nearest vertices.

This is intentionally simple:
  - high d_vert blobs become mesh vertices
  - high d_tri / low d_vert blobs become face centroids
  - each vertex blob and centroid blob is collapsed to a tiny core for debug PLYs
  - each centroid chooses its three nearest vertices and creates one triangle face

Only coords, d_tri, and d_vert are used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from extract_barycentric_ridge_graph import (
    component_core_indices,
    component_position,
    connected_components,
    default_output_stem,
    default_results_dir,
    extract_vertex_blobs,
    infer_resolution,
    load_triangle_field,
    threshold_to_raw_max_barycentric,
    voxel_centers,
)


ALGO_NAME = "Barycentric Centroid Nearest-Face Mesh"


def raw_min_barycentric_from_dtri(value: float) -> float:
    return float(value) / 3.0


def sorted_edge(edge: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    if a == b:
        raise ValueError("self-edge is not allowed")
    return (a, b) if a < b else (b, a)


def edge_array_from_faces(faces: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        edges.add(sorted_edge((a, b)))
        edges.add(sorted_edge((b, c)))
        edges.add(sorted_edge((c, a)))
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(edges), dtype=np.int32)


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    center = vertices.mean(axis=0) if vertices.size else np.zeros(3, dtype=np.float32)
    oriented = faces.copy()
    for i, face in enumerate(oriented):
        p0, p1, p2 = vertices[face]
        normal = np.cross(p1 - p0, p2 - p0)
        face_center = (p0 + p1 + p2) / 3.0
        if float(np.dot(normal, face_center - center)) < 0.0:
            oriented[i] = np.asarray([face[0], face[2], face[1]], dtype=np.int32)
    return oriented


def extract_centroid_blobs(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    *,
    dtri_threshold: float,
    max_dvert: float,
    connectivity: int,
    min_component_size: int,
    target_count: int,
    fill_small_components_by: str,
    position_mode: str,
    core_mode: str,
) -> dict[str, np.ndarray]:
    mask = (d_tri >= float(dtri_threshold)) & (d_vert <= float(max_dvert))
    components = connected_components(coords, mask, connectivity=connectivity)

    records = []
    small_records = []
    for component_id, component in enumerate(components):
        position = component_position(
            coords,
            d_tri,
            component,
            resolution,
            dtri_threshold,
            position_mode,
        )
        core_idx = component_core_indices(
            coords,
            d_tri,
            component,
            resolution,
            position,
            core_mode,
        )
        record = {
            "component_id": int(component_id),
            "component": component,
            "position": position,
            "core_indices": core_idx,
            "size": int(component.size),
            "peak_d_tri": float(d_tri[component].max()),
            "mean_d_vert": float(d_vert[component].mean()),
        }
        if component.size >= int(min_component_size):
            records.append(record)
        else:
            small_records.append(record)

    target_fill_count = 0
    target = int(target_count)
    if target > 0 and len(records) < target and fill_small_components_by != "none":
        needed = target - len(records)
        if fill_small_components_by == "peak_dtri":
            small_records.sort(key=lambda item: (-float(item["peak_d_tri"]), -int(item["size"]), int(item["component_id"])))
        elif fill_small_components_by == "size_then_peak_dtri":
            small_records.sort(key=lambda item: (-int(item["size"]), -float(item["peak_d_tri"]), int(item["component_id"])))
        else:
            raise ValueError(f"Unsupported fill_small_components_by: {fill_small_components_by}")
        fill_records = small_records[:needed]
        records.extend(fill_records)
        target_fill_count = int(len(fill_records))

    records.sort(key=lambda item: int(item["component_id"]))

    positions = []
    core_indices = []
    core_counts = []
    peak_dtris = []
    mean_dverts = []
    sizes = []
    centroid_id_for_index = np.full(coords.shape[0], -1, dtype=np.int32)
    centroid_core_id_for_index = np.full(coords.shape[0], -1, dtype=np.int32)
    centroid_mask = np.zeros(coords.shape[0], dtype=bool)
    centroid_core_mask = np.zeros(coords.shape[0], dtype=bool)

    for record in records:
        component = np.asarray(record["component"], dtype=np.int64)
        centroid_id = len(positions)
        centroid_id_for_index[component] = centroid_id
        centroid_mask[component] = True

        position = np.asarray(record["position"], dtype=np.float32)
        core_idx = np.asarray(record["core_indices"], dtype=np.int64)
        if core_idx.size:
            centroid_core_mask[core_idx] = True
            centroid_core_id_for_index[core_idx] = centroid_id
            core_indices.append(int(core_idx[0]))
        else:
            core_indices.append(-1)

        positions.append(position)
        core_counts.append(int(core_idx.size))
        peak_dtris.append(float(record["peak_d_tri"]))
        mean_dverts.append(float(record["mean_d_vert"]))
        sizes.append(int(record["size"]))

    if positions:
        centroids = np.stack(positions).astype(np.float32, copy=False)
    else:
        centroids = np.zeros((0, 3), dtype=np.float32)

    return {
        "centroids": centroids,
        "centroid_peak_d_tri": np.asarray(peak_dtris, dtype=np.float32),
        "centroid_mean_d_vert": np.asarray(mean_dverts, dtype=np.float32),
        "centroid_component_size": np.asarray(sizes, dtype=np.int32),
        "centroid_core_index": np.asarray(core_indices, dtype=np.int32),
        "centroid_core_count": np.asarray(core_counts, dtype=np.int32),
        "centroid_id_for_index": centroid_id_for_index,
        "centroid_core_id_for_index": centroid_core_id_for_index,
        "centroid_mask": centroid_mask,
        "centroid_core_mask": centroid_core_mask,
        "centroid_components_total": np.asarray(len(components), dtype=np.int32),
        "centroid_components_before_target": np.asarray(len(records) - target_fill_count, dtype=np.int32),
        "centroid_small_components_available": np.asarray(len(small_records), dtype=np.int32),
        "centroid_target_fill_count": np.asarray(target_fill_count, dtype=np.int32),
    }


def nearest_faces_from_centroids(
    vertices: np.ndarray,
    centroids: np.ndarray,
    *,
    max_vertex_distance_voxels: float,
    resolution: int,
    min_face_area: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    if vertices.shape[0] < 3 or centroids.shape[0] == 0:
        empty_stats = {
            "face_centroid_id": np.zeros((0,), dtype=np.int32),
            "face_distance_sum_voxels": np.zeros((0,), dtype=np.float32),
            "face_nearest_max_distance_voxels": np.zeros((0,), dtype=np.float32),
            "face_area": np.zeros((0,), dtype=np.float32),
            "rejected_too_few_vertices": int(centroids.shape[0]) if vertices.shape[0] < 3 else 0,
            "rejected_distance": 0,
            "rejected_degenerate": 0,
            "duplicate_faces": 0,
        }
        return np.zeros((0, 3), dtype=np.int32), empty_stats

    face_by_key: dict[tuple[int, int, int], dict[str, int | float | tuple[int, int, int]]] = {}
    rejected_distance = 0
    rejected_degenerate = 0
    duplicate_faces = 0

    for centroid_id, centroid in enumerate(centroids):
        diff = vertices - centroid[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        nearest = np.argpartition(dist2, 2)[:3]
        nearest = nearest[np.argsort(dist2[nearest], kind="stable")]

        distances_voxels = np.sqrt(dist2[nearest]) * float(resolution)
        max_distance = float(distances_voxels.max())
        if float(max_vertex_distance_voxels) > 0.0 and max_distance > float(max_vertex_distance_voxels):
            rejected_distance += 1
            continue

        face = tuple(int(v) for v in nearest.tolist())
        p0, p1, p2 = vertices[np.asarray(face, dtype=np.int32)]
        area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        if area < float(min_face_area):
            rejected_degenerate += 1
            continue

        key = tuple(sorted(face))
        score = float(distances_voxels.sum())
        record = {
            "face": face,
            "centroid_id": int(centroid_id),
            "distance_sum_voxels": score,
            "max_vertex_distance_voxels": max_distance,
            "area": float(area),
        }
        old = face_by_key.get(key)
        if old is None:
            face_by_key[key] = record
        else:
            duplicate_faces += 1
            if score < float(old["distance_sum_voxels"]):
                face_by_key[key] = record

    records = [face_by_key[key] for key in sorted(face_by_key)]
    if records:
        faces = np.asarray([record["face"] for record in records], dtype=np.int32)
    else:
        faces = np.zeros((0, 3), dtype=np.int32)

    stats = {
        "face_centroid_id": np.asarray([record["centroid_id"] for record in records], dtype=np.int32),
        "face_distance_sum_voxels": np.asarray(
            [record["distance_sum_voxels"] for record in records],
            dtype=np.float32,
        ),
        "face_nearest_max_distance_voxels": np.asarray(
            [record["max_vertex_distance_voxels"] for record in records],
            dtype=np.float32,
        ),
        "face_area": np.asarray([record["area"] for record in records], dtype=np.float32),
        "rejected_too_few_vertices": 0,
        "rejected_distance": int(rejected_distance),
        "rejected_degenerate": int(rejected_degenerate),
        "duplicate_faces": int(duplicate_faces),
    }
    return faces, stats


def extract_mesh(
    coords: np.ndarray,
    features: np.ndarray,
    resolution: int,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | int | float | str]]:
    d_tri = np.asarray(features[:, 0], dtype=np.float32)
    d_vert = np.asarray(features[:, 1], dtype=np.float32)

    vertex_data = extract_vertex_blobs(
        coords,
        d_vert,
        resolution,
        threshold=args.vertex_dvert_threshold,
        connectivity=args.vertex_connectivity,
        min_component_size=args.vertex_min_component_size,
        position_mode=args.vertex_position_mode,
        core_mode=args.vertex_core_mode,
    )
    vertices = vertex_data["vertices"]

    centroid_data = extract_centroid_blobs(
        coords,
        d_tri,
        d_vert,
        resolution,
        dtri_threshold=args.centroid_dtri_threshold,
        max_dvert=args.centroid_max_dvert,
        connectivity=args.centroid_connectivity,
        min_component_size=args.centroid_min_component_size,
        target_count=args.centroid_target_count,
        fill_small_components_by=args.centroid_fill_small_components_by,
        position_mode=args.centroid_position_mode,
        core_mode=args.centroid_core_mode,
    )
    centroids = centroid_data["centroids"]

    faces, face_stats = nearest_faces_from_centroids(
        vertices,
        centroids,
        max_vertex_distance_voxels=args.face_max_vertex_distance_voxels,
        resolution=resolution,
        min_face_area=args.min_face_area,
    )
    if not args.no_orient_outward:
        faces = orient_faces_outward(vertices, faces)
    edges = edge_array_from_faces(faces)

    stats: dict[str, np.ndarray | int | float | str] = {
        "algorithm": ALGO_NAME,
        "resolution": int(resolution),
        "vertices": int(vertices.shape[0]),
        "centroids": int(centroids.shape[0]),
        "faces": int(faces.shape[0]),
        "edges": int(edges.shape[0]),
        "vertex_dvert_threshold": float(args.vertex_dvert_threshold),
        "vertex_raw_max_barycentric": float(threshold_to_raw_max_barycentric(args.vertex_dvert_threshold)),
        "vertex_connectivity": int(args.vertex_connectivity),
        "vertex_position_mode": str(args.vertex_position_mode),
        "vertex_core_mode": str(args.vertex_core_mode),
        "vertex_blob_voxels": int(vertex_data["vertex_mask"].sum()),
        "vertex_core_voxels": int(vertex_data["vertex_core_mask"].sum()),
        "centroid_dtri_threshold": float(args.centroid_dtri_threshold),
        "centroid_raw_min_barycentric": raw_min_barycentric_from_dtri(args.centroid_dtri_threshold),
        "centroid_max_dvert": float(args.centroid_max_dvert),
        "centroid_raw_max_barycentric": float(threshold_to_raw_max_barycentric(args.centroid_max_dvert)),
        "centroid_connectivity": int(args.centroid_connectivity),
        "centroid_position_mode": str(args.centroid_position_mode),
        "centroid_core_mode": str(args.centroid_core_mode),
        "centroid_blob_voxels": int(centroid_data["centroid_mask"].sum()),
        "centroid_core_voxels": int(centroid_data["centroid_core_mask"].sum()),
        "centroid_min_component_size": int(args.centroid_min_component_size),
        "centroid_target_count": int(args.centroid_target_count),
        "centroid_fill_small_components_by": str(args.centroid_fill_small_components_by),
        "centroid_components_total": int(np.asarray(centroid_data["centroid_components_total"]).reshape(())),
        "centroid_components_before_target": int(
            np.asarray(centroid_data["centroid_components_before_target"]).reshape(())
        ),
        "centroid_small_components_available": int(
            np.asarray(centroid_data["centroid_small_components_available"]).reshape(())
        ),
        "centroid_target_fill_count": int(np.asarray(centroid_data["centroid_target_fill_count"]).reshape(())),
        "face_max_vertex_distance_voxels": float(args.face_max_vertex_distance_voxels),
        "min_face_area": float(args.min_face_area),
        "rejected_too_few_vertices": int(face_stats["rejected_too_few_vertices"]),
        "rejected_distance": int(face_stats["rejected_distance"]),
        "rejected_degenerate": int(face_stats["rejected_degenerate"]),
        "duplicate_faces": int(face_stats["duplicate_faces"]),
        "vertex_peak_d_vert": vertex_data["vertex_peak_d_vert"],
        "vertex_component_size": vertex_data["vertex_component_size"],
        "vertex_core_count": vertex_data["vertex_core_count"],
        "centroid_peak_d_tri": centroid_data["centroid_peak_d_tri"],
        "centroid_mean_d_vert": centroid_data["centroid_mean_d_vert"],
        "centroid_component_size": centroid_data["centroid_component_size"],
        "centroid_core_count": centroid_data["centroid_core_count"],
        "centroid_positions": centroids,
        "face_centroid_id": face_stats["face_centroid_id"],
        "face_distance_sum_voxels": face_stats["face_distance_sum_voxels"],
        "face_nearest_max_distance_voxels": face_stats["face_nearest_max_distance_voxels"],
        "face_area": face_stats["face_area"],
        "vertex_core_mask": vertex_data["vertex_core_mask"],
        "centroid_core_mask": centroid_data["centroid_core_mask"],
    }
    return vertices, edges, faces, stats


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {ALGO_NAME}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            f.write(f"f {a} {b} {c}\n")


def write_mesh_ply(path: str | Path, vertices: np.ndarray, edges: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment algorithm {ALGO_NAME}\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for edge in edges:
            f.write(f"{int(edge[0])} {int(edge[1])}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_core_voxel_ply(
    path: str | Path,
    coords: np.ndarray,
    resolution: int,
    vertex_core_mask: np.ndarray,
    centroid_core_mask: np.ndarray,
) -> None:
    labels = np.zeros(coords.shape[0], dtype=np.uint8)
    labels[vertex_core_mask] = 1
    labels[centroid_core_mask] = 2
    selected = np.flatnonzero(labels > 0)
    points = voxel_centers(coords[selected], resolution) if selected.size else np.zeros((0, 3), dtype=np.float32)
    colors = {
        1: (255, 40, 40),
        2: (40, 230, 120),
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment algorithm {ALGO_NAME}\n")
        f.write("comment color 1 vertex_core rgb=(255,40,40)\n")
        f.write("comment color 2 centroid_core rgb=(40,230,120)\n")
        f.write(f"element vertex {selected.size}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")
        for point, label in zip(points, labels[selected]):
            r, g, b = colors[int(label)]
            f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {r} {g} {b} {int(label)}\n")


def write_npz(
    path: str | Path,
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    stats: dict[str, np.ndarray | int | float | str],
) -> None:
    payload = {
        "vertices": vertices.astype(np.float32, copy=False),
        "edges": edges.astype(np.int32, copy=False),
        "faces": faces.astype(np.int32, copy=False),
    }
    for key, value in stats.items():
        if key.endswith("_mask"):
            continue
        if isinstance(value, np.ndarray):
            payload[key] = value
        elif isinstance(value, str):
            payload[key] = np.asarray(value)
        elif isinstance(value, int):
            payload[key] = np.asarray(value, dtype=np.int32)
        else:
            payload[key] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(path, **payload)


def write_csv(path: str | Path, faces: np.ndarray, stats: dict[str, np.ndarray | int | float | str]) -> None:
    scalar_keys = [
        key
        for key, value in stats.items()
        if not isinstance(value, np.ndarray) and not key.endswith("_mask")
    ]
    array_keys = [
        "face_centroid_id",
        "face_distance_sum_voxels",
        "face_nearest_max_distance_voxels",
        "face_area",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        for key in sorted(scalar_keys):
            writer.writerow([key, stats[key]])
        writer.writerow([])
        writer.writerow(["face_id", "v0", "v1", "v2", *array_keys])
        for face_id, face in enumerate(faces):
            row = [face_id, int(face[0]), int(face[1]), int(face[2])]
            for key in array_keys:
                row.append(stats[key][face_id])
            writer.writerow(row)


def print_summary(stats: dict[str, np.ndarray | int | float | str]) -> None:
    print(f"Algorithm: {ALGO_NAME}")
    print(f"Resolution: {stats['resolution']}")
    print(f"Vertices: {stats['vertices']}")
    print(f"Centroids: {stats['centroids']}")
    print(f"Faces: {stats['faces']}")
    print(f"Edges: {stats['edges']}")
    print("Vertex extraction:")
    print(
        f"  d_vert >= {stats['vertex_dvert_threshold']} "
        f"(raw max barycentric >= {float(stats['vertex_raw_max_barycentric']):.6f})"
    )
    print(f"  connectivity = {stats['vertex_connectivity']}")
    print(f"  position mode = {stats['vertex_position_mode']}")
    print(f"  core mode = {stats['vertex_core_mode']}")
    print(f"  blob voxels = {stats['vertex_blob_voxels']}")
    print(f"  core voxels = {stats['vertex_core_voxels']}")
    print("Centroid extraction:")
    print(
        f"  d_tri >= {stats['centroid_dtri_threshold']} "
        f"(raw min barycentric >= {float(stats['centroid_raw_min_barycentric']):.6f})"
    )
    print(
        f"  d_vert <= {stats['centroid_max_dvert']} "
        f"(raw max barycentric <= {float(stats['centroid_raw_max_barycentric']):.6f})"
    )
    print(f"  connectivity = {stats['centroid_connectivity']}")
    print(f"  min component size = {stats['centroid_min_component_size']}")
    print(f"  target count = {stats['centroid_target_count']} (0 disables target fill)")
    print(f"  fill small components by = {stats['centroid_fill_small_components_by']}")
    print(f"  components before target fill = {stats['centroid_components_before_target']}")
    print(f"  small components available = {stats['centroid_small_components_available']}")
    print(f"  target-filled small components = {stats['centroid_target_fill_count']}")
    print(f"  position mode = {stats['centroid_position_mode']}")
    print(f"  core mode = {stats['centroid_core_mode']}")
    print(f"  blob voxels = {stats['centroid_blob_voxels']}")
    print(f"  core voxels = {stats['centroid_core_voxels']}")
    print("Face construction:")
    print("  each centroid connects to its 3 nearest vertices")
    print(f"  max vertex distance = {stats['face_max_vertex_distance_voxels']} voxel(s); 0 disables this gate")
    print(f"  rejected distance = {stats['rejected_distance']}")
    print(f"  rejected degenerate = {stats['rejected_degenerate']}")
    print(f"  duplicate faces collapsed = {stats['duplicate_faces']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO_NAME}: centroid-to-three-nearest-vertices faces.")
    parser.add_argument("path", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--resolution", type=int, default=None)

    parser.add_argument("--vertex_dvert_threshold", type=float, default=0.84)
    parser.add_argument("--vertex_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--vertex_min_component_size", type=int, default=1)
    parser.add_argument("--vertex_position_mode", choices=["peak", "mean", "weighted"], default="weighted")
    parser.add_argument(
        "--vertex_core_mode",
        choices=["closest4", "peak", "position_nearest", "blob", "none"],
        default="closest4",
    )

    parser.add_argument("--centroid_dtri_threshold", type=float, default=0.60)
    parser.add_argument("--centroid_max_dvert", type=float, default=0.55)
    parser.add_argument("--centroid_connectivity", type=int, choices=[6, 18, 26], default=18)
    parser.add_argument("--centroid_min_component_size", type=int, default=4)
    parser.add_argument("--centroid_target_count", type=int, default=0)
    parser.add_argument(
        "--centroid_fill_small_components_by",
        choices=["none", "peak_dtri", "size_then_peak_dtri"],
        default="none",
    )
    parser.add_argument("--centroid_position_mode", choices=["peak", "mean", "weighted"], default="weighted")
    parser.add_argument(
        "--centroid_core_mode",
        choices=["closest4", "peak", "position_nearest", "blob", "none"],
        default="closest4",
    )

    parser.add_argument("--face_max_vertex_distance_voxels", type=float, default=0.0)
    parser.add_argument("--min_face_area", type=float, default=1e-12)
    parser.add_argument("--no_orient_outward", action="store_true")

    parser.add_argument("--out_obj", type=str, default=None)
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--out_npz", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_core_ply", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)

    vertices, edges, faces, stats = extract_mesh(coords, features, resolution, args)

    stem = default_output_stem(path)
    out_dir = default_results_dir(path)
    out_obj = Path(args.out_obj) if args.out_obj else out_dir / f"{stem}_centroid_nearest_mesh.obj"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_centroid_nearest_mesh.ply"
    out_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{stem}_centroid_nearest_mesh.npz"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"{stem}_centroid_nearest_faces.csv"
    out_core_ply = (
        Path(args.out_core_ply)
        if args.out_core_ply
        else out_dir / f"{stem}_centroid_nearest_cores_colored.ply"
    )

    write_obj(out_obj, vertices, faces)
    write_mesh_ply(out_ply, vertices, edges, faces)
    write_npz(out_npz, vertices, edges, faces, stats)
    write_csv(out_csv, faces, stats)
    write_core_voxel_ply(
        out_core_ply,
        coords,
        resolution,
        np.asarray(stats["vertex_core_mask"], dtype=bool),
        np.asarray(stats["centroid_core_mask"], dtype=bool),
    )

    print_summary(stats)
    print(f"Saved {out_obj}")
    print(f"Saved {out_ply}")
    print(f"Saved {out_npz}")
    print(f"Saved {out_csv}")
    print(f"Saved {out_core_ply}")


if __name__ == "__main__":
    main()
