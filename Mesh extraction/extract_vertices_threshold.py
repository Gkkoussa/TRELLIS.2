"""
Threshold-based vertex extraction for triangle-field voxels.

The triangle-field features are expected to contain:
  coords   (N, 3) int
  features (N, C) float, with features[:, 0] = d_tri and features[:, 1] = d_vert

`d_vert` is the normalized max barycentric coordinate:
  d_vert = (max(a, b, c) - 1/3) / (2/3)

It is high near triangle vertices. This script keeps voxels whose d_vert is
above a threshold, optionally keeps only local maxima, merges neighboring
selected voxels into connected components, and writes one extracted vertex per
component. It deliberately uses only coords and d_vert.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import deque
from pathlib import Path

import numpy as np


def load_triangle_field(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.name.endswith(".npz.zst"):
        import zstandard as zstd

        with open(path, "rb") as f:
            payload = zstd.ZstdDecompressor().decompress(f.read())
        data = np.load(io.BytesIO(payload), allow_pickle=False)
    else:
        data = np.load(path, allow_pickle=False)

    coords = np.asarray(data["coords"])
    features = np.asarray(data["features"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    if features.ndim != 2 or features.shape[1] < 2:
        raise ValueError(f"features must be (N, >=2), got {features.shape}")
    if coords.shape[0] != features.shape[0]:
        raise ValueError(
            f"coords/features length mismatch: {coords.shape[0]} vs {features.shape[0]}"
        )
    return coords.astype(np.int32, copy=False), features.astype(np.float32, copy=False)


def infer_resolution(coords: np.ndarray) -> int:
    return int(coords.max()) + 1


def voxel_centers(
    coords: np.ndarray,
    resolution: int,
    origin: tuple[float, float, float] = (-0.5, -0.5, -0.5),
) -> np.ndarray:
    return (coords.astype(np.float32) + 0.5) / float(resolution) + np.asarray(
        origin, dtype=np.float32
    )


def local_maximum_mask(
    coords: np.ndarray,
    values: np.ndarray,
    candidate_mask: np.ndarray,
    neighborhood: int = 1,
    eps: float = 1e-8,
) -> np.ndarray:
    """Keep thresholded voxels that are not lower than any sparse neighbor."""
    value_by_coord = {tuple(c.tolist()): float(v) for c, v in zip(coords, values)}
    keep = np.zeros(candidate_mask.shape, dtype=bool)
    selected = np.flatnonzero(candidate_mask)

    offsets = []
    for dx in range(-neighborhood, neighborhood + 1):
        for dy in range(-neighborhood, neighborhood + 1):
            for dz in range(-neighborhood, neighborhood + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))

    for idx in selected:
        x, y, z = coords[idx].tolist()
        center_value = float(values[idx])
        is_max = True
        for dx, dy, dz in offsets:
            neighbor_value = value_by_coord.get((x + dx, y + dy, z + dz))
            if neighbor_value is not None and neighbor_value > center_value + eps:
                is_max = False
                break
        keep[idx] = is_max
    return keep


def connected_components(
    coords: np.ndarray,
    selected_mask: np.ndarray,
    connectivity: int = 26,
) -> list[np.ndarray]:
    if connectivity not in (6, 18, 26):
        raise ValueError(f"connectivity must be 6, 18, or 26, got {connectivity}")

    selected = np.flatnonzero(selected_mask)
    coord_to_index = {tuple(coords[i].tolist()): int(i) for i in selected}
    visited: set[int] = set()
    components: list[np.ndarray] = []

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))

    for start in selected:
        start = int(start)
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component = []
        while queue:
            idx = queue.popleft()
            component.append(idx)
            x, y, z = coords[idx].tolist()
            for dx, dy, dz in offsets:
                neighbor = coord_to_index.get((x + dx, y + dy, z + dz))
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def extract_vertices_from_d_vert(
    coords: np.ndarray,
    d_vert: np.ndarray,
    resolution: int | None = None,
    threshold: float = 0.95,
    min_component_size: int = 1,
    connectivity: int = 26,
    position_mode: str = "weighted",
    local_maxima_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one world-space vertex per high-d_vert component.

    Returns:
        vertices: (M, 3) float32 world-space positions in TRELLIS [-0.5, 0.5] coordinates
        vertex_values: (M,) float32 peak d_vert value per component
        component_sizes: (M,) int32 selected voxel count per component
    """
    if resolution is None:
        resolution = infer_resolution(coords)
    if position_mode not in ("peak", "mean", "weighted"):
        raise ValueError(f"Unsupported position_mode: {position_mode}")

    d_vert = np.asarray(d_vert, dtype=np.float32).reshape(-1)
    selected_mask = d_vert >= float(threshold)
    if local_maxima_only:
        selected_mask = local_maximum_mask(coords, d_vert, selected_mask)

    centers = voxel_centers(coords, resolution)
    vertices = []
    values = []
    sizes = []

    for component in connected_components(coords, selected_mask, connectivity):
        if component.size < min_component_size:
            continue
        component_values = d_vert[component]
        peak_local = int(np.argmax(component_values))
        peak_idx = int(component[peak_local])

        if position_mode == "peak":
            vertex = centers[peak_idx]
        elif position_mode == "mean":
            vertex = centers[component].mean(axis=0)
        else:
            weights = component_values - float(threshold)
            weights = np.maximum(weights, 0.0) + 1e-6
            vertex = np.average(centers[component], axis=0, weights=weights)

        vertices.append(vertex.astype(np.float32, copy=False))
        values.append(float(d_vert[peak_idx]))
        sizes.append(int(component.size))

    if not vertices:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.stack(vertices).astype(np.float32, copy=False),
        np.asarray(values, dtype=np.float32),
        np.asarray(sizes, dtype=np.int32),
    )


def threshold_to_raw_max_barycentric(threshold: float) -> float:
    return (1.0 / 3.0) + float(threshold) * (2.0 / 3.0)


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    if not thresholds:
        raise ValueError("No thresholds provided")
    return thresholds


def sweep_thresholds(
    coords: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    thresholds: list[float],
    min_component_size: int = 1,
    connectivity: int = 26,
    position_mode: str = "weighted",
    local_maxima_only: bool = False,
) -> list[dict[str, float | int]]:
    rows = []
    d_vert = np.asarray(d_vert, dtype=np.float32).reshape(-1)
    for threshold in thresholds:
        selected_mask = d_vert >= float(threshold)
        selected_voxels = int(selected_mask.sum())
        vertices, values, sizes = extract_vertices_from_d_vert(
            coords,
            d_vert,
            resolution=resolution,
            threshold=threshold,
            min_component_size=min_component_size,
            connectivity=connectivity,
            position_mode=position_mode,
            local_maxima_only=local_maxima_only,
        )
        rows.append({
            "threshold": float(threshold),
            "raw_max_barycentric": threshold_to_raw_max_barycentric(threshold),
            "selected_voxels": selected_voxels,
            "candidate_vertices": int(vertices.shape[0]),
            "mean_component_size": float(sizes.mean()) if sizes.size else 0.0,
            "max_component_size": int(sizes.max()) if sizes.size else 0,
            "min_component_size": int(sizes.min()) if sizes.size else 0,
            "mean_peak_d_vert": float(values.mean()) if values.size else 0.0,
        })
    return rows


def write_sweep_csv(path: str | Path, rows: list[dict[str, float | int]]) -> None:
    path = Path(path)
    fieldnames = [
        "threshold",
        "raw_max_barycentric",
        "selected_voxels",
        "candidate_vertices",
        "mean_component_size",
        "max_component_size",
        "min_component_size",
        "mean_peak_d_vert",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ply_points(path: str | Path, vertices: np.ndarray, values: np.ndarray, sizes: np.ndarray) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float d_vert\n")
        f.write("property int component_size\n")
        f.write("end_header\n")
        for vertex, value, size in zip(vertices, values, sizes):
            f.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{float(value):.8f} {int(size)}\n"
            )


def default_output_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".npz.zst"):
        return name[: -len(".npz.zst")]
    if name.endswith(".npz"):
        return name[: -len(".npz")]
    return path.stem


def default_results_dir(path: Path) -> Path:
    out_dir = path.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract thresholded triangle vertices from normalized d_vert."
    )
    parser.add_argument("path", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--threshold", type=float, default=0.95, help="d_vert cutoff")
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated d_vert cutoffs to sweep instead of extracting one cutoff",
    )
    parser.add_argument("--resolution", type=int, default=None, help="Voxel grid resolution")
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=1,
        help="Drop high-d_vert connected components smaller than this many voxels",
    )
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    parser.add_argument(
        "--position_mode",
        type=str,
        default="weighted",
        choices=["peak", "mean", "weighted"],
        help="How to place each extracted vertex inside a high-d_vert component",
    )
    parser.add_argument(
        "--local_maxima_only",
        action="store_true",
        help="Threshold first, then keep only sparse 3x3x3 local maxima before components",
    )
    parser.add_argument("--out_npz", type=str, default=None, help="Output vertex .npz")
    parser.add_argument("--out_ply", type=str, default=None, help="Output point-cloud .ply")
    parser.add_argument("--sweep_out_csv", type=str, default=None, help="Output CSV for --thresholds")
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)
    stem = default_output_stem(path)

    if args.thresholds is not None:
        thresholds = parse_thresholds(args.thresholds)
        rows = sweep_thresholds(
            coords,
            features[:, 1],
            resolution=resolution,
            thresholds=thresholds,
            min_component_size=args.min_component_size,
            connectivity=args.connectivity,
            position_mode=args.position_mode,
            local_maxima_only=args.local_maxima_only,
        )
        out_dir = default_results_dir(path)
        out_csv = (
            Path(args.sweep_out_csv)
            if args.sweep_out_csv
            else out_dir / f"{stem}_vertices_threshold_sweep.csv"
        )
        write_sweep_csv(out_csv, rows)
        print(f"Loaded {coords.shape[0]} sparse voxels at resolution={resolution}")
        print(f"Saved sweep CSV {out_csv}")
        print("threshold, raw_max_barycentric, selected_voxels, candidate_vertices")
        for row in rows:
            print(
                f"{row['threshold']:.6f}, "
                f"{row['raw_max_barycentric']:.6f}, "
                f"{row['selected_voxels']}, "
                f"{row['candidate_vertices']}"
            )
        return

    vertices, values, sizes = extract_vertices_from_d_vert(
        coords,
        features[:, 1],
        resolution=resolution,
        threshold=args.threshold,
        min_component_size=args.min_component_size,
        connectivity=args.connectivity,
        position_mode=args.position_mode,
        local_maxima_only=args.local_maxima_only,
    )

    out_dir = default_results_dir(path)
    out_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{stem}_vertices_threshold.npz"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_vertices_threshold.ply"
    np.savez_compressed(
        out_npz,
        vertices=vertices,
        d_vert=values,
        component_size=sizes,
        threshold=np.asarray(args.threshold, dtype=np.float32),
        resolution=np.asarray(resolution, dtype=np.int32),
    )
    write_ply_points(out_ply, vertices, values, sizes)

    print(f"Loaded {coords.shape[0]} sparse voxels at resolution={resolution}")
    print(f"Extracted {vertices.shape[0]} vertices with d_vert >= {args.threshold}")
    print(f"Saved {out_npz}")
    print(f"Saved {out_ply}")


if __name__ == "__main__":
    main()
