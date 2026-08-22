"""
Rotate extracted TRELLIS voxel-frame outputs into the input OBJ coordinate frame.

The extracted meshes are in voxel/world axes:
  extracted = (x, y, z)

For this heart field, comparison against input.obj shows the input mesh frame is:
  input_frame = (x, z, -y)

This script writes rotated copies and leaves originals untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def rotate_xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    rotated = np.empty_like(points)
    rotated[..., 0] = points[..., 0]
    rotated[..., 1] = points[..., 2]
    rotated[..., 2] = -points[..., 1]
    return rotated


def output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def rotate_obj(path: Path, out_path: Path) -> None:
    with open(path, "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith("v "):
                parts = line.split()
                point = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                x, y, z = rotate_xyz(point)
                rest = " ".join(parts[4:])
                if rest:
                    dst.write(f"v {x:.8f} {y:.8f} {z:.8f} {rest}\n")
                else:
                    dst.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
            else:
                dst.write(line)


def rotate_ascii_ply(path: Path, out_path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not a PLY file")
    if not any(line.strip() == "format ascii 1.0" for line in lines[:10]):
        raise ValueError(f"{path} is not an ASCII PLY file")

    vertex_count = None
    header_end = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
        if line.strip() == "end_header":
            header_end = i
            break

    if vertex_count is None or header_end is None:
        raise ValueError(f"Could not parse PLY header for {path}")

    first_vertex = header_end + 1
    last_vertex = first_vertex + vertex_count
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:first_vertex])
        for line in lines[first_vertex:last_vertex]:
            parts = line.split()
            point = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
            x, y, z = rotate_xyz(point)
            rest = " ".join(parts[3:])
            if rest:
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {rest}\n")
            else:
                f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        f.writelines(lines[last_vertex:])


def should_rotate_npz_array(key: str, value: np.ndarray) -> bool:
    if value.ndim < 2 or value.shape[-1] != 3:
        return False
    if not np.issubdtype(value.dtype, np.floating):
        return False
    return key in {
        "vertices",
        "face_seed_positions",
        "centroid_positions",
    } or key.endswith("_positions")


def rotate_npz(path: Path, out_path: Path) -> None:
    data = np.load(path, allow_pickle=False)
    payload = {}
    for key in data.files:
        value = data[key]
        if should_rotate_npz_array(key, value):
            payload[key] = rotate_xyz(value)
        else:
            payload[key] = value
    np.savez_compressed(out_path, **payload)


def rotate_file(path: Path, suffix: str) -> Path:
    out_path = output_path(path, suffix)
    if path.suffix == ".obj":
        rotate_obj(path, out_path)
    elif path.suffix == ".ply":
        rotate_ascii_ply(path, out_path)
    elif path.suffix == ".npz":
        rotate_npz(path, out_path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate extracted outputs from voxel axes to input.obj axes.")
    parser.add_argument("paths", nargs="+", help="OBJ, ASCII PLY, or NPZ files to rotate")
    parser.add_argument("--suffix", default="_input_axes")
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        out_path = rotate_file(path, args.suffix)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
