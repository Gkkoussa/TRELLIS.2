#!/usr/bin/env python3
"""Save triangle-field voxels with low GT d_tri as a PLY point cloud."""

import argparse
import io
from pathlib import Path

import numpy as np


def load_npz(path: Path):
    if path.name.endswith(".npz.zst"):
        import zstandard as zstd

        payload = zstd.ZstdDecompressor().decompress(path.read_bytes())
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def find_input(root: Path, resolution: int, instance: str) -> Path:
    directory = root / f"triangle_field_voxels_{resolution}"
    for suffix in (".npz.zst", ".npz"):
        path = directory / f"{instance}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No triangle-field file for {instance} in {directory}")


def save_ply(path: Path, points: np.ndarray, d_tri: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float d_tri\n")
        handle.write("end_header\n")
        for point, value in zip(points, d_tri):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {value:.8f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="Mesh SHA256 / dataset instance name")
    parser.add_argument("--root", type=Path, required=True, help="Dataset ROOT")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.175)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_path = find_input(args.root, args.resolution, args.instance)
    with load_npz(input_path) as data:
        coords = np.asarray(data["coords"], dtype=np.int32)
        d_tri = np.asarray(data["features"][:, 0], dtype=np.float32)

    selected = d_tri < args.threshold
    selected_coords = coords[selected]
    selected_d_tri = d_tri[selected]
    points = (selected_coords.astype(np.float32) + 0.5) / args.resolution - 0.5

    output = args.output or Path("outputs") / (
        f"{args.instance}_r{args.resolution}_dtri_lt_{args.threshold:g}.ply"
    )
    save_ply(output, points, selected_d_tri)

    print(f"Input: {input_path}")
    print(f"Selected: {len(points)} / {len(coords)} voxels (d_tri < {args.threshold})")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
