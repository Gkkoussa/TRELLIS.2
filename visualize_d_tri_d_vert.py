"""
One-file triangle-field voxel rasterizer.

Loads a .npz / .npz.zst with keys:
  coords   (N, 3) int
  features (N, C) float   # channel 0 = d_tri, channel 1 = d_vert

Rasters colored voxels with o_voxel (same base as TRELLIS.2 viz) and writes PNGs.

Deps: torch, numpy, o_voxel, utils3d, Pillow, zstandard (for .zst)
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np
import torch
import utils3d
from PIL import Image

import o_voxel


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


def to_01(values: torch.Tensor, distance_transform: str = "none") -> torch.Tensor:
    values = values.float().reshape(-1)
    if distance_transform == "minus_one_one":
        values = values * 0.5 + 0.5
    elif distance_transform != "none":
        raise ValueError(f"Unsupported distance_transform: {distance_transform}")
    return values.clamp(0.0, 1.0)


def scalar_to_color(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1, 1).expand(-1, 3).contiguous()


def infer_resolution(coords: np.ndarray) -> int:
    return int(coords.max()) + 1


def orbit_cameras(
    nviews: int = 4,
    yaw_offset: float = -16 / 180 * math.pi,
    pitch: float = 20 / 180 * math.pi,
    r: float = 2.0,
    fov_deg: float = 30.0,
    device: torch.device = torch.device("cuda"),
):
    """Same orbit setup as trellis2.utils.render_utils.snapshot_orbit_cameras."""
    yaws = np.linspace(0.0, 2.0 * math.pi, nviews, endpoint=False) + yaw_offset
    extrinsics, intrinsics = [], []
    fov = torch.deg2rad(torch.tensor(float(fov_deg), device=device))
    look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    for yaw in yaws:
        eye = torch.tensor(
            [
                math.sin(yaw) * math.cos(pitch),
                math.cos(yaw) * math.cos(pitch),
                math.sin(pitch),
            ],
            device=device,
            dtype=torch.float32,
        ) * r
        # Prefer utils3d.torch if present (TRELLIS), else top-level utils3d (o-voxel examples).
        if hasattr(utils3d, "torch"):
            extr = utils3d.torch.extrinsics_look_at(eye, look_at, up)
            intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        else:
            extr = utils3d.extrinsics_look_at(eye=eye, look_at=look_at, up=up)
            intr = utils3d.intrinsics_from_fov_xy(fov_x=fov, fov_y=fov)
        extrinsics.append(extr.to(device=device, dtype=torch.float32))
        intrinsics.append(intr.to(device=device, dtype=torch.float32))
    return extrinsics, intrinsics


def voxel_positions(
    coords: torch.Tensor,
    resolution: int,
    origin=(-0.5, -0.5, -0.5),
) -> tuple[torch.Tensor, float]:
    """World-space voxel centers, matching trellis2.representations.Voxel.position."""
    origin_t = torch.tensor(origin, dtype=torch.float32, device=coords.device)
    voxel_size = 1.0 / resolution
    position = (coords.float() + 0.5) * voxel_size + origin_t[None, :]
    return position, voxel_size


@torch.no_grad()
def render_channel(
    renderer,
    position: torch.Tensor,
    colors: torch.Tensor,
    voxel_size: float,
    extrinsics,
    intrinsics,
    render_resolution: int,
) -> np.ndarray:
    """Render 4 orbit views into a 2x2 RGB uint8 image."""
    canvas = np.zeros((render_resolution * 2, render_resolution * 2, 3), dtype=np.uint8)
    tile = [2, 2]
    for j, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        out = renderer.render(
            position=position,
            attrs=colors,
            voxel_size=voxel_size,
            extrinsics=ext,
            intrinsics=intr,
        )
        # o_voxel may return EasyDict-like (.attr) or dict ['attr']
        attr = out.attr if hasattr(out, "attr") else out["attr"]
        panel = (attr.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(
            np.uint8
        )
        r0 = render_resolution * (j // tile[1])
        c0 = render_resolution * (j % tile[1])
        canvas[r0 : r0 + render_resolution, c0 : c0 + render_resolution] = panel
    return canvas


@torch.no_grad()
def rasterize_d_tri_d_vert(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    resolution: int,
    *,
    render_resolution: int = 512,
    ssaa: int = 4,
    distance_transform: str = "none",
    out_dir: str | Path | None = None,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """
    Rasterize d_tri / d_vert sparse voxels to PNG-ready images.

    Returns:
        {"d_tri": (2R, 2R, 3) uint8, "d_vert": (2R, 2R, 3) uint8}
    """
    if not torch.cuda.is_available():
        raise RuntimeError("o_voxel rasterization requires CUDA")

    device = torch.device("cuda")
    coords_t = torch.from_numpy(np.asarray(coords, dtype=np.int32)).to(device)
    position, voxel_size = voxel_positions(coords_t, resolution)

    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": render_resolution, "ssaa": ssaa}
    )
    extrinsics, intrinsics = orbit_cameras(device=device)

    images = {}
    for name, values in (("d_tri", d_tri), ("d_vert", d_vert)):
        values_t = to_01(torch.as_tensor(values, device=device), distance_transform)
        colors = scalar_to_color(values_t).float()
        images[name] = render_channel(
            renderer,
            position,
            colors,
            voxel_size,
            extrinsics,
            intrinsics,
            render_resolution,
        )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, image in images.items():
            path = out_dir / f"{prefix}{name}.png"
            Image.fromarray(image).save(path)
            print(f"Saved {path}")

    return images


def rasterize_file(
    path: str | Path,
    *,
    resolution: int | None = None,
    out_dir: str | Path | None = None,
    render_resolution: int = 512,
    ssaa: int = 4,
    distance_transform: str = "none",
) -> dict[str, np.ndarray]:
    path = Path(path)
    coords, features = load_triangle_field(path)
    if resolution is None:
        resolution = infer_resolution(coords)
        print(f"Inferred resolution={resolution} from coords.max()+1")

    if out_dir is None:
        name = path.name
        if name.endswith(".npz.zst"):
            stem = name[: -len(".npz.zst")]
        elif name.endswith(".npz"):
            stem = name[: -len(".npz")]
        else:
            stem = path.stem
        out_dir = path.parent / f"{stem}_raster"

    return rasterize_d_tri_d_vert(
        coords,
        features[:, 0],
        features[:, 1],
        resolution=resolution,
        render_resolution=render_resolution,
        ssaa=ssaa,
        distance_transform=distance_transform,
        out_dir=out_dir,
        prefix="",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize triangle-field .npz/.npz.zst d_tri/d_vert to PNG via o_voxel."
    )
    parser.add_argument("path", type=str, help="Path to .npz or .npz.zst")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Voxel grid resolution (default: coords.max()+1)",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for d_tri.png / d_vert.png")
    parser.add_argument("--render_resolution", type=int, default=512, help="Per-view render size")
    parser.add_argument("--ssaa", type=int, default=4, help="Supersampling factor")
    parser.add_argument(
        "--distance_transform",
        type=str,
        default="none",
        choices=["none", "minus_one_one"],
    )
    args = parser.parse_args()

    rasterize_file(
        args.path,
        resolution=args.resolution,
        out_dir=args.out_dir,
        render_resolution=args.render_resolution,
        ssaa=args.ssaa,
        distance_transform=args.distance_transform,
    )


if __name__ == "__main__":
    main()
