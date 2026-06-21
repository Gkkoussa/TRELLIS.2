import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from trellis2.datasets.sparse_voxel_triangle_field import find_triangle_field_path, load_triangle_field_npz
from trellis2.modules import sparse as sp
from trellis2.renderers import VoxelRenderer
from trellis2.representations import Voxel
from trellis2.utils.render_utils import snapshot_orbit_cameras


CHANNELS = {"d_tri": 0, "d_vert": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 512 triangle-field voxels and parent-bin averaged "
            "downsampled fields at multiple lower resolutions."
        )
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--source_resolution", type=int, default=512)
    parser.add_argument("--resolutions", type=str, default="512,256,128,64,32")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--render_resolution", type=int, default=384)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--d_tri_colormap", choices=("magma", "gray"), default="magma")
    parser.add_argument("--d_vert_colormap", choices=("magma", "gray"), default="gray")
    return parser.parse_args()


def load_instances(root: Path, split: str, instances: str | None, random: bool, seed: int) -> list[str]:
    if instances is None:
        candidates = [
            root / "splits" / f"{split}_triangle_field_512" / "instances.txt",
            root / "splits" / split / "instances.txt",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No instances.txt found for split {split} under {root / 'splits'}")
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif "," in instances:
        values = [item.strip() for item in instances.split(",") if item.strip()]
    else:
        path = Path(instances)
        if path.exists():
            values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        else:
            values = [instances.strip()]
    if random:
        rng = np.random.default_rng(seed)
        values = list(rng.permutation(values))
    return values


def load_raw_triangle_field(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with load_triangle_field_npz(str(path)) as data:
        coords = torch.from_numpy(data["coords"].astype(np.int32, copy=False))
        features = torch.from_numpy(data["features"].astype(np.float32, copy=False))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} has invalid coords shape {tuple(coords.shape)}")
    if features.ndim != 2 or features.shape[0] != coords.shape[0]:
        raise ValueError(f"{path} has invalid features shape {tuple(features.shape)}")
    return coords, features


def average_downsample(coords: torch.Tensor, features: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    if factor == 1:
        return coords, features
    parent = torch.div(coords, factor, rounding_mode="floor").int()
    unique_parent, inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    sums = torch.zeros(
        (unique_parent.shape[0], features.shape[1]),
        dtype=torch.float32,
        device=features.device,
    )
    counts = torch.zeros((unique_parent.shape[0], 1), dtype=torch.float32, device=features.device)
    sums.index_add_(0, inverse, features.float())
    counts.index_add_(0, inverse, torch.ones((features.shape[0], 1), dtype=torch.float32, device=features.device))
    return unique_parent, sums / counts.clamp_min(1.0)


def scalar_to_gray(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1, 1).float().clamp(0, 1)
    return values.expand(-1, 3)


def scalar_to_magma(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1).float().clamp(0, 1)
    stops = torch.tensor(
        [
            [0.001, 0.000, 0.014],
            [0.251, 0.066, 0.430],
            [0.478, 0.125, 0.514],
            [0.741, 0.214, 0.329],
            [0.944, 0.498, 0.145],
            [0.987, 0.991, 0.749],
        ],
        dtype=torch.float32,
        device=values.device,
    )
    scaled = values * (stops.shape[0] - 1)
    idx0 = torch.floor(scaled).long().clamp(0, stops.shape[0] - 1)
    idx1 = (idx0 + 1).clamp(0, stops.shape[0] - 1)
    t = (scaled - idx0.float()).reshape(-1, 1)
    return stops[idx0] * (1.0 - t) + stops[idx1] * t


def colorize(values: torch.Tensor, colormap: str) -> torch.Tensor:
    if colormap == "gray":
        return scalar_to_gray(values)
    if colormap == "magma":
        return scalar_to_magma(values)
    raise ValueError(f"Unsupported colormap: {colormap}")


@torch.no_grad()
def render_channel(
    coords: torch.Tensor,
    features: torch.Tensor,
    resolution: int,
    channel: int,
    colormap: str,
    render_resolution: int,
    ssaa: int,
) -> Image.Image:
    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = ssaa
    exts, ints = snapshot_orbit_cameras()

    sparse_coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1).int().cuda()
    sparse = sp.SparseTensor(features.float().cuda(), sparse_coords)
    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / resolution,
        coords=sparse.coords[:, 1:].contiguous(),
        attrs=None,
        layout={"color": slice(0, 3)},
    )
    attr = colorize(sparse.feats[:, channel], colormap).float()

    image = torch.zeros(3, render_resolution * 2, render_resolution * 2, dtype=torch.float32, device="cuda")
    for view_idx, (ext, intr) in enumerate(zip(exts, ints)):
        with torch.autocast(device_type="cuda", enabled=False):
            out = renderer.render(rep, ext.float(), intr.float(), colors_overwrite=attr)
        row = view_idx // 2
        col = view_idx % 2
        image[
            :,
            render_resolution * row:render_resolution * (row + 1),
            render_resolution * col:render_resolution * (col + 1),
        ] = out["color"].float()

    arr = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_side_by_side(images: list[Image.Image], labels: list[str], title: str, out_path: Path) -> None:
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    label_h = 42
    title_h = 34
    pad = 8
    canvas = Image.new("RGB", (sum(widths) + pad * (len(images) + 1), max(heights) + label_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(0, 0, 0))
    x = pad
    for image, label in zip(images, labels):
        draw.text((x, title_h + 10), label, fill=(0, 0, 0))
        canvas.paste(image, (x, title_h + label_h))
        x += image.width + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [int(item) for item in args.resolutions.split(",") if item.strip()]
    if resolutions[0] != args.source_resolution:
        raise ValueError("--resolutions should start with --source_resolution for this comparison.")
    for resolution in resolutions:
        if args.source_resolution % resolution != 0:
            raise ValueError(f"Source resolution {args.source_resolution} is not divisible by {resolution}.")

    source_root = root / "splits" / args.split / f"triangle_field_voxels_{args.source_resolution}"
    instances = load_instances(root, args.split, args.instances, args.random, args.seed)
    kept = []
    stats = []

    for instance in tqdm(instances, desc="Rendering downsampled fields"):
        try:
            path = Path(find_triangle_field_path(str(source_root), instance))
        except FileNotFoundError:
            continue
        coords, features = load_raw_triangle_field(path)
        per_resolution = {}
        for resolution in resolutions:
            factor = args.source_resolution // resolution
            ds_coords, ds_features = average_downsample(coords, features, factor)
            per_resolution[resolution] = (ds_coords, ds_features)

        for channel_name, channel_idx in CHANNELS.items():
            colormap = args.d_tri_colormap if channel_name == "d_tri" else args.d_vert_colormap
            rendered = [
                render_channel(
                    per_resolution[resolution][0],
                    per_resolution[resolution][1],
                    resolution,
                    channel_idx,
                    colormap,
                    args.render_resolution,
                    args.ssaa,
                )
                for resolution in resolutions
            ]
            labels = [
                f"{resolution} ({per_resolution[resolution][0].shape[0]:,} vox)"
                for resolution in resolutions
            ]
            make_side_by_side(
                rendered,
                labels,
                f"{instance} {channel_name} averaged from {args.source_resolution}",
                output_dir / f"{instance}_{channel_name}_downsampled_fields.jpg",
            )

        kept.append(instance)
        stats.append(
            {
                "instance": instance,
                "voxels": {str(resolution): int(per_resolution[resolution][0].shape[0]) for resolution in resolutions},
            }
        )
        if len(kept) >= args.num_samples:
            break

    if not kept:
        raise RuntimeError(f"No {args.source_resolution} triangle-field voxels found under {source_root}.")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "split": args.split,
                "source_resolution": args.source_resolution,
                "resolutions": resolutions,
                "instances": kept,
                "stats": stats,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(kept)} sample visualizations to {output_dir}")


if __name__ == "__main__":
    main()
