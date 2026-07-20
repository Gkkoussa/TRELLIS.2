import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from trellis2.datasets.sparse_voxel_triangle_field import find_triangle_field_path, load_triangle_field_npz
from visualize_triangle_field_downsampled_fields import render_channel


CHANNELS = {"d_tri": 0, "d_vert": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render native triangle-field payloads against avg-from-512 payloads for sanity checking."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolutions", type=str, default="32,64,128,256")
    parser.add_argument("--source_resolution", type=int, default=512)
    parser.add_argument("--avg_suffix", type=str, default="avg_from_512")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--render_resolution", type=int, default=320)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--d_tri_colormap", choices=("magma", "gray"), default="magma")
    parser.add_argument("--d_vert_colormap", choices=("magma", "gray"), default="gray")
    return parser.parse_args()


def load_instances(root: Path, instances_path: Path | None, random: bool, seed: int) -> list[str]:
    if instances_path is None:
        instances_path = root / "splits" / "test" / "instances.txt"
    instances = [line.strip() for line in instances_path.read_text().splitlines() if line.strip()]
    if random:
        rng = np.random.default_rng(seed)
        instances = list(rng.permutation(instances))
    return instances


def load_field(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with load_triangle_field_npz(str(path)) as data:
        coords = torch.from_numpy(data["coords"].astype(np.int32, copy=False))
        features = torch.from_numpy(data["features"].astype(np.float32, copy=False))
    return coords, features


def render_or_blank(
    path: Path | None,
    resolution: int,
    channel_idx: int,
    colormap: str,
    render_resolution: int,
    ssaa: int,
) -> tuple[Image.Image, int | None]:
    if path is None or not path.exists():
        return Image.new("RGB", (render_resolution * 2, render_resolution * 2), "white"), None
    coords, features = load_field(path)
    image = render_channel(coords, features, resolution, channel_idx, colormap, render_resolution, ssaa)
    return image, int(coords.shape[0])


def find_optional(root: Path, dirname: str, instance: str) -> Path | None:
    try:
        return Path(find_triangle_field_path(str(root / dirname), instance))
    except FileNotFoundError:
        return None


def make_grid(images: list[Image.Image], labels: list[str], title: str, out_path: Path, columns: int) -> None:
    if len(images) == 0:
        return
    pad = 10
    title_h = 34
    label_h = 34
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * pad, title_h + rows * (label_h + tile_h + pad) + pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(0, 0, 0))
    for idx, (image, label) in enumerate(zip(images, labels)):
        row = idx // columns
        col = idx % columns
        x = pad + col * (tile_w + pad)
        y = title_h + row * (label_h + tile_h + pad)
        draw.text((x, y), label, fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main():
    args = parse_args()
    resolutions = [int(item) for item in args.resolutions.split(",") if item.strip()]
    instances = load_instances(args.root, args.instances, args.random, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    stats = []
    for instance in tqdm(instances, desc="Rendering avg payload sanity checks"):
        if len(kept) >= args.num_samples:
            break
        source_path = find_optional(args.root, f"triangle_field_voxels_{args.source_resolution}", instance)
        if source_path is None:
            continue

        instance_stats = {"instance": instance, "source_resolution": args.source_resolution, "counts": {}}
        for channel_name, channel_idx in CHANNELS.items():
            colormap = args.d_tri_colormap if channel_name == "d_tri" else args.d_vert_colormap
            images = []
            labels = []

            image, count = render_or_blank(
                source_path,
                args.source_resolution,
                channel_idx,
                colormap,
                args.render_resolution,
                args.ssaa,
            )
            images.append(image)
            labels.append(f"native {args.source_resolution} ({count:,} vox)")
            instance_stats["counts"][f"native_{args.source_resolution}"] = count

            for resolution in resolutions:
                native_path = find_optional(args.root, f"triangle_field_voxels_{resolution}", instance)
                avg_path = find_optional(args.root, f"triangle_field_voxels_{resolution}_{args.avg_suffix}", instance)
                for label_prefix, path in (("native", native_path), ("avg512", avg_path)):
                    image, count = render_or_blank(
                        path,
                        resolution,
                        channel_idx,
                        colormap,
                        args.render_resolution,
                        args.ssaa,
                    )
                    images.append(image)
                    labels.append(f"{label_prefix} {resolution} ({'missing' if count is None else f'{count:,} vox'})")
                    instance_stats["counts"][f"{label_prefix}_{resolution}"] = count

            make_grid(
                images,
                labels,
                f"{instance} {channel_name}: native vs avg-from-{args.source_resolution}",
                args.output_dir / f"{instance}_{channel_name}_native_vs_avg_payloads.jpg",
                columns=3,
            )

        kept.append(instance)
        stats.append(instance_stats)

    if not kept:
        raise RuntimeError("No instances with source payloads were found.")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "root": str(args.root),
                "instances": kept,
                "resolutions": resolutions,
                "source_resolution": args.source_resolution,
                "stats": stats,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(kept)} sanity-check visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
