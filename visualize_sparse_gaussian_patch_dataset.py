import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import o_voxel
import utils3d
from trellis2.datasets import SparseGaussianPatchDataset


def build_camera(device: torch.device):
    extrinsics = utils3d.extrinsics_look_at(
        eye=torch.tensor([1.2, 0.5, 1.2]),
        look_at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
    ).to(device)
    intrinsics = utils3d.intrinsics_from_fov_xy(
        fov_x=torch.deg2rad(torch.tensor(45.0)),
        fov_y=torch.deg2rad(torch.tensor(45.0)),
    ).to(device)
    return extrinsics, intrinsics


def render_attr(renderer, position, attrs, voxel_size, extrinsics, intrinsics, resolution):
    if position.shape[0] == 0:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    output = renderer.render(
        position=position,
        attrs=attrs,
        voxel_size=voxel_size,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    image = output.attr.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def grayscale_channel(attr: torch.Tensor, channel_idx: int) -> torch.Tensor:
    if attr.shape[0] == 0:
        return attr
    channel = attr[:, channel_idx:channel_idx + 1]
    return channel.expand(-1, 3)


def sparse_patch_to_attrs(x_0):
    local_coords = x_0.coords[:, 1:].long().cpu()
    attrs = ((x_0.feats.detach().float().cpu() + 1.0) * 0.5).clamp(0, 1)
    edge = attrs[:, 0:3]
    vertex = attrs[:, 3:6] if attrs.shape[1] >= 6 else edge
    return local_coords, edge, vertex


def render_full_and_sparse_patch(dataset, root, instance, item, panel_resolution, ssaa, device):
    full_coords, full_feats = dataset._read_gaussian_voxel(root, instance)
    full_attrs = ((full_feats + 1.0) * 0.5).clamp(0, 1)
    full_edge = full_attrs[:, 0:3]
    full_vertex = full_attrs[:, 3:6] if full_attrs.shape[1] >= 6 else full_edge

    origin = item["patch_origin"].round().long()
    patch_coords, patch_edge, patch_vertex = sparse_patch_to_attrs(item["x_0"])

    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": panel_resolution, "ssaa": ssaa}
    )
    extrinsics, intrinsics = build_camera(device)

    full_position = (full_coords.float() / dataset.resolution - 0.5).to(device)
    patch_position = (patch_coords.float() / dataset.patch_size - 0.5).to(device)
    full_edge = full_edge.to(device)
    full_vertex = full_vertex.to(device)
    patch_edge = patch_edge.to(device)
    patch_vertex = patch_vertex.to(device)

    full_voxel_size = 1.0 / dataset.resolution
    patch_voxel_size = 1.0 / dataset.patch_size

    images = {
        "full_edge": render_attr(renderer, full_position, full_edge, full_voxel_size, extrinsics, intrinsics, panel_resolution),
        "patch_edge": render_attr(renderer, patch_position, patch_edge, patch_voxel_size, extrinsics, intrinsics, panel_resolution),
        "full_vertex": render_attr(renderer, full_position, full_vertex, full_voxel_size, extrinsics, intrinsics, panel_resolution),
        "patch_vertex": render_attr(renderer, patch_position, patch_vertex, patch_voxel_size, extrinsics, intrinsics, panel_resolution),
        "full_edge_ch0": render_attr(renderer, full_position, grayscale_channel(full_edge, 0), full_voxel_size, extrinsics, intrinsics, panel_resolution),
        "patch_edge_ch0": render_attr(renderer, patch_position, grayscale_channel(patch_edge, 0), patch_voxel_size, extrinsics, intrinsics, panel_resolution),
        "full_vertex_ch0": render_attr(renderer, full_position, grayscale_channel(full_vertex, 0), full_voxel_size, extrinsics, intrinsics, panel_resolution),
        "patch_vertex_ch0": render_attr(renderer, patch_position, grayscale_channel(patch_vertex, 0), patch_voxel_size, extrinsics, intrinsics, panel_resolution),
    }

    patch_voxels = int(patch_coords.shape[0])
    return images, {
        "origin": origin.tolist(),
        "full_voxels": int(full_coords.shape[0]),
        "patch_voxels": patch_voxels,
        "patch_occupied_ratio": patch_voxels / float(dataset.patch_size ** 3),
        "local_coord_min": patch_coords.min(dim=0).values.tolist() if patch_voxels else [],
        "local_coord_max": patch_coords.max(dim=0).values.tolist() if patch_voxels else [],
    }


def make_contact_sheet(images: dict, sample_idx: int, instance: str, stats: dict, panel_resolution: int):
    panels = [
        ("full_edge", "Full edge RGB"),
        ("patch_edge", "Sparse patch edge RGB"),
        ("full_vertex", "Full vertex RGB"),
        ("patch_vertex", "Sparse patch vertex RGB"),
        ("full_edge_ch0", "Full edge ch0"),
        ("patch_edge_ch0", "Sparse patch edge ch0"),
        ("full_vertex_ch0", "Full vertex ch0"),
        ("patch_vertex_ch0", "Sparse patch vertex ch0"),
    ]
    cols = 2
    rows = 4
    label_h = 24
    header_h = 52
    canvas = Image.new(
        "RGB",
        (cols * panel_resolution, header_h + rows * (panel_resolution + label_h)),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 8),
        (
            f"sample={sample_idx} sha={instance} origin={stats['origin']} "
            f"patch_voxels={stats['patch_voxels']} ratio={stats['patch_occupied_ratio']:.5f}"
        ),
        fill=(255, 255, 255),
        font=font,
    )
    draw.text(
        (8, 26),
        (
            f"local_min={stats['local_coord_min']} local_max={stats['local_coord_max']} "
            f"full_voxels={stats['full_voxels']}"
        ),
        fill=(255, 255, 255),
        font=font,
    )

    for idx, (key, label) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x0 = col * panel_resolution
        y0 = header_h + row * (panel_resolution + label_h)
        canvas.paste(Image.fromarray(images[key], mode="RGB"), (x0, y0))
        draw.rectangle(
            [x0, y0 + panel_resolution, x0 + panel_resolution, y0 + panel_resolution + label_h],
            fill=(20, 20, 20),
        )
        draw.text((x0 + 8, y0 + panel_resolution + 6), label, fill=(255, 255, 255), font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Same JSON data_dir string used by train.py")
    parser.add_argument("--out_dir", default="sparse_gaussian_patch_dataset_vis")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--min_aesthetic_score", type=float, default=4.5)
    parser.add_argument("--max_active_voxels", type=int, default=32768)
    parser.add_argument("--min_patch_active_voxels", type=int, default=1)
    parser.add_argument("--max_resample_attempts", type=int, default=16)
    parser.add_argument("--foreground_patch_prob", type=float, default=1.0)
    parser.add_argument("--background_value", type=float, default=-1.0)
    parser.add_argument("--num_read_threads", type=int, default=4)
    parser.add_argument("--render_resolution", type=int, default=256)
    parser.add_argument("--ssaa", type=int, default=2)
    parser.add_argument("--zero_cond", action="store_true", help="Return zero model conditioning, matching the sparse DiT config.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    json.loads(args.data_dir)

    dataset = SparseGaussianPatchDataset(
        args.data_dir,
        resolution=args.resolution,
        patch_size=args.patch_size,
        min_aesthetic_score=args.min_aesthetic_score,
        max_active_voxels=args.max_active_voxels,
        min_patch_active_voxels=args.min_patch_active_voxels,
        max_resample_attempts=args.max_resample_attempts,
        attrs=["base_color", "emissive"],
        voxel_root_key="gaussian_distance_voxel",
        voxelized_flag_column="gaussian_distance_voxelized",
        num_voxels_column="num_gaussian_distance_voxels",
        foreground_patch_prob=args.foreground_patch_prob,
        background_value=args.background_value,
        num_read_threads=args.num_read_threads,
        zero_cond=args.zero_cond,
        return_origin=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dataset)
    print(f"Using device: {device}")
    print(f"Saving {args.num_samples} visualizations to {args.out_dir}")

    summary = []
    for sample_idx in range(args.num_samples):
        dataset_idx = sample_idx % len(dataset)
        root, instance = dataset.instances[dataset_idx]
        item = dataset[dataset_idx]
        images, stats = render_full_and_sparse_patch(
            dataset,
            root,
            instance,
            item,
            args.render_resolution,
            args.ssaa,
            device,
        )
        image = make_contact_sheet(images, sample_idx, instance, stats, args.render_resolution)
        path = os.path.join(args.out_dir, f"sample_{sample_idx:04d}.png")
        image.save(path)

        summary.append({
            "sample": sample_idx,
            "sha256": instance,
            "path": path,
            **stats,
        })

    with open(os.path.join(args.out_dir, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main()
