import argparse
import json
import os
from pathlib import Path

from easydict import EasyDict as edict
import numpy as np
import torch
from PIL import Image, ImageDraw

from data_toolkit.encode_triangle_field_latent import (
    build_triangle_field_sparse_tensor,
    require_triangle_field_dataset_args,
    to_cpu_cache,
    trim_decoder_spatial_cache,
)
from trellis2 import models
from trellis2.datasets.sparse_voxel_triangle_field import find_triangle_field_path
from trellis2.modules import sparse as sp
from trellis2.renderers import VoxelRenderer
from trellis2.representations import Voxel
from trellis2.utils.render_utils import snapshot_orbit_cameras


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether 256 triangle-field latents can be transformed onto true 512 "
            "latent/cache support by parent-copying features."
        )
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--vae_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="step0100000")
    parser.add_argument("--source_resolution", type=int, default=256)
    parser.add_argument("--target_resolution", type=int, default=512)
    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument(
        "--strategies",
        type=str,
        default="parent_copy,trilinear,idw,parent_smooth",
        help="Comma-separated feature transforms to test.",
    )
    parser.add_argument("--idw_radius", type=int, default=1)
    parser.add_argument("--idw_power", type=float, default=2.0)
    parser.add_argument("--smooth_radius", type=int, default=1)
    return parser.parse_args()


def load_instances(instances: str, num_samples: int) -> list[str]:
    if "," in instances:
        values = [item.strip() for item in instances.split(",") if item.strip()]
    else:
        path = Path(instances)
        if path.exists():
            values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        else:
            values = [instances.strip()]
    return values[:num_samples]


def load_model(config, key: str, ckpt_path: Path, force_fp32: bool = False):
    model_cfg = config["models"][key]
    model = getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda().eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=False)
    if force_fp32 and hasattr(model, "convert_to_fp32"):
        model.convert_to_fp32()
        if hasattr(model, "dtype"):
            model.dtype = torch.float32
    return model


def to_edict(value):
    if isinstance(value, dict):
        return edict({k: to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_edict(v) for v in value]
    return value


def encode_target_latent(root: Path, split: str, resolution: int, instance: str, dataset_args, encoder):
    voxel_root = root / "splits" / split / f"{dataset_args.voxel_dirname}_{resolution}"
    if not voxel_root.exists():
        voxel_root = root / f"{dataset_args.voxel_dirname}_{resolution}"
    path = find_triangle_field_path(str(voxel_root), instance)
    x = build_triangle_field_sparse_tensor(path, dataset_args).cuda()
    z = encoder(x)
    cache = {
        "scale": z._scale,
        "spatial_cache": to_cpu_cache(trim_decoder_spatial_cache(z._spatial_cache)),
    }
    return z, cache


def attach_cache(z: sp.SparseTensor, cache: dict) -> sp.SparseTensor:
    z._scale = cache_to_device(cache["scale"], z.device)
    z._spatial_cache = cache_to_device(cache["spatial_cache"], z.device)
    return z


def cache_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: cache_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cache_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [cache_to_device(v, device) for v in value]
    return value


def parent_copy_features(source: sp.SparseTensor, target: sp.SparseTensor, scale: int):
    source_coords = source.coords[:, 1:].detach().cpu()
    target_coords = target.coords[:, 1:].detach().cpu()
    lookup = {tuple(coord.tolist()): idx for idx, coord in enumerate(source_coords)}
    source_indices = []
    target_indices = []
    for target_idx, coord in enumerate(target_coords):
        parent = tuple((coord // scale).tolist())
        source_idx = lookup.get(parent)
        if source_idx is not None:
            source_indices.append(source_idx)
            target_indices.append(target_idx)
    if len(target_indices) != target.feats.shape[0]:
        raise ValueError(
            "Parent-copy would change target latent support: "
            f"{len(target_indices)}/{target.feats.shape[0]} target coords have a source parent."
        )
    source_indices = torch.tensor(source_indices, dtype=torch.long, device=source.feats.device)
    out = sp.SparseTensor(source.feats[source_indices], target.coords)
    return out, len(source_indices), target.feats.shape[0]


def trilinear_features(source: sp.SparseTensor, target: sp.SparseTensor, scale: int):
    source_coords = source.coords[:, 1:].detach().cpu()
    target_coords = target.coords[:, 1:].detach().cpu()
    lookup = {tuple(coord.tolist()): idx for idx, coord in enumerate(source_coords)}
    feats = []
    full_hits = 0
    for coord in target_coords:
        pos = coord.float() / float(scale)
        lo = torch.floor(pos).to(torch.int64)
        frac = pos - lo.float()
        weighted = None
        total_weight = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = (int(lo[0] + dx), int(lo[1] + dy), int(lo[2] + dz))
                    idx = lookup.get(corner)
                    if idx is None:
                        continue
                    wx = frac[0].item() if dx else 1.0 - frac[0].item()
                    wy = frac[1].item() if dy else 1.0 - frac[1].item()
                    wz = frac[2].item() if dz else 1.0 - frac[2].item()
                    weight = wx * wy * wz
                    if weight <= 0:
                        continue
                    value = source.feats[idx] * weight
                    weighted = value if weighted is None else weighted + value
                    total_weight += weight
        if weighted is None:
            parent = tuple((coord // scale).tolist())
            idx = lookup.get(parent)
            if idx is None:
                raise ValueError(f"No trilinear or parent source token for target coord {coord.tolist()}.")
            feats.append(source.feats[idx])
        else:
            feats.append(weighted / total_weight)
            if abs(total_weight - 1.0) < 1e-6:
                full_hits += 1
    return sp.SparseTensor(torch.stack(feats, dim=0), target.coords), len(feats), target.feats.shape[0], full_hits


def idw_features(source: sp.SparseTensor, target: sp.SparseTensor, scale: int, radius: int, power: float):
    source_coords = source.coords[:, 1:].detach().cpu()
    target_coords = target.coords[:, 1:].detach().cpu()
    lookup = {tuple(coord.tolist()): idx for idx, coord in enumerate(source_coords)}
    feats = []
    neighbor_counts = []
    for coord in target_coords:
        pos = coord.float() / float(scale)
        center = torch.floor(pos).to(torch.int64)
        weighted = None
        total_weight = 0.0
        count = 0
        for dx in range(-radius, radius + 2):
            for dy in range(-radius, radius + 2):
                for dz in range(-radius, radius + 2):
                    key = (int(center[0] + dx), int(center[1] + dy), int(center[2] + dz))
                    idx = lookup.get(key)
                    if idx is None:
                        continue
                    src = source_coords[idx].float()
                    dist = torch.linalg.norm(pos - src).item()
                    weight = 1.0 / max(dist, 1e-6) ** power
                    value = source.feats[idx] * weight
                    weighted = value if weighted is None else weighted + value
                    total_weight += weight
                    count += 1
        if weighted is None:
            parent = tuple((coord // scale).tolist())
            idx = lookup.get(parent)
            if idx is None:
                raise ValueError(f"No IDW or parent source token for target coord {coord.tolist()}.")
            feats.append(source.feats[idx])
        else:
            feats.append(weighted / total_weight)
        neighbor_counts.append(count)
    out = sp.SparseTensor(torch.stack(feats, dim=0), target.coords)
    return out, len(feats), target.feats.shape[0], float(np.mean(neighbor_counts))


def smooth_target_features(x: sp.SparseTensor, radius: int):
    coords = x.coords[:, 1:].detach().cpu()
    lookup = {tuple(coord.tolist()): idx for idx, coord in enumerate(coords)}
    feats = []
    counts = []
    for coord in coords:
        values = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    idx = lookup.get((int(coord[0] + dx), int(coord[1] + dy), int(coord[2] + dz)))
                    if idx is not None:
                        values.append(x.feats[idx])
        feats.append(torch.stack(values, dim=0).mean(dim=0))
        counts.append(len(values))
    return sp.SparseTensor(torch.stack(feats, dim=0), x.coords), float(np.mean(counts))


def make_experiment_features(source: sp.SparseTensor, target: sp.SparseTensor, scale: int, strategy: str, args):
    if strategy == "parent_copy":
        z, hits, total = parent_copy_features(source, target, scale)
        return z, {"parent_hit_tokens": hits, "target_tokens": total}
    if strategy == "trilinear":
        z, hits, total, full_hits = trilinear_features(source, target, scale)
        return z, {"parent_hit_tokens": hits, "target_tokens": total, "full_trilinear_weight_tokens": full_hits}
    if strategy == "idw":
        z, hits, total, mean_neighbors = idw_features(source, target, scale, args.idw_radius, args.idw_power)
        return z, {"parent_hit_tokens": hits, "target_tokens": total, "mean_idw_neighbors": mean_neighbors}
    if strategy == "parent_smooth":
        z, hits, total = parent_copy_features(source, target, scale)
        z, mean_neighbors = smooth_target_features(z, args.smooth_radius)
        return z, {"parent_hit_tokens": hits, "target_tokens": total, "mean_smooth_neighbors": mean_neighbors}
    raise ValueError(f"Unknown strategy: {strategy}")


def decode(decoder, z: sp.SparseTensor, cache: dict):
    z = attach_cache(z, cache)
    with torch.no_grad():
        return decoder(z)


def scalar_to_color(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1, 1).clamp(0, 1)
    return values.expand(-1, 3)


def render_voxel_channels(voxel, out_prefix: Path, voxel_resolution: int, render_resolution: int, distance_transform: str):
    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = 4
    exts, ints = snapshot_orbit_cameras()
    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / voxel_resolution,
        coords=voxel.coords[:, 1:].contiguous(),
        attrs=None,
        layout={"color": slice(0, 3)},
    )
    feats = voxel.feats
    if distance_transform == "minus_one_one":
        feats = feats * 0.5 + 0.5
    feats = feats.clamp(0, 1)
    fields = {"d_tri": 0, "d_vert": 1}
    paths = {}
    for name, channel in fields.items():
        image = torch.zeros(3, render_resolution * 2, render_resolution * 2, dtype=torch.float32, device="cuda")
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            attr = scalar_to_color(feats[:, channel]).float()
            with torch.autocast(device_type="cuda", enabled=False):
                res = renderer.render(rep, ext.float(), intr.float(), colors_overwrite=attr)
            row = j // 2
            col = j % 2
            image[
                :,
                render_resolution * row:render_resolution * (row + 1),
                render_resolution * col:render_resolution * (col + 1),
            ] = res["color"].float()
        arr = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        path = out_prefix.with_name(f"{out_prefix.name}_{name}.jpg")
        Image.fromarray(arr).save(path, quality=95)
        paths[name] = path
    return paths


def make_comparison(rows, output_dir: Path, field: str, strategy: str):
    images = []
    for row in rows:
        baseline_img = Image.open(row[f"baseline_{field}"]).convert("RGB")
        exp_img = Image.open(row[f"{strategy}_{field}"]).convert("RGB")
        w = max(baseline_img.width, exp_img.width)
        h = max(baseline_img.height, exp_img.height)
        canvas = Image.new("RGB", (2 * w, h + 40), "white")
        canvas.paste(baseline_img, ((w - baseline_img.width) // 2, 40))
        canvas.paste(exp_img, (w + (w - exp_img.width) // 2, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{row['sha256'][:10]} | baseline 256 encode/decode", fill=(0, 0, 0))
        draw.text((w + 8, 8), f"512 support/cache + {strategy}", fill=(0, 0, 0))
        images.append(canvas)
    sheet = Image.new("RGB", (images[0].width, sum(im.height for im in images)), "white")
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output_dir / f"{field}_baseline256_vs_{strategy}_512.jpg", quality=95)


def main():
    args = parse_args()
    root = Path(args.root)
    vae_dir = Path(args.vae_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_resolution % args.source_resolution != 0:
        raise ValueError("--target_resolution must be divisible by --source_resolution.")
    scale = args.target_resolution // args.source_resolution
    cfg = json.load(open(vae_dir / "config.json", "r"))
    source_dataset_args = require_triangle_field_dataset_args(to_edict(cfg), args.source_resolution)
    target_dataset_args = to_edict(dict(source_dataset_args))
    target_dataset_args.resolution = args.target_resolution

    encoder = load_model(cfg, "encoder", vae_dir / "ckpts" / f"encoder_{args.ckpt}.pt")
    decoder = load_model(cfg, "decoder", vae_dir / "ckpts" / f"decoder_{args.ckpt}.pt", force_fp32=True)

    instances = load_instances(args.instances, args.num_samples)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    rows = []
    summary = {
        "source_resolution": args.source_resolution,
        "target_resolution": args.target_resolution,
        "source_latents": "runtime_encoder_output",
        "strategies": strategies,
        "instances": [],
    }
    for instance in instances:
        source_z, source_cache = encode_target_latent(
            root, args.split, args.source_resolution, instance, source_dataset_args, encoder
        )
        target_z, target_cache = encode_target_latent(
            root, args.split, args.target_resolution, instance, target_dataset_args, encoder
        )
        baseline_voxel = decode(decoder, source_z, source_cache)

        baseline_paths = render_voxel_channels(
            baseline_voxel,
            output_dir / f"{instance}_baseline_256",
            args.source_resolution,
            args.render_resolution,
            source_dataset_args.distance_transform,
        )
        row = {
            "sha256": instance,
            "baseline_d_tri": baseline_paths["d_tri"],
            "baseline_d_vert": baseline_paths["d_vert"],
        }
        instance_summary = {
            "sha256": instance,
            "source_tokens": int(source_z.feats.shape[0]),
            "strategy_stats": {},
        }
        for strategy in strategies:
            experiment_z, stats = make_experiment_features(source_z, target_z, scale, strategy, args)
            experiment_voxel = decode(decoder, experiment_z, target_cache)
            experiment_paths = render_voxel_channels(
                experiment_voxel,
                output_dir / f"{instance}_{strategy}_512",
                args.target_resolution,
                args.render_resolution,
                source_dataset_args.distance_transform,
            )
            row[f"{strategy}_d_tri"] = experiment_paths["d_tri"]
            row[f"{strategy}_d_vert"] = experiment_paths["d_vert"]
            instance_summary["strategy_stats"][strategy] = stats
        rows.append(row)
        summary["instances"].append(instance_summary)

    for strategy in strategies:
        make_comparison(rows, output_dir, "d_tri", strategy)
        make_comparison(rows, output_dir, "d_vert", strategy)
    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
