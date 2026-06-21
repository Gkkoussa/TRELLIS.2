import argparse
import json
from pathlib import Path

from easydict import EasyDict as edict
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

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
            "Test whether 512 triangle-field latents become valid 256 latents "
            "after averaging active 2x2x2 latent children into parent cells."
        )
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--vae_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--source_resolution", type=int, default=512)
    parser.add_argument("--target_resolution", type=int, default=256)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--force_fp32_decoder", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--require_matching_latent_support", action="store_true")
    return parser.parse_args()


def find_ckpt_step(vae_dir: Path, ckpt: str) -> str:
    if ckpt != "latest":
        return ckpt
    ckpts = sorted((vae_dir / "ckpts").glob("encoder_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No encoder_step*.pt checkpoints found in {vae_dir / 'ckpts'}")
    return ckpts[-1].stem.replace("encoder_", "")


def to_edict(value):
    if isinstance(value, dict):
        return edict({k: to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_edict(v) for v in value]
    return value


def load_instances(root: Path, split: str, instances: str | None, random: bool, seed: int):
    if instances is None:
        path = root / "splits" / split / "instances.txt"
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


def filter_instances_with_voxels(
    root: Path,
    split: str,
    instances: list[str],
    source_resolution: int,
    target_resolution: int,
    source_dataset_args,
    target_dataset_args,
    num_samples: int,
):
    source_root = root / "splits" / split / f"{source_dataset_args.voxel_dirname}_{source_resolution}"
    target_root = root / "splits" / split / f"{target_dataset_args.voxel_dirname}_{target_resolution}"
    kept = []
    missing_source = 0
    missing_target = 0
    for instance in instances:
        try:
            find_triangle_field_path(str(source_root), instance)
        except FileNotFoundError:
            missing_source += 1
            continue
        try:
            find_triangle_field_path(str(target_root), instance)
        except FileNotFoundError:
            missing_target += 1
            continue
        kept.append(instance)
        if len(kept) >= num_samples:
            break
    if len(kept) == 0:
        raise RuntimeError(
            f"No instances had both {source_resolution} and {target_resolution} triangle-field voxels. "
            f"Missing source before target check: {missing_source}; missing target: {missing_target}."
        )
    print(
        f"Using {len(kept)} instances with both {source_resolution} and {target_resolution} voxels "
        f"(skipped missing source={missing_source}, missing target={missing_target})."
    )
    return kept, {"missing_source": missing_source, "missing_target": missing_target}


def load_model(config, key: str, ckpt_path: Path, force_fp32: bool = False):
    model_cfg = config["models"][key]
    model = getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda().eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=False)
    if force_fp32 and hasattr(model, "convert_to_fp32"):
        model.convert_to_fp32()
        if hasattr(model, "dtype"):
            model.dtype = torch.float32
    return model


def dataset_args_for_resolution(config, resolution: int):
    cfg = to_edict(config)
    dataset_args = require_triangle_field_dataset_args(cfg, int(config["dataset"]["args"]["resolution"]))
    dataset_args = to_edict(dict(dataset_args))
    dataset_args.resolution = resolution
    return dataset_args


def encode_field(root: Path, split: str, resolution: int, instance: str, dataset_args, encoder):
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
    return x, z, cache


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


def attach_cache(z: sp.SparseTensor, cache: dict) -> sp.SparseTensor:
    z._scale = cache_to_device(cache["scale"], z.device)
    z._spatial_cache = cache_to_device(cache["spatial_cache"], z.device)
    return z


def parent_average_latents(z: sp.SparseTensor, scale: int) -> tuple[sp.SparseTensor, dict]:
    coords = z.coords.clone()
    parent_coords = coords.clone()
    parent_coords[:, 1:] = parent_coords[:, 1:] // scale
    unique_coords, inverse, counts = torch.unique(
        parent_coords,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    feats = torch.zeros(
        unique_coords.shape[0],
        z.feats.shape[1],
        device=z.feats.device,
        dtype=z.feats.dtype,
    )
    feats.scatter_add_(0, inverse[:, None].expand(-1, z.feats.shape[1]), z.feats)
    feats = feats / counts.to(device=z.feats.device, dtype=z.feats.dtype)[:, None]
    out = sp.SparseTensor(feats, unique_coords)
    stats = {
        "source_tokens": int(z.feats.shape[0]),
        "parent_tokens": int(out.feats.shape[0]),
        "mean_children_per_parent": float(counts.float().mean().item()),
        "max_children_per_parent": int(counts.max().item()),
    }
    return out, stats


def latent_support_stats(parent_z: sp.SparseTensor, target_z: sp.SparseTensor) -> dict:
    parent_set = {tuple(coord.tolist()) for coord in parent_z.coords.detach().cpu()}
    target_set = {tuple(coord.tolist()) for coord in target_z.coords.detach().cpu()}
    intersection = parent_set & target_set
    coords_equal = (
        parent_z.coords.shape == target_z.coords.shape
        and torch.equal(parent_z.coords.detach().cpu(), target_z.coords.detach().cpu())
    )
    return {
        "coords_equal_ordered": bool(coords_equal),
        "parent_tokens": len(parent_set),
        "target_tokens": len(target_set),
        "intersection_tokens": len(intersection),
        "parent_only_tokens": len(parent_set - target_set),
        "target_only_tokens": len(target_set - parent_set),
        "intersection_over_target": len(intersection) / max(len(target_set), 1),
        "intersection_over_parent": len(intersection) / max(len(parent_set), 1),
    }


def decode(decoder, z: sp.SparseTensor, cache: dict | None):
    if cache is not None:
        z = attach_cache(z, cache)
    with torch.no_grad():
        return decoder(z)


def maybe_inverse_distance_transform(values: torch.Tensor, distance_transform: str) -> torch.Tensor:
    if distance_transform == "minus_one_one":
        return values * 0.5 + 0.5
    if distance_transform == "none":
        return values
    raise ValueError(f"Unsupported distance_transform: {distance_transform}")


def metrics(pred: sp.SparseTensor, target: sp.SparseTensor, distance_transform: str):
    if pred.feats.shape != target.feats.shape or not torch.equal(pred.coords, target.coords):
        return {
            "support_matches_target": False,
            "pred_tokens": int(pred.feats.shape[0]),
            "target_tokens": int(target.feats.shape[0]),
        }
    pred_values = maybe_inverse_distance_transform(pred.feats.float(), distance_transform)
    target_values = maybe_inverse_distance_transform(target.feats.float(), distance_transform)
    err = pred_values - target_values
    return {
        "support_matches_target": True,
        "pred_tokens": int(pred.feats.shape[0]),
        "target_tokens": int(target.feats.shape[0]),
        "l1_d_tri": float(err[:, 0].abs().mean().item()),
        "l1_d_vert": float(err[:, 1].abs().mean().item()),
        "rmse_d_tri": float(err[:, 0].square().mean().sqrt().item()),
        "rmse_d_vert": float(err[:, 1].square().mean().sqrt().item()),
    }


def scalar_to_color(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1, 1).clamp(0, 1)
    return values.expand(-1, 3)


def render_field(voxel: sp.SparseTensor, path: Path, voxel_resolution: int, render_resolution: int, distance_transform: str, channel: int):
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
    attr = scalar_to_color(feats[:, channel]).float()
    image = torch.zeros(3, render_resolution * 2, render_resolution * 2, dtype=torch.float32, device="cuda")
    for j, (ext, intr) in enumerate(zip(exts, ints)):
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
    Image.fromarray(arr).save(path, quality=95)
    return path


def make_row_image(paths: list[Path], labels: list[str], out_path: Path):
    images = [Image.open(path).convert("RGB") for path in paths]
    w = max(image.width for image in images)
    h = max(image.height for image in images)
    label_h = 36
    canvas = Image.new("RGB", (len(images) * w, h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        x0 = i * w + (w - image.width) // 2
        canvas.paste(image, (x0, label_h))
        draw.text((i * w + 8, 10), label, fill=(0, 0, 0))
    canvas.save(out_path, quality=95)


def main():
    args = parse_args()
    root = Path(args.root)
    vae_dir = Path(args.vae_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_resolution % args.target_resolution != 0:
        raise ValueError("--source_resolution must be divisible by --target_resolution.")
    scale = args.source_resolution // args.target_resolution

    ckpt = find_ckpt_step(vae_dir, args.ckpt)
    config = json.load(open(vae_dir / "config.json", "r"))
    source_dataset_args = dataset_args_for_resolution(config, args.source_resolution)
    target_dataset_args = dataset_args_for_resolution(config, args.target_resolution)
    distance_transform = target_dataset_args.distance_transform

    encoder = load_model(config, "encoder", vae_dir / "ckpts" / f"encoder_{ckpt}.pt")
    decoder = load_model(
        config,
        "decoder",
        vae_dir / "ckpts" / f"decoder_{ckpt}.pt",
        force_fp32=args.force_fp32_decoder,
    )
    decoder_predicts_subdiv = bool(getattr(decoder, "pred_subdiv", False))

    candidate_instances = load_instances(root, args.split, args.instances, args.random, args.seed)
    instances, instance_filter_stats = filter_instances_with_voxels(
        root,
        args.split,
        candidate_instances,
        args.source_resolution,
        args.target_resolution,
        source_dataset_args,
        target_dataset_args,
        args.num_samples,
    )
    rows = []
    aggregate = {
        "baseline": {"l1_d_tri": [], "l1_d_vert": [], "rmse_d_tri": [], "rmse_d_vert": []},
        "downsampled_512_latent": {"l1_d_tri": [], "l1_d_vert": [], "rmse_d_tri": [], "rmse_d_vert": []},
    }

    with torch.no_grad():
        for instance in tqdm(instances, desc="Downsample latent smoke test"):
            source_x, source_z, _ = encode_field(root, args.split, args.source_resolution, instance, source_dataset_args, encoder)
            target_x, target_z, target_cache = encode_field(root, args.split, args.target_resolution, instance, target_dataset_args, encoder)
            parent_z, parent_stats = parent_average_latents(source_z, scale=scale)
            support_stats = latent_support_stats(parent_z, target_z)
            if args.require_matching_latent_support and not support_stats["coords_equal_ordered"]:
                raise RuntimeError(
                    f"Latent support mismatch for {instance}: {support_stats}. "
                    "The parent-averaged 512 latent cannot be safely decoded with 256 caches."
                )

            baseline = decode(decoder, target_z, None if decoder_predicts_subdiv else target_cache)
            experiment = decode(decoder, parent_z, None if decoder_predicts_subdiv else target_cache)
            target = target_x.replace(target_x.feats[:, :2])

            baseline_metrics = metrics(baseline, target, distance_transform)
            experiment_metrics = metrics(experiment, target, distance_transform)
            row = {
                "sha256": instance,
                "parent_average": parent_stats,
                "latent_support": support_stats,
                "baseline": baseline_metrics,
                "downsampled_512_latent": experiment_metrics,
            }
            rows.append(row)
            for key in aggregate:
                for metric_name in aggregate[key]:
                    if row[key].get("support_matches_target", False):
                        aggregate[key][metric_name].append(row[key][metric_name])

            if not args.skip_render:
                sample_dir = output_dir / instance
                sample_dir.mkdir(parents=True, exist_ok=True)
                rendered = {}
                for channel_name, channel_idx in (("d_tri", 0), ("d_vert", 1)):
                    rendered[f"gt_{channel_name}"] = render_field(
                        target,
                        sample_dir / f"gt_{channel_name}.jpg",
                        args.target_resolution,
                        args.render_resolution,
                        distance_transform,
                        channel_idx,
                    )
                    rendered[f"baseline_{channel_name}"] = render_field(
                        baseline,
                        sample_dir / f"baseline_{channel_name}.jpg",
                        args.target_resolution,
                        args.render_resolution,
                        distance_transform,
                        channel_idx,
                    )
                    rendered[f"experiment_{channel_name}"] = render_field(
                        experiment,
                        sample_dir / f"downsampled_512_latent_{channel_name}.jpg",
                        args.target_resolution,
                        args.render_resolution,
                        distance_transform,
                        channel_idx,
                    )
                    make_row_image(
                        [
                            rendered[f"gt_{channel_name}"],
                            rendered[f"baseline_{channel_name}"],
                            rendered[f"experiment_{channel_name}"],
                        ],
                        [
                            "GT 256",
                            "baseline: encode 256 -> decode 256",
                            "encode 512 -> avg latent children -> decode 256",
                        ],
                        sample_dir / f"compare_{channel_name}.jpg",
                    )

    summary = {
        "vae_dir": str(vae_dir),
        "ckpt": ckpt,
        "split": args.split,
        "source_resolution": args.source_resolution,
        "target_resolution": args.target_resolution,
        "latent_parent_scale": scale,
        "decoder_predicts_subdiv": decoder_predicts_subdiv,
        "instance_filter": instance_filter_stats,
        "instances": rows,
        "aggregate": {},
    }
    for key, values in aggregate.items():
        summary["aggregate"][key] = {
            metric_name: (float(np.mean(metric_values)) if metric_values else None)
            for metric_name, metric_values in values.items()
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
