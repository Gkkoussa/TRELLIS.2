import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from eval_triangle_field_flow import find_ckpt_step, load_denoiser_checkpoint
from trellis2 import models
from trellis2.datasets.sparse_voxel_triangle_field import load_triangle_field_npz
from trellis2.datasets.structured_latent_triangle_field import MichelangeloShapeConditionedTriangleFieldSLat
from trellis2.modules import sparse as sp
from trellis2.pipelines.samplers import FlowEulerCfgSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test a 256-trained triangle-field flow on 512-derived shape latent coordinates."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default="0.9999")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--voxel_resolution", type=int, default=512)
    parser.add_argument(
        "--upsample_shape_from_resolution",
        type=int,
        default=None,
        help=(
            "If set, encode shape latents at this lower voxel resolution and copy each "
            "parent feature onto occupied target-resolution latent children. The target "
            "support still comes from --voxel_resolution."
        ),
    )
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--sampling_steps", type=int, default=12)
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--triangle_field_latent_name", type=str, default="triangle_field_vae_51685536_step0100000_256")
    parser.add_argument("--michelangelo_latent_name", type=str, default="shapevae256_pretrained")
    parser.add_argument("--shape_latent_name", type=str, default="occupancy_shape_vae_triangle_filtered_51728720_step0100000_256")
    parser.add_argument(
        "--shape_vae_dir",
        type=str,
        default="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/outputs/occupancy_shape_vae_triangle_filtered_51728720",
    )
    parser.add_argument("--shape_vae_ckpt", type=str, default="step0100000")
    return parser.parse_args()


def load_instances(root: Path, split: str, instances: str | None, num_samples: int) -> list[str]:
    if instances is not None:
        if "," in instances:
            values = [item.strip() for item in instances.split(",") if item.strip()]
        else:
            path = Path(instances)
            if path.exists():
                values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            else:
                values = [instances.strip()]
    else:
        path = root / "splits" / split / f"triangle_field_voxels_512" / "instances.txt"
        if not path.exists():
            path = root / "splits" / split / "instances.txt"
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return values[:num_samples]


def load_triangle_field_coords(root: Path, split: str, resolution: int, instance: str) -> torch.Tensor:
    split_path = root / "splits" / split / f"triangle_field_voxels_{resolution}" / f"{instance}.npz.zst"
    canonical_path = root / f"triangle_field_voxels_{resolution}" / f"{instance}.npz.zst"
    path = split_path if split_path.exists() else canonical_path
    with load_triangle_field_npz(str(path)) as data:
        coords = torch.from_numpy(data["coords"].astype(np.int32, copy=False))
    return coords


def make_sparse_occupancy(coords: torch.Tensor) -> sp.SparseTensor:
    batch = torch.zeros_like(coords[:, :1])
    sparse_coords = torch.cat([batch, coords], dim=1).int()
    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
    return sp.SparseTensor(feats, sparse_coords)


def load_model_from_config(config, key, ckpt_path: Path):
    model_cfg = config["models"][key]
    model = getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda().eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=False)
    return model


def normalize_sparse(x: sp.SparseTensor, normalization_path: Path) -> sp.SparseTensor:
    with open(normalization_path, "r") as fp:
        stats = json.load(fp)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=x.feats.device).reshape(1, -1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=x.feats.device).reshape(1, -1)
    return x.replace((x.feats - mean) / std)


def copy_parent_features_to_target_support(source: sp.SparseTensor, target: sp.SparseTensor, scale: int):
    if scale < 1:
        raise ValueError(f"Upsample scale must be >= 1, got {scale}.")
    source_xyz = source.coords[:, 1:].detach().cpu()
    target_xyz = target.coords[:, 1:].detach().cpu()

    source_lookup = {tuple(coord.tolist()): idx for idx, coord in enumerate(source_xyz)}
    source_indices = []
    target_indices = []
    for target_idx, coord in enumerate(target_xyz):
        parent = tuple((coord // scale).tolist())
        source_idx = source_lookup.get(parent)
        if source_idx is not None:
            source_indices.append(source_idx)
            target_indices.append(target_idx)

    if not target_indices:
        raise ValueError("No target latent coordinates had an occupied source parent.")

    source_indices = torch.tensor(source_indices, dtype=torch.long, device=source.feats.device)
    target_indices = torch.tensor(target_indices, dtype=torch.long, device=target.feats.device)
    return sp.SparseTensor(source.feats[source_indices], target.coords[target_indices])


def tile_images(paths: list[Path], out_path: Path, label: str):
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = images[0].size
    canvas = Image.new("RGB", (width, height * len(images) + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill=(0, 0, 0))
    y = 28
    for image in images:
        canvas.paste(image, (0, y))
        y += height
    canvas.save(out_path, quality=95)


def main():
    args = parse_args()
    root = Path(args.root)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    flow_cfg = json.load(open(run_dir / "config.json", "r"))
    shape_cfg = json.load(open(Path(args.shape_vae_dir) / "config.json", "r"))

    shape_encoder = load_model_from_config(
        shape_cfg,
        "encoder",
        Path(args.shape_vae_dir) / "ckpts" / f"encoder_{args.shape_vae_ckpt}.pt",
    )

    denoiser_cfg = flow_cfg["models"]["denoiser"]
    denoiser = getattr(models, denoiser_cfg["name"])(**denoiser_cfg["args"]).cuda().eval()
    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    ckpt_path = load_denoiser_checkpoint(denoiser, run_dir, ckpt_step, args.ema_rate, torch.device("cuda"))

    data_dir = {
        args.split: {
            "metadata": str(root / "splits" / args.split),
            "triangle_field_latent": str(root / "triangle_field_latents" / args.triangle_field_latent_name),
            "michelangelo_latent": str(root / "michelangelo_latents" / args.michelangelo_latent_name),
            "shape_latent": str(root / "shape_latents" / args.shape_latent_name),
        }
    }
    dataset_args = json.loads(json.dumps(flow_cfg["dataset"]["args"]))
    dataset_args["resolution"] = args.voxel_resolution
    dataset_args["snapshot_render_resolution"] = args.render_resolution
    dataset = MichelangeloShapeConditionedTriangleFieldSLat(json.dumps(data_dir), **dataset_args)

    shape_norm_path = root / "shape_latents" / args.shape_latent_name / "normalization.json"
    instances = load_instances(root, args.split, args.instances, args.num_samples)
    sampler = FlowEulerCfgSampler(flow_cfg["trainer"]["args"]["sigma_min"])

    sample_images = {"d_tri": [], "d_vert": []}
    summary = {
        "instances": [],
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "guidance_strength": args.guidance_strength,
        "sampling_steps": args.sampling_steps,
    }

    for instance in instances:
        coords = load_triangle_field_coords(root, args.split, args.voxel_resolution, instance)
        occupancy = make_sparse_occupancy(coords).cuda()
        target_shape_z_raw = shape_encoder(occupancy, sample_posterior=False)
        target_shape_z = normalize_sparse(target_shape_z_raw, shape_norm_path)

        source_shape_z_tokens = None
        target_shape_z_tokens = int(target_shape_z.feats.shape[0])
        parent_hit_tokens = None
        if args.upsample_shape_from_resolution is None:
            shape_z = target_shape_z
        else:
            if args.voxel_resolution % args.upsample_shape_from_resolution != 0:
                raise ValueError(
                    f"--voxel_resolution ({args.voxel_resolution}) must be divisible by "
                    f"--upsample_shape_from_resolution ({args.upsample_shape_from_resolution})."
                )
            scale = args.voxel_resolution // args.upsample_shape_from_resolution
            source_coords = load_triangle_field_coords(root, args.split, args.upsample_shape_from_resolution, instance)
            source_occupancy = make_sparse_occupancy(source_coords).cuda()
            source_shape_z_raw = shape_encoder(source_occupancy, sample_posterior=False)
            source_shape_z = normalize_sparse(source_shape_z_raw, shape_norm_path)
            source_shape_z_tokens = int(source_shape_z.feats.shape[0])
            shape_z = copy_parent_features_to_target_support(source_shape_z, target_shape_z, scale)
            parent_hit_tokens = int(shape_z.feats.shape[0])

        cond_npz = np.load(root / "michelangelo_latents" / args.michelangelo_latent_name / f"{instance}.npz")
        cond = torch.from_numpy(cond_npz["feats"]).float().unsqueeze(0).cuda()
        neg_cond = torch.zeros_like(cond)

        noise = shape_z.replace(torch.randn((shape_z.feats.shape[0], 32), device="cuda", dtype=torch.float32))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sample = sampler.sample(
                denoiser,
                noise=noise,
                cond=cond,
                neg_cond=neg_cond,
                concat_cond=shape_z,
                steps=args.sampling_steps,
                guidance_strength=args.guidance_strength,
                verbose=False,
            ).samples

        vis = dataset.visualize_sample({"x_0": sample, "concat_cond": shape_z})
        for key, value in vis.items():
            img = (value[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            path = output_dir / f"{instance}_{key}.jpg"
            Image.fromarray(img).save(path, quality=95)
            if key in sample_images:
                sample_images[key].append(path)

        summary["instances"].append({
            "sha256": instance,
            "input_active_voxels": int(coords.shape[0]),
            "shape_latent_tokens": int(shape_z.feats.shape[0]),
            "target_shape_latent_tokens": target_shape_z_tokens,
            "source_shape_latent_tokens": source_shape_z_tokens,
            "parent_hit_tokens": parent_hit_tokens,
        })

    for key, paths in sample_images.items():
        if paths:
            tile_images(paths, output_dir / f"summary_{key}.jpg", f"{key} 512-derived shape latent coords")

    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
