import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from trellis2.datasets.sparse_voxel_triangle_field import SparseVoxelTriangleFieldVisMixin
from trellis2.modules import sparse as sp

from eval_triangle_field_latent_sr_cascade import (
    save_image_grid,
    sparse_condition_from_low_to_high,
)
from eval_triangle_field_latent_sr_flow import (
    apply_conditioning_augmentation_overrides,
    build_trainer,
    find_ckpt_step,
    load_config,
    load_encoder_checkpoint,
)
from eval_triangle_field_latent_sr_stage_repeat_cascade import (
    average_downsample_to_low,
    constant_density_to_high,
    run_stage,
)


class SupportOnlyDataset(SparseVoxelTriangleFieldVisMixin):
    def __init__(self, resolution: int):
        self.resolution = int(resolution)
        self.distance_transform = "minus_one_one"
        self.loads = [1]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        coords = torch.zeros((1, 4), dtype=torch.int32)
        feats = torch.zeros((1, 2), dtype=torch.float32)
        x = sp.SparseTensor(feats, coords)
        return {"x_0": x, "cond": x}

    @staticmethod
    def collate_fn(batch, split_size=None):
        return batch[0]

    def __str__(self):
        return f"SupportOnlyDataset(resolution={self.resolution})"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Support-only repeat-cascade latent SR eval for OBJ/GLB meshes."
    )
    parser.add_argument("--mesh_dir", "--obj_dir", dest="mesh_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--base_guidance_strength", type=float, default=0.0)
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument("--stage_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--support_cache_dir", type=str, default=None)
    parser.add_argument("--max_active_voxels", type=int, default=1000000)
    parser.add_argument("--apply_conditioning_augmentation", action="store_true")
    parser.add_argument("--conditioning_augmentation_noise_level", type=float, default=None)
    parser.add_argument("--conditioning_augmentation_blur_sigma", type=float, default=None)
    parser.add_argument("--conditioning_augmentation_disable_blur", action="store_true")
    parser.add_argument("--constant_density_conditioning", action="store_true")
    parser.add_argument("--constant_density_value", type=float, default=-5.0)
    parser.add_argument("--constant_density_base_resolution", type=int, default=128)
    parser.add_argument("--constant_density_scale_mode", choices=("none", "voxel_size"), default="voxel_size")
    return parser.parse_args()


def require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "OBJ/GLB support extraction uses trimesh. Install it in the trellis2 env "
            "or run: pip install trimesh"
        ) from exc
    return trimesh


def mesh_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_meshes(mesh_dir: Path, recursive: bool) -> list[Path]:
    suffixes = {".obj", ".glb"}
    iterator = mesh_dir.rglob("*") if recursive else mesh_dir.iterdir()
    paths = sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"No .obj or .glb meshes found under {mesh_dir}")
    return paths


def load_normalized_mesh(path: Path):
    trimesh = require_trimesh()
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"{path} did not load as a non-empty triangle mesh")

    mesh = loaded.copy()
    bounds = mesh.bounds
    extent = float((bounds[1] - bounds[0]).max())
    if extent <= 0:
        raise ValueError(f"{path} has zero spatial extent")
    mesh.apply_translation(-0.5 * (bounds[0] + bounds[1]))
    mesh.apply_scale(0.99999 / extent)
    return mesh


def support_coords_from_mesh(path: Path, resolution: int, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{mesh_sha1(path)}_r{resolution}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            return data["coords"].astype(np.int32, copy=False)

    mesh = load_normalized_mesh(path)
    voxels = mesh.voxelized(pitch=1.0 / float(resolution))
    points = np.asarray(voxels.points, dtype=np.float32)
    coords = np.floor((points + 0.5) * float(resolution)).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    if coords.size == 0:
        coords = np.empty((0, 3), dtype=np.int32)
    else:
        coords = np.unique(coords, axis=0).astype(np.int32)

    np.savez_compressed(cache_path, coords=coords, mesh_path=str(path), resolution=resolution)
    return coords


def make_support_tensor(supports: dict[int, list[np.ndarray]], resolution: int, start: int, end: int, device):
    coords_parts, feats_parts = [], []
    for batch_idx, coords_np in enumerate(supports[resolution][start:end]):
        coords = torch.as_tensor(coords_np, dtype=torch.int32, device=device)
        batch = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32, device=device)
        coords_parts.append(torch.cat([batch, coords], dim=1))
        feats_parts.append(torch.zeros((coords.shape[0], 2), dtype=torch.float32, device=device))
    return sp.SparseTensor(torch.cat(feats_parts, dim=0), torch.cat(coords_parts, dim=0))


def add_visuals(images: dict, visualizer: SupportOnlyDataset, prefix: str, tensor: sp.SparseTensor):
    vis = visualizer.visualize_sample({"target": tensor})
    for key, value in vis.items():
        images.setdefault(f"{prefix}_{key}", []).append(value.cpu())


def main():
    args = parse_args()
    if args.stage_repeats < 1:
        raise ValueError("--stage_repeats must be >= 1")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mesh_dir = Path(args.mesh_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(run_dir)
    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / (
        f"eval_mesh_folder_stage_repeat{args.stage_repeats}_cascade_128to512"
        f"_step{ckpt_step:07d}_cfg{args.guidance_strength:g}"
        f"_basecfg{args.base_guidance_strength:g}_steps{args.steps}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.support_cache_dir).resolve() if args.support_cache_dir else output_dir / "support_cache"

    mesh_paths = collect_meshes(mesh_dir, args.recursive)
    if args.num_samples is not None:
        mesh_paths = mesh_paths[:args.num_samples]
    mesh_hashes = [mesh_sha1(path) for path in mesh_paths]

    stages = [(64, 128), (128, 256), (256, 512)]
    target_resolutions = [high for _, high in stages]
    supports = {resolution: [] for resolution in target_resolutions}
    for path in mesh_paths:
        for resolution in target_resolutions:
            coords = support_coords_from_mesh(path, resolution, cache_dir)
            if coords.shape[0] == 0:
                raise ValueError(f"{path} produced empty support at resolution {resolution}")
            if coords.shape[0] > args.max_active_voxels:
                raise ValueError(
                    f"{path} has {coords.shape[0]} support voxels at {resolution}, "
                    f"above --max_active_voxels {args.max_active_voxels}"
                )
            supports[resolution].append(coords)

    trainer = build_trainer(cfg, SupportOnlyDataset(128), output_dir)
    apply_conditioning_augmentation_overrides(trainer, args)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)
    latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])

    images = {}
    for start in range(0, len(mesh_paths), args.batch_size):
        end = min(start + args.batch_size, len(mesh_paths))
        previous_refined = None
        for stage_idx, (low_res, high_res) in enumerate(stages):
            high_support = make_support_tensor(supports, high_res, start, end, trainer.device)
            visualizer = SupportOnlyDataset(high_res)
            add_visuals(images, visualizer, f"stage{low_res}to{high_res}_support", high_support)

            cond = high_support.replace(torch.zeros_like(high_support.feats)) if stage_idx == 0 else (
                sparse_condition_from_low_to_high(previous_refined, high_support, low_res, high_res)
            )
            guidance = args.base_guidance_strength if stage_idx == 0 else args.guidance_strength
            density_cond = None
            if args.constant_density_conditioning:
                density_cond = constant_density_to_high(
                    high_support,
                    args.constant_density_value,
                    args.constant_density_base_resolution,
                    high_res,
                    args.constant_density_scale_mode,
                )

            prefix = f"stage{low_res}to{high_res}"
            sample = None
            for repeat_idx in range(args.stage_repeats):
                repeat_num = repeat_idx + 1
                sample, pred_last = run_stage(
                    trainer,
                    {"x_0": high_support},
                    cond,
                    latent_channels,
                    args.steps,
                    guidance,
                    args.apply_conditioning_augmentation,
                    density_cond,
                )
                add_visuals(images, visualizer, f"{prefix}_iter{repeat_num}_cond", cond)
                add_visuals(images, visualizer, f"{prefix}_iter{repeat_num}_sample", sample)
                add_visuals(images, visualizer, f"{prefix}_iter{repeat_num}_pred_z0_last", pred_last)

                if repeat_idx < args.stage_repeats - 1:
                    feedback_low = average_downsample_to_low(sample, factor=2)
                    add_visuals(images, SupportOnlyDataset(low_res), f"{prefix}_iter{repeat_num}_avg_down_to_{low_res}", feedback_low)
                    cond = sparse_condition_from_low_to_high(feedback_low, high_support, low_res, high_res)
                    guidance = args.guidance_strength
            previous_refined = sample

    suffix = (
        f"step{ckpt_step:07d}_stage_repeat{args.stage_repeats}_128to512"
        f"_cfg{args.guidance_strength:g}_basecfg{args.base_guidance_strength:g}"
    )
    for name, chunks in images.items():
        save_image_grid(torch.cat(chunks, dim=0)[:len(mesh_paths)], output_dir / f"{name}_{suffix}.jpg")

    summary = {
        "mesh_dir": str(mesh_dir),
        "meshes": [{"path": str(path), "sha1": sha} for path, sha in zip(mesh_paths, mesh_hashes)],
        "support_only": True,
        "support_extractor": "trimesh.load(...).voxelized(pitch=1/resolution)",
        "normalization": "bbox center, max extent scaled to 0.99999",
        "stages": stages,
        "stage_repeats": args.stage_repeats,
        "steps": args.steps,
        "base_guidance_strength": args.base_guidance_strength,
        "guidance_strength": args.guidance_strength,
        "checkpoint": ckpt_path,
        "support_cache_dir": str(cache_dir),
        "constant_density_conditioning": args.constant_density_conditioning,
        "constant_density_value": args.constant_density_value,
        "constant_density_base_resolution": args.constant_density_base_resolution,
        "constant_density_scale_mode": args.constant_density_scale_mode,
        "support_counts": {
            str(resolution): [int(coords.shape[0]) for coords in supports[resolution]]
            for resolution in target_resolutions
        },
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved mesh-folder support-only cascade eval to {output_dir}")


if __name__ == "__main__":
    main()
