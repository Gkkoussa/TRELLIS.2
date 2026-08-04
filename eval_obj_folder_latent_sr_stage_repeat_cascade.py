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
    parser.add_argument("--early_stage_repeats", type=int, default=None)
    parser.add_argument(
        "--early_stage_repeats_through_resolution",
        type=int,
        choices=(32, 64, 128, 256, 512),
        default=None,
        help="Use --early_stage_repeats for stages whose output resolution is at most this value.",
    )
    parser.add_argument(
        "--start_resolution",
        type=int,
        choices=(16, 64, 256),
        default=64,
        help=(
            "Low resolution of the first SR stage; 16 adds 16->32->64 before the usual "
            "cascade, while 256 runs only the direct 256->512 stage."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--support_cache_dir", type=str, default=None)
    parser.add_argument(
        "--nested_supports",
        action="store_true",
        help="Derive 256 and 128 supports from the 512 support to guarantee parent coverage.",
    )
    parser.add_argument("--max_active_voxels", type=int, default=1000000)
    parser.add_argument("--apply_conditioning_augmentation", action="store_true")
    parser.add_argument("--conditioning_augmentation_noise_level", type=float, default=None)
    parser.add_argument("--conditioning_augmentation_blur_sigma", type=float, default=None)
    parser.add_argument("--conditioning_augmentation_disable_blur", action="store_true")
    parser.add_argument("--constant_density_conditioning", action="store_true")
    parser.add_argument(
        "--density_payload_dir",
        type=str,
        default=None,
        help=(
            "Directory of per-mesh 512-resolution coords+density payloads. "
            "Density is averaged onto each nested support and voxel-size scaled."
        ),
    )
    parser.add_argument("--constant_density_value", type=float, default=-5.0)
    parser.add_argument("--constant_density_base_resolution", type=int, default=128)
    parser.add_argument("--constant_density_scale_mode", choices=("none", "voxel_size"), default="voxel_size")
    parser.add_argument("--density_guidance_strength", type=float, default=None)
    parser.add_argument(
        "--override_decoded_density",
        action="store_true",
        help="Replace the decoder density channel with --constant_density_value before encoder input assembly.",
    )
    parser.add_argument(
        "--decoded_density_external_condition_max",
        type=float,
        default=None,
        help="Condition on the decoder density as a separate field after clamping it to this maximum.",
    )
    parser.add_argument(
        "--decoded_density_external_condition_base_resolution",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--decoded_density_external_condition_scale_mode",
        choices=("none", "voxel_size"),
        default="none",
    )
    parser.add_argument(
        "--propagate_decoded_density_conditioning",
        action="store_true",
        help=(
            "Use a constant density condition for the first stage, then condition each "
            "repeat/stage on a clamped copy of the preceding final decoded density. "
            "The decoder density channel itself is never modified."
        ),
    )
    parser.add_argument("--propagated_density_initial_value", type=float, default=-2.5)
    parser.add_argument("--propagated_density_clamp_max", type=float, default=None)
    parser.add_argument(
        "--propagated_density_target_mean",
        type=float,
        default=None,
        help=(
            "After clamping propagated density, shift it per sample so its mean equals "
            "this value. The value uses the propagated density base-resolution convention."
        ),
    )
    parser.add_argument("--propagated_density_base_resolution", type=int, default=128)
    parser.add_argument(
        "--density_conditioning_max_resolution",
        type=int,
        choices=(32, 64, 128, 256, 512),
        default=None,
        help="Drop density conditioning above this stage output resolution.",
    )
    parser.add_argument("--drop_density_conditioning", action="store_true")
    parser.add_argument("--constant_elongation_conditioning", action="store_true")
    parser.add_argument("--constant_elongation_value", type=float, default=0.0)
    parser.add_argument("--elongation_guidance_strength", type=float, default=None)
    parser.add_argument("--drop_elongation_conditioning", action="store_true")
    parser.add_argument(
        "--elongation_conditioning_max_resolution",
        type=int,
        choices=(32, 64, 128, 256, 512),
        default=None,
        help="Drop elongation conditioning above this stage output resolution.",
    )
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


def triangle_field_channels(tensor: sp.SparseTensor) -> sp.SparseTensor:
    if tensor.feats.shape[1] < 2:
        raise ValueError(
            f"Triangle-field conditioning requires at least two channels, got {tensor.feats.shape[1]}"
        )
    return tensor if tensor.feats.shape[1] == 2 else tensor.replace(tensor.feats[:, :2])


def load_multires_density_payloads(
    payload_dir: Path,
    mesh_hashes: list[str],
    supports: dict[int, list[np.ndarray]],
    resolutions: list[int],
) -> dict[int, list[np.ndarray]]:
    density_by_resolution = {resolution: [] for resolution in resolutions}
    for mesh_idx, mesh_hash in enumerate(mesh_hashes):
        payload_path = payload_dir / f"{mesh_hash}.npz"
        if not payload_path.exists():
            raise FileNotFoundError(payload_path)
        with np.load(payload_path, allow_pickle=False) as payload:
            coords_512 = payload["coords"].astype(np.int32, copy=False)
            density_512 = payload["density"].astype(np.float32, copy=False).reshape(-1)
            payload_resolution = int(payload["resolution"])
        if payload_resolution != 512:
            raise ValueError(f"{payload_path} has resolution {payload_resolution}, expected 512")
        if coords_512.shape != (density_512.shape[0], 3):
            raise ValueError(
                f"Mismatched coords/density shapes in {payload_path}: "
                f"{coords_512.shape} vs {density_512.shape}"
            )

        for resolution in resolutions:
            factor = 512 // resolution
            parent_coords = coords_512 if factor == 1 else coords_512 // factor
            unique_coords, inverse = np.unique(
                parent_coords,
                axis=0,
                return_inverse=True,
            )
            counts = np.bincount(inverse)
            values = np.bincount(inverse, weights=density_512) / counts
            expected_coords = supports[resolution][mesh_idx]
            if not np.array_equal(unique_coords, expected_coords):
                raise ValueError(
                    f"Density payload support does not match nested {resolution} support "
                    f"for mesh {mesh_hash}"
                )
            values += 2.0 * math.log(512.0 / float(resolution))
            density_by_resolution[resolution].append(
                values.astype(np.float32, copy=False)[:, None]
            )
    return density_by_resolution


def make_density_tensor(
    density_by_resolution: dict[int, list[np.ndarray]],
    resolution: int,
    start: int,
    end: int,
    high_support: sp.SparseTensor,
) -> sp.SparseTensor:
    feats = np.concatenate(density_by_resolution[resolution][start:end], axis=0)
    return high_support.replace(torch.as_tensor(
        feats,
        dtype=high_support.feats.dtype,
        device=high_support.device,
    ))


def shift_mean_then_upper_clamp(
    tensor: sp.SparseTensor,
    clamp_max: float,
    target_mean: float,
) -> tuple[sp.SparseTensor, list[dict]]:
    if target_mean > clamp_max:
        raise ValueError(
            f"Density target mean {target_mean} exceeds clamp maximum {clamp_max}"
        )

    feats = tensor.feats.float()
    adjusted = torch.empty_like(feats)
    stats = []
    for batch_idx in tensor.coords[:, 0].unique():
        mask = tensor.coords[:, 0] == batch_idx
        values = feats[mask]
        input_mean = values.mean()
        shifted = values + (target_mean - input_mean)
        clamped = shifted.clamp(max=clamp_max)
        post_clamp_mean = clamped.mean()
        adjusted[mask] = clamped
        stats.append(
            {
                "batch_index": int(batch_idx.item()),
                "input_mean": float(input_mean.item()),
                "pre_clamp_mean": float(target_mean),
                "post_clamp_mean": float(post_clamp_mean.item()),
                "mean_reduction": float((target_mean - post_clamp_mean).item()),
                "clipped_fraction": float((shifted > clamp_max).float().mean().item()),
            }
        )

    return tensor.replace(adjusted.to(tensor.feats.dtype)), stats


def main():
    args = parse_args()
    if args.stage_repeats < 1:
        raise ValueError("--stage_repeats must be >= 1")
    if (args.early_stage_repeats is None) != (
        args.early_stage_repeats_through_resolution is None
    ):
        raise ValueError(
            "--early_stage_repeats and --early_stage_repeats_through_resolution "
            "must be provided together"
        )
    if args.early_stage_repeats is not None and args.early_stage_repeats < 1:
        raise ValueError("--early_stage_repeats must be >= 1")
    if args.drop_density_conditioning and args.constant_density_conditioning:
        raise ValueError(
            "Choose --drop_density_conditioning or --constant_density_conditioning, not both"
        )
    if args.density_payload_dir and (
        args.constant_density_conditioning
        or args.drop_density_conditioning
        or args.override_decoded_density
        or args.decoded_density_external_condition_max is not None
        or args.propagate_decoded_density_conditioning
    ):
        raise ValueError(
            "--density_payload_dir is mutually exclusive with other density source modes"
        )
    if args.decoded_density_external_condition_max is not None and (
        args.constant_density_conditioning
        or args.drop_density_conditioning
        or args.override_decoded_density
        or args.propagate_decoded_density_conditioning
    ):
        raise ValueError(
            "--decoded_density_external_condition_max is mutually exclusive with "
            "constant, dropped, overridden, or propagated decoded density modes"
        )
    if args.propagate_decoded_density_conditioning and (
        args.constant_density_conditioning
        or args.drop_density_conditioning
        or args.override_decoded_density
    ):
        raise ValueError(
            "--propagate_decoded_density_conditioning is mutually exclusive with "
            "constant, dropped, or overridden density modes"
        )
    if (
        args.propagate_decoded_density_conditioning
        and args.propagated_density_clamp_max is None
    ):
        raise ValueError(
            "--propagated_density_clamp_max is required with "
            "--propagate_decoded_density_conditioning"
        )
    if args.propagated_density_base_resolution <= 0:
        raise ValueError("--propagated_density_base_resolution must be positive")
    if (
        args.propagated_density_target_mean is not None
        and not args.propagate_decoded_density_conditioning
    ):
        raise ValueError(
            "--propagated_density_target_mean requires "
            "--propagate_decoded_density_conditioning"
        )
    if (
        args.propagated_density_target_mean is not None
        and args.propagated_density_target_mean
        > args.propagated_density_clamp_max
    ):
        raise ValueError(
            "--propagated_density_target_mean cannot exceed "
            "--propagated_density_clamp_max"
        )
    if args.decoded_density_external_condition_base_resolution <= 0:
        raise ValueError(
            "--decoded_density_external_condition_base_resolution must be positive"
        )
    if args.drop_elongation_conditioning and args.constant_elongation_conditioning:
        raise ValueError(
            "Choose --drop_elongation_conditioning or --constant_elongation_conditioning, not both"
        )
    if args.density_guidance_strength is not None:
        if (
            not args.constant_density_conditioning
            and not args.density_payload_dir
            and args.decoded_density_external_condition_max is None
            and not args.propagate_decoded_density_conditioning
        ):
            raise ValueError(
                "--density_guidance_strength requires constant, decoded external, "
                "or propagated decoded density conditioning"
            )
        if args.base_guidance_strength not in (0.0, 1.0) or args.guidance_strength not in (0.0, 1.0):
            raise ValueError(
                "Density CFG requires low-resolution guidance strengths of 0 or 1"
            )
    if args.elongation_guidance_strength is not None:
        if not args.constant_elongation_conditioning:
            raise ValueError(
                "--elongation_guidance_strength requires --constant_elongation_conditioning"
            )
        if args.base_guidance_strength not in (0.0, 1.0) or args.guidance_strength not in (0.0, 1.0):
            raise ValueError(
                "Elongation-only CFG requires low-resolution guidance strengths of 0 or 1"
            )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mesh_dir = Path(args.mesh_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(run_dir)
    cfg["trainer"]["args"].pop("batch_size_per_gpu_by_high_resolution", None)
    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    stages = []
    low_resolution = args.start_resolution
    while low_resolution < 512:
        stages.append((low_resolution, low_resolution * 2))
        low_resolution *= 2
    target_resolutions = [high for _, high in stages]
    cascade_range = f"{target_resolutions[0]}to{target_resolutions[-1]}"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / (
        f"eval_mesh_folder_stage_repeat{args.stage_repeats}_cascade_{cascade_range}"
        f"_step{ckpt_step:07d}_cfg{args.guidance_strength:g}"
        f"{f'_densitycfg{args.density_guidance_strength:g}' if args.density_guidance_strength is not None else ''}"
        f"{'_densitypayload' if args.density_payload_dir else ''}"
        f"{f'_densitythrough{args.density_conditioning_max_resolution}' if args.density_conditioning_max_resolution is not None else ''}"
        f"{'_densitydrop' if args.drop_density_conditioning else ''}"
        f"{f'_decodedensitycondmax{args.decoded_density_external_condition_max:g}' if args.decoded_density_external_condition_max is not None else ''}"
        f"{'_decodedensitycondvoxelscale' if args.decoded_density_external_condition_scale_mode == 'voxel_size' else ''}"
        f"{f'_propdensityinit{args.propagated_density_initial_value:g}clamp{args.propagated_density_clamp_max:g}' if args.propagate_decoded_density_conditioning else ''}"
        f"{f'mean{args.propagated_density_target_mean:g}' if args.propagated_density_target_mean is not None else ''}"
        f"{f'_elong{args.constant_elongation_value:g}' if args.constant_elongation_conditioning else ''}"
        f"{f'_elongcfg{args.elongation_guidance_strength:g}' if args.elongation_guidance_strength is not None else ''}"
        f"{f'_elongthrough{args.elongation_conditioning_max_resolution}' if args.elongation_conditioning_max_resolution is not None else ''}"
        f"{'_elongdrop' if args.drop_elongation_conditioning else ''}"
        f"{'_nested_supports' if args.nested_supports else ''}"
        f"{f'_earlyrepeat{args.early_stage_repeats}through{args.early_stage_repeats_through_resolution}' if args.early_stage_repeats is not None else ''}"
        f"_basecfg{args.base_guidance_strength:g}_steps{args.steps}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.support_cache_dir).resolve() if args.support_cache_dir else output_dir / "support_cache"

    mesh_paths = collect_meshes(mesh_dir, args.recursive)
    if args.num_samples is not None:
        mesh_paths = mesh_paths[:args.num_samples]
    mesh_hashes = [mesh_sha1(path) for path in mesh_paths]

    supports = {resolution: [] for resolution in target_resolutions}
    for path in mesh_paths:
        if args.nested_supports:
            coords_by_resolution = {512: support_coords_from_mesh(path, 512, cache_dir)}
            for resolution in reversed(target_resolutions[:-1]):
                coords_by_resolution[resolution] = np.unique(
                    coords_by_resolution[resolution * 2] // 2,
                    axis=0,
                ).astype(np.int32)
        else:
            coords_by_resolution = {
                resolution: support_coords_from_mesh(path, resolution, cache_dir)
                for resolution in target_resolutions
            }
        for resolution, coords in coords_by_resolution.items():
            if coords.shape[0] == 0:
                raise ValueError(f"{path} produced empty support at resolution {resolution}")
            if coords.shape[0] > args.max_active_voxels:
                raise ValueError(
                    f"{path} has {coords.shape[0]} support voxels at {resolution}, "
                    f"above --max_active_voxels {args.max_active_voxels}"
                )
            supports[resolution].append(coords)

    density_by_resolution = None
    density_payload_dir = None
    if args.density_payload_dir:
        if not args.nested_supports:
            raise ValueError("--density_payload_dir requires --nested_supports")
        density_payload_dir = Path(args.density_payload_dir).resolve()
        density_by_resolution = load_multires_density_payloads(
            density_payload_dir,
            mesh_hashes,
            supports,
            target_resolutions,
        )

    trainer = build_trainer(cfg, SupportOnlyDataset(target_resolutions[0]), output_dir)
    apply_conditioning_augmentation_overrides(trainer, args)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)
    latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])
    if args.override_decoded_density:
        if not args.constant_density_conditioning:
            raise ValueError("--override_decoded_density requires --constant_density_conditioning")
        if getattr(trainer, "decoded_density_mode", "none") != "condition":
            raise ValueError(
                "--override_decoded_density requires trainer decoded_density_mode='condition'"
            )
    if args.decoded_density_external_condition_max is not None:
        if getattr(trainer, "decoded_density_mode", "none") != "state":
            raise ValueError(
                "--decoded_density_external_condition_max requires trainer "
                "decoded_density_mode='state'"
            )
        density_application = "decoded_density_clamped_external_condition"
    elif args.propagate_decoded_density_conditioning:
        if getattr(trainer, "decoded_density_mode", "none") != "state":
            raise ValueError(
                "--propagate_decoded_density_conditioning requires trainer "
                "decoded_density_mode='state'"
            )
        density_application = "propagated_final_decoded_density_condition"
    elif args.override_decoded_density:
        density_application = "decoded_density_override"
    elif args.density_payload_dir:
        density_application = "external_point_density_condition"
    else:
        density_application = "external_condition"
    print(f"Density application: {density_application}")

    images = {}
    density_mean_clamp_stats = []
    for start in range(0, len(mesh_paths), args.batch_size):
        end = min(start + args.batch_size, len(mesh_paths))
        previous_refined = None
        previous_density_condition = None
        for stage_idx, (low_res, high_res) in enumerate(stages):
            high_support = make_support_tensor(supports, high_res, start, end, trainer.device)
            visualizer = SupportOnlyDataset(high_res)
            add_visuals(images, visualizer, f"stage{low_res}to{high_res}_support", high_support)

            cond = high_support.replace(torch.zeros_like(high_support.feats)) if stage_idx == 0 else (
                sparse_condition_from_low_to_high(
                    triangle_field_channels(previous_refined),
                    high_support,
                    low_res,
                    high_res,
                )
            )
            guidance = args.base_guidance_strength if stage_idx == 0 else args.guidance_strength
            density_cond = None
            density_source_enabled = (
                args.constant_density_conditioning
                or args.density_payload_dir
                or args.decoded_density_external_condition_max is not None
                or args.propagate_decoded_density_conditioning
            )
            density_active = (
                density_source_enabled
                and (
                    args.density_conditioning_max_resolution is None
                    or high_res <= args.density_conditioning_max_resolution
                )
            )
            if args.density_payload_dir:
                if density_active:
                    density_cond = make_density_tensor(
                        density_by_resolution,
                        high_res,
                        start,
                        end,
                        high_support,
                    )
                else:
                    density_cond = constant_density_to_high(
                        high_support,
                        trainer.condition_drop_values["density"],
                        high_res,
                        high_res,
                        "none",
                    )
            elif args.propagate_decoded_density_conditioning:
                if stage_idx == 0:
                    density_cond = constant_density_to_high(
                        high_support,
                        args.propagated_density_initial_value,
                        args.propagated_density_base_resolution,
                        high_res,
                        "voxel_size",
                    )
                else:
                    if previous_density_condition is None:
                        raise RuntimeError("Missing propagated density from previous stage")
                    density_cond = sparse_condition_from_low_to_high(
                        previous_density_condition,
                        high_support,
                        low_res,
                        high_res,
                    )
                    density_cond = density_cond.replace(
                        density_cond.feats
                        - (2.0 * math.log(float(high_res) / float(low_res)))
                    )
            elif args.constant_density_conditioning:
                if density_active:
                    density_cond = constant_density_to_high(
                        high_support,
                        args.constant_density_value,
                        args.constant_density_base_resolution,
                        high_res,
                        args.constant_density_scale_mode,
                    )
                else:
                    density_cond = constant_density_to_high(
                        high_support,
                        trainer.condition_drop_values["density"],
                        high_res,
                        high_res,
                        "none",
                    )
            elif args.drop_density_conditioning:
                density_cond = constant_density_to_high(
                    high_support,
                    trainer.condition_drop_values["density"],
                    high_res,
                    high_res,
                    "none",
                )
            decoded_density_external_condition_max = (
                args.decoded_density_external_condition_max
                if density_active
                else None
            )
            if (
                decoded_density_external_condition_max is not None
                and args.decoded_density_external_condition_scale_mode == "voxel_size"
            ):
                decoded_density_external_condition_max -= 2.0 * math.log(
                    float(high_res)
                    / float(args.decoded_density_external_condition_base_resolution)
                )
            elongation_cond = None
            elongation_active = (
                args.constant_elongation_conditioning
                and (
                    args.elongation_conditioning_max_resolution is None
                    or high_res <= args.elongation_conditioning_max_resolution
                )
            )
            if args.constant_elongation_conditioning:
                elongation_cond = high_support.replace(torch.full(
                    (high_support.feats.shape[0], 1),
                    float(
                        args.constant_elongation_value
                        if elongation_active
                        else trainer.condition_drop_values["elongation"]
                    ),
                    dtype=high_support.feats.dtype,
                    device=high_support.device,
                ))
            elif args.drop_elongation_conditioning:
                elongation_cond = high_support.replace(torch.full(
                    (high_support.feats.shape[0], 1),
                    float(trainer.condition_drop_values["elongation"]),
                    dtype=high_support.feats.dtype,
                    device=high_support.device,
                ))
            density_guidance = args.density_guidance_strength if density_active else None
            elongation_guidance = args.elongation_guidance_strength if elongation_active else None
            always_dropped_condition_names = {
                name
                for name, dropped in (
                    (
                        "density",
                        args.drop_density_conditioning
                        or (density_source_enabled and not density_active),
                    ),
                    ("elongation", args.drop_elongation_conditioning),
                )
                if dropped
            }

            prefix = f"stage{low_res}to{high_res}"
            sample = None
            stage_repeats = (
                args.early_stage_repeats
                if (
                    args.early_stage_repeats is not None
                    and high_res <= args.early_stage_repeats_through_resolution
                )
                else args.stage_repeats
            )
            for repeat_idx in range(stage_repeats):
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
                    density_guidance,
                    elongation_cond,
                    elongation_guidance,
                    args.override_decoded_density,
                    always_dropped_condition_names,
                    decoded_density_external_condition_max,
                    high_resolution=high_res,
                )
                add_visuals(images, visualizer, f"{prefix}_iter{repeat_num}_cond", cond)
                add_visuals(images, visualizer, f"{prefix}_iter{repeat_num}_sample", sample)
                add_visuals(images, visualizer, f"{prefix}_iter{repeat_num}_pred_z0_last", pred_last)

                if args.propagate_decoded_density_conditioning:
                    if sample.feats.shape[1] != 3:
                        raise ValueError(
                            "Propagated density conditioning requires a three-channel "
                            f"decoder output, got {sample.feats.shape[1]}"
                        )
                    clamp_max = (
                        float(args.propagated_density_clamp_max)
                        - 2.0
                        * math.log(
                            float(high_res)
                            / float(args.propagated_density_base_resolution)
                        )
                    )
                    density_cond = sample.replace(sample.feats[:, 2:3])
                    if args.propagated_density_target_mean is None:
                        density_cond = density_cond.replace(
                            density_cond.feats.clamp(max=clamp_max)
                        )
                    else:
                        target_mean = (
                            float(args.propagated_density_target_mean)
                            - 2.0
                            * math.log(
                                float(high_res)
                                / float(args.propagated_density_base_resolution)
                            )
                        )
                        density_cond, clamp_stats = shift_mean_then_upper_clamp(
                            density_cond,
                            clamp_max,
                            target_mean,
                        )
                        for stat in clamp_stats:
                            stat.update(
                                {
                                    "mesh_index": start + stat["batch_index"],
                                    "resolution": high_res,
                                    "repeat": repeat_num,
                                    "target_mean": target_mean,
                                    "clamp_max": clamp_max,
                                }
                            )
                        density_mean_clamp_stats.extend(clamp_stats)

                if repeat_idx < stage_repeats - 1:
                    feedback_low = average_downsample_to_low(
                        triangle_field_channels(sample),
                        factor=2,
                    )
                    add_visuals(images, SupportOnlyDataset(low_res), f"{prefix}_iter{repeat_num}_avg_down_to_{low_res}", feedback_low)
                    cond = sparse_condition_from_low_to_high(feedback_low, high_support, low_res, high_res)
                    guidance = args.guidance_strength
            previous_refined = sample
            if args.propagate_decoded_density_conditioning:
                previous_density_condition = density_cond

    suffix = (
        f"step{ckpt_step:07d}_stage_repeat{args.stage_repeats}_{cascade_range}"
        f"_cfg{args.guidance_strength:g}_basecfg{args.base_guidance_strength:g}"
        f"{f'_densitycfg{args.density_guidance_strength:g}' if args.density_guidance_strength is not None else ''}"
        f"{'_densitypayload' if args.density_payload_dir else ''}"
        f"{f'_densitythrough{args.density_conditioning_max_resolution}' if args.density_conditioning_max_resolution is not None else ''}"
        f"{'_densitydrop' if args.drop_density_conditioning else ''}"
        f"{f'_decodedensitycondmax{args.decoded_density_external_condition_max:g}' if args.decoded_density_external_condition_max is not None else ''}"
        f"{'_decodedensitycondvoxelscale' if args.decoded_density_external_condition_scale_mode == 'voxel_size' else ''}"
        f"{f'_propdensityinit{args.propagated_density_initial_value:g}clamp{args.propagated_density_clamp_max:g}' if args.propagate_decoded_density_conditioning else ''}"
        f"{f'mean{args.propagated_density_target_mean:g}' if args.propagated_density_target_mean is not None else ''}"
        f"{f'_elong{args.constant_elongation_value:g}' if args.constant_elongation_conditioning else ''}"
        f"{f'_elongcfg{args.elongation_guidance_strength:g}' if args.elongation_guidance_strength is not None else ''}"
        f"{f'_elongthrough{args.elongation_conditioning_max_resolution}' if args.elongation_conditioning_max_resolution is not None else ''}"
        f"{f'_earlyrepeat{args.early_stage_repeats}through{args.early_stage_repeats_through_resolution}' if args.early_stage_repeats is not None else ''}"
    )
    for name, chunks in images.items():
        save_image_grid(torch.cat(chunks, dim=0)[:len(mesh_paths)], output_dir / f"{name}_{suffix}.jpg")

    density_mean_clamp_by_resolution = {}
    for resolution in target_resolutions:
        resolution_stats = [
            stat
            for stat in density_mean_clamp_stats
            if stat["resolution"] == resolution
        ]
        if not resolution_stats:
            continue
        density_mean_clamp_by_resolution[str(resolution)] = {
            "count": len(resolution_stats),
            "target_mean": resolution_stats[0]["target_mean"],
            "clamp_max": resolution_stats[0]["clamp_max"],
            "average_post_clamp_mean": float(
                np.mean([stat["post_clamp_mean"] for stat in resolution_stats])
            ),
            "average_mean_reduction": float(
                np.mean([stat["mean_reduction"] for stat in resolution_stats])
            ),
            "maximum_mean_reduction": float(
                np.max([stat["mean_reduction"] for stat in resolution_stats])
            ),
            "average_clipped_fraction": float(
                np.mean([stat["clipped_fraction"] for stat in resolution_stats])
            ),
        }
        aggregate = density_mean_clamp_by_resolution[str(resolution)]
        print(
            f"Density clamp mean effect at {resolution}: "
            f"target={aggregate['target_mean']:.6f}, "
            f"post={aggregate['average_post_clamp_mean']:.6f}, "
            f"mean reduction={aggregate['average_mean_reduction']:.6f} avg / "
            f"{aggregate['maximum_mean_reduction']:.6f} max, "
            f"clipped={aggregate['average_clipped_fraction']:.2%}"
        )

    summary = {
        "mesh_dir": str(mesh_dir),
        "meshes": [{"path": str(path), "sha1": sha} for path, sha in zip(mesh_paths, mesh_hashes)],
        "support_only": True,
        "support_extractor": (
            "trimesh 512 support; " + "; ".join(
                f"{resolution}=unique({resolution * 2}//2)"
                for resolution in reversed(target_resolutions[:-1])
            )
            if args.nested_supports
            else "trimesh.load(...).voxelized(pitch=1/resolution)"
        ),
        "nested_supports": args.nested_supports,
        "normalization": "bbox center, max extent scaled to 0.99999",
        "stages": stages,
        "stage_repeats": args.stage_repeats,
        "early_stage_repeats": args.early_stage_repeats,
        "early_stage_repeats_through_resolution": args.early_stage_repeats_through_resolution,
        "stage_repeats_by_stage": {
            f"{low_resolution}to{high_resolution}": (
                args.early_stage_repeats
                if (
                    args.early_stage_repeats is not None
                    and high_resolution <= args.early_stage_repeats_through_resolution
                )
                else args.stage_repeats
            )
            for low_resolution, high_resolution in stages
        },
        "steps": args.steps,
        "base_guidance_strength": args.base_guidance_strength,
        "guidance_strength": args.guidance_strength,
        "density_guidance_strength": args.density_guidance_strength,
        "density_conditioning_max_resolution": args.density_conditioning_max_resolution,
        "elongation_guidance_strength": args.elongation_guidance_strength,
        "elongation_conditioning_max_resolution": args.elongation_conditioning_max_resolution,
        "checkpoint": ckpt_path,
        "support_cache_dir": str(cache_dir),
        "constant_density_conditioning": args.constant_density_conditioning,
        "density_payload_dir": (
            str(density_payload_dir) if density_payload_dir is not None else None
        ),
        "density_payload_downsampling": (
            "mean 512 child densities on nested support, then +2*log(512/resolution)"
            if density_payload_dir is not None else None
        ),
        "constant_density_value": args.constant_density_value,
        "constant_density_base_resolution": args.constant_density_base_resolution,
        "constant_density_scale_mode": args.constant_density_scale_mode,
        "density_application": density_application,
        "decoded_density_external_condition_max": args.decoded_density_external_condition_max,
        "decoded_density_external_condition_base_resolution": (
            args.decoded_density_external_condition_base_resolution
        ),
        "decoded_density_external_condition_scale_mode": (
            args.decoded_density_external_condition_scale_mode
        ),
        "propagate_decoded_density_conditioning": (
            args.propagate_decoded_density_conditioning
        ),
        "propagated_density_initial_value": args.propagated_density_initial_value,
        "propagated_density_clamp_max": args.propagated_density_clamp_max,
        "propagated_density_target_mean": args.propagated_density_target_mean,
        "propagated_density_mean_constraint_order": "shift_then_upper_clamp",
        "density_mean_clamp_stats": density_mean_clamp_stats,
        "density_mean_clamp_by_resolution": density_mean_clamp_by_resolution,
        "propagated_density_base_resolution": (
            args.propagated_density_base_resolution
        ),
        "drop_density_conditioning": args.drop_density_conditioning,
        "density_drop_value": (
            float(trainer.condition_drop_values["density"])
            if args.drop_density_conditioning else None
        ),
        "constant_elongation_conditioning": args.constant_elongation_conditioning,
        "constant_elongation_value": args.constant_elongation_value,
        "drop_elongation_conditioning": args.drop_elongation_conditioning,
        "elongation_drop_value": float(trainer.condition_drop_values["elongation"]),
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
