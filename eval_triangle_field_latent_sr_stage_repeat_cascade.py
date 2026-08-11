import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision import utils as tv_utils

from trellis2.modules import sparse as sp
from trellis2.datasets.sparse_voxel_triangle_field import (
    EXTENDED_INPUT_LAYOUT,
    find_triangle_field_path,
    load_triangle_field_npz,
)
from trellis2.utils.data_utils import recursive_to_device

from eval_triangle_field_latent_sr_cascade import (
    collate_slice,
    restrict_datasets_to_same_instances,
    sparse_condition_from_low_to_high,
    visualize,
)
from eval_triangle_field_latent_sr_flow import (
    apply_conditioning_augmentation_overrides,
    build_dataset,
    build_support_latents,
    build_trainer,
    find_ckpt_step,
    load_config,
    load_encoder_checkpoint,
    sample_latent_sr,
)
from eval_metadata_filters import add_eval_metadata_filter_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a latent-space triangle-field SR cascade where each stage is "
            "sampled multiple times. Each generated sample at a stage is "
            "average-downsampled back to the conditioning resolution and used "
            "to rerun that same stage before advancing upward."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--base_guidance_strength", type=float, default=0.0)
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument(
        "--apply_conditioning_augmentation",
        action="store_true",
        help="Apply the trainer's configured conditioning augmentation to the positive conditioning path.",
    )
    parser.add_argument(
        "--conditioning_augmentation_noise_level",
        type=float,
        default=None,
        help="Eval-only override for conditioning_augmentation.noise_level.",
    )
    parser.add_argument(
        "--conditioning_augmentation_blur_sigma",
        type=float,
        default=None,
        help="Eval-only override/default for conditioning_augmentation.blur_sigma.",
    )
    parser.add_argument(
        "--conditioning_augmentation_disable_blur",
        action="store_true",
        help="Eval-only override that applies conditioning noise without sparse blur.",
    )
    parser.add_argument("--stage_repeats", type=int, default=2)
    parser.add_argument(
        "--stage0_cond_source",
        type=str,
        default="gt",
        choices=("gt", "zero", "dit256_downsample64"),
        help=(
            "Initial 64->128 conditioning source. 'zero' uses known support with zero field values; "
            "'dit256_downsample64' loads generated DiT 256 fields and average-downsamples to 64."
        ),
    )
    parser.add_argument(
        "--dit_generation_dir",
        type=str,
        default=None,
        help="Directory containing per-instance DiT generation folders with pred_triangle_fields.npz.",
    )
    parser.add_argument(
        "--dit_disable_distance_transform",
        action="store_true",
        help="Do not map DiT d_tri/d_vert fields from [0, 1] to [-1, 1] before SR conditioning.",
    )
    parser.add_argument(
        "--oracle_density_conditioning",
        action="store_true",
        help=(
            "Feed oracle density_conditioning to density-conditioned models. "
            "The base density field is loaded at --oracle_density_base_resolution and "
            "duplicated onto every stage's high-res support."
        ),
    )
    parser.add_argument("--oracle_density_base_resolution", type=int, default=128)
    parser.add_argument(
        "--oracle_density_voxel_dir",
        type=str,
        default=None,
        help=(
            "Directory containing base-resolution density-field voxels. Defaults to "
            "<root>/triangle_field_voxels_density_field/triangle_field_voxels_<base_resolution>."
        ),
    )
    parser.add_argument("--oracle_density_channel", type=str, default="density_field")
    parser.add_argument(
        "--oracle_density_scale_mode",
        type=str,
        default="none",
        choices=("none", "voxel_size"),
        help=(
            "'none' keeps duplicated density values unchanged. 'voxel_size' applies "
            "density_R = density_base - 2 * log(R / base_R), matching "
            "-log(area / voxel_size^2)."
        ),
    )
    parser.add_argument(
        "--constant_density_conditioning",
        action="store_true",
        help=(
            "Feed a constant density_cond to density-conditioned models. The value is "
            "defined at --constant_density_base_resolution and optionally scaled to "
            "each stage's high resolution."
        ),
    )
    parser.add_argument("--constant_density_value", type=float, default=-5.0)
    parser.add_argument("--constant_density_base_resolution", type=int, default=128)
    parser.add_argument(
        "--constant_density_scale_mode",
        type=str,
        default="voxel_size",
        choices=("none", "voxel_size"),
        help=(
            "'none' keeps the constant unchanged. 'voxel_size' applies "
            "density_R = density_base - 2 * log(R / base_R)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_resolution", type=int, default=None)
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.sqrt(images.shape[0])))
    tv_utils.save_image(images, str(path), nrow=nrow)


def add_visuals(images: dict, dataset, prefix: str, tensor: sp.SparseTensor) -> None:
    vis = visualize(dataset, tensor)
    for key, value in vis.items():
        images.setdefault(f"{prefix}_{key}", []).append(value.cpu())


def stage_args(args, low_resolution: int, high_resolution: int):
    return SimpleNamespace(
        low_resolution=low_resolution,
        high_resolution=high_resolution,
        latent_name=None,
        no_latents=True,
        split=args.split,
        instances=args.instances,
        render_resolution=args.render_resolution,
        metadata_filter_csv=args.metadata_filter_csv,
        no_train_duplicate_csv=args.no_train_duplicate_csv,
        triangle_filter_csv=args.triangle_filter_csv,
        disable_default_eval_filters=args.disable_default_eval_filters,
        disable_dataset_density_conditioning=True,
    )


def run_stage(
    trainer,
    data,
    cond: sp.SparseTensor,
    latent_channels: int,
    steps: int,
    guidance_strength: float,
    apply_conditioning_augmentation: bool,
    density_cond: sp.SparseTensor | None = None,
    density_guidance_strength: float | None = None,
    elongation_cond: sp.SparseTensor | None = None,
    elongation_guidance_strength: float | None = None,
    override_decoded_density: bool = False,
    always_dropped_condition_names: set[str] | None = None,
    decoded_density_external_condition_max: float | None = None,
    high_resolution: int | None = None,
    resolution_condition: int | float | None = None,
    density_statistics: torch.Tensor | None = None,
    density_stat_minimum_guidance_strength: float | None = None,
    density_stat_median_guidance_strength: float | None = None,
    density_stat_maximum_guidance_strength: float | None = None,
    shape_tokens: torch.Tensor | None = None,
    shape_guidance_strength: float | None = None,
):
    z_0, caches = build_support_latents(data["x_0"], latent_channels)
    sample_z, pred_z0_last = sample_latent_sr(
        trainer,
        z_0,
        cond,
        caches,
        None,
        steps,
        guidance_strength,
        apply_conditioning_augmentation,
        density_cond,
        density_guidance_strength,
        elongation_cond,
        elongation_guidance_strength,
        override_decoded_density,
        always_dropped_condition_names,
        decoded_density_external_condition_max,
        high_resolution=high_resolution,
        resolution_condition=resolution_condition,
        density_statistics=density_statistics,
        density_stat_minimum_guidance_strength=density_stat_minimum_guidance_strength,
        density_stat_median_guidance_strength=density_stat_median_guidance_strength,
        density_stat_maximum_guidance_strength=density_stat_maximum_guidance_strength,
        shape_tokens=shape_tokens,
        shape_guidance_strength=shape_guidance_strength,
    )
    sample = trainer._decode_latents_with_cache(sample_z, caches=caches)
    pred_last = trainer._decode_latents_with_cache(pred_z0_last, caches=caches)
    return sample, pred_last


def average_downsample_to_low(x_high: sp.SparseTensor, factor: int = 2) -> sp.SparseTensor:
    downsample = sp.SparseDownsample(factor, mode="mean").to(x_high.device)
    return downsample(x_high)


def support_zero_condition(data: dict) -> sp.SparseTensor:
    return data["x_0"].replace(torch.zeros_like(data["cond"].feats))


def _coord_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.long()
    return coords[:, 0] * (resolution * resolution) + coords[:, 1] * resolution + coords[:, 2]


def _load_density_field(
    density_dir: Path,
    sha256: str,
    channel: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if channel not in EXTENDED_INPUT_LAYOUT:
        raise ValueError(f"Unknown density channel {channel}; available: {sorted(EXTENDED_INPUT_LAYOUT)}")
    path = find_triangle_field_path(str(density_dir), sha256)
    with load_triangle_field_npz(path) as data:
        coords = torch.from_numpy(data["coords"].astype(np.int32, copy=False)).to(device=device)
        features = torch.from_numpy(data["features"].astype(np.float32, copy=False)).to(device=device)
    slc = EXTENDED_INPUT_LAYOUT[channel]
    if features.shape[1] < slc.stop:
        raise ValueError(f"{path} has {features.shape[1]} channels; {channel} requires {slc.stop}")
    return coords.int(), features[:, slc].float()


def oracle_density_from_base_to_high(
    density_dir: Path,
    sha256s: list[str],
    high_support: sp.SparseTensor,
    base_resolution: int,
    high_resolution: int,
    channel: str,
    scale_mode: str,
) -> sp.SparseTensor:
    if high_resolution % base_resolution != 0 and base_resolution % high_resolution != 0:
        raise ValueError(
            f"Oracle density currently expects integer resolution ratios, got "
            f"base={base_resolution}, high={high_resolution}"
        )

    density_feats = []
    density_coords = []
    missing = []
    for batch_idx, sha256 in enumerate(sha256s):
        base_coords, base_feats = _load_density_field(
            density_dir,
            sha256,
            channel,
            high_support.device,
        )
        high_layout = high_support.layout[batch_idx]
        high_coords = high_support.coords[high_layout].clone()
        high_spatial = high_coords[:, 1:].long()

        if high_resolution >= base_resolution:
            factor = high_resolution // base_resolution
            query_spatial = torch.div(high_spatial, factor, rounding_mode="floor")
        else:
            factor = base_resolution // high_resolution
            query_spatial = high_spatial * factor

        base_keys = _coord_keys(base_coords, base_resolution)
        query_keys = _coord_keys(query_spatial, base_resolution)
        order = torch.argsort(base_keys)
        sorted_keys = base_keys[order]
        sorted_feats = base_feats[order]
        idx = torch.searchsorted(sorted_keys, query_keys)
        valid = (
            (idx < sorted_keys.numel()) &
            (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == query_keys)
        )
        feats = torch.zeros(
            (high_spatial.shape[0], base_feats.shape[1]),
            dtype=base_feats.dtype,
            device=high_support.device,
        )
        if valid.any():
            feats[valid] = sorted_feats[idx[valid]]
        if scale_mode == "voxel_size":
            feats = feats - (2.0 * math.log(float(high_resolution) / float(base_resolution)))
        elif scale_mode != "none":
            raise ValueError(f"Unsupported oracle density scale mode: {scale_mode}")

        density_feats.append(feats)
        density_coords.append(high_coords)
        missing.append(1.0 - valid.float().mean().item())

    if any(value > 0 for value in missing):
        print(f"Warning: oracle density missed high-res support parents; max missing={max(missing):.6f}")
    return sp.SparseTensor(torch.cat(density_feats, dim=0), torch.cat(density_coords, dim=0))


def constant_density_to_high(
    high_support: sp.SparseTensor,
    value: float,
    base_resolution: int,
    high_resolution: int,
    scale_mode: str,
) -> sp.SparseTensor:
    density_value = float(value)
    if scale_mode == "voxel_size":
        density_value = density_value - (2.0 * math.log(float(high_resolution) / float(base_resolution)))
    elif scale_mode != "none":
        raise ValueError(f"Unsupported constant density scale mode: {scale_mode}")
    feats = torch.full(
        (high_support.feats.shape[0], 1),
        density_value,
        dtype=high_support.feats.dtype,
        device=high_support.device,
    )
    return high_support.replace(feats)


def find_dit_generation_path(generation_dir: Path, sha256: str) -> Path:
    matches = sorted(generation_dir.glob(f"*_{sha256}"))
    if not matches:
        raise FileNotFoundError(f"No DiT generation folder found for {sha256} under {generation_dir}")
    for folder in matches:
        pred = folder / "pred_triangle_fields.npz"
        if pred.exists():
            return pred
        pred = folder / "pred_decoded_dtri_dvert.npz"
        if pred.exists():
            return pred
    raise FileNotFoundError(f"No DiT prediction npz found for {sha256} under {matches[0]}")


def load_dit_256_batch(
    generation_dir: Path,
    sha256s: list[str],
    device: torch.device,
    apply_minus_one_one: bool,
) -> sp.SparseTensor:
    coords_parts = []
    feats_parts = []
    for batch_idx, sha256 in enumerate(sha256s):
        path = find_dit_generation_path(generation_dir, sha256)
        payload = np.load(path)
        coords_np = payload["coords"]
        if "features" in payload.files:
            feats_np = payload["features"]
        else:
            feats_np = payload["feats"]
        coords = torch.as_tensor(coords_np, device=device, dtype=torch.int32)
        batch = torch.full((coords.shape[0], 1), batch_idx, device=device, dtype=torch.int32)
        coords_parts.append(torch.cat([batch, coords], dim=1))
        feats = torch.as_tensor(feats_np, device=device, dtype=torch.float32)
        if apply_minus_one_one:
            feats = feats * 2.0 - 1.0
        feats_parts.append(feats)
    return sp.SparseTensor(torch.cat(feats_parts, dim=0), torch.cat(coords_parts, dim=0))


def load_dit_64_condition(
    generation_dir: Path,
    sha256s: list[str],
    high_support: sp.SparseTensor,
    apply_minus_one_one: bool,
) -> sp.SparseTensor:
    dit_256 = load_dit_256_batch(generation_dir, sha256s, high_support.device, apply_minus_one_one)
    dit_128 = average_downsample_to_low(dit_256, factor=2)
    dit_64 = average_downsample_to_low(dit_128, factor=2)
    return sparse_condition_from_low_to_high(dit_64, high_support, 64, 128)


def main():
    args = parse_args()
    if args.stage_repeats < 1:
        raise ValueError(f"--stage_repeats must be at least 1, got {args.stage_repeats}")
    if args.oracle_density_conditioning and args.constant_density_conditioning:
        raise ValueError("Choose either --oracle_density_conditioning or --constant_density_conditioning, not both.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    root = Path(args.root).resolve()
    cfg = load_config(run_dir)
    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / (
            f"eval_filtered_{args.split}_stage_repeat{args.stage_repeats}_cascade_128to512_support_only"
            f"_step{ckpt_step:07d}_cfg{args.guidance_strength:g}"
            f"_basecfg{args.base_guidance_strength:g}_steps{args.steps}_n{args.num_samples}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage0_cond_source == "dit256_downsample64" and args.dit_generation_dir is None:
        raise ValueError("--dit_generation_dir is required when --stage0_cond_source=dit256_downsample64")
    dit_generation_dir = Path(args.dit_generation_dir).resolve() if args.dit_generation_dir is not None else None
    oracle_density_dir = None
    if args.oracle_density_conditioning:
        oracle_density_dir = (
            Path(args.oracle_density_voxel_dir).resolve()
            if args.oracle_density_voxel_dir is not None
            else root / "triangle_field_voxels_density_field" / f"triangle_field_voxels_{args.oracle_density_base_resolution}"
        )
        if not (oracle_density_dir / "metadata.csv").exists():
            raise FileNotFoundError(f"Oracle density metadata not found: {oracle_density_dir / 'metadata.csv'}")

    stages = [(64, 128), (128, 256), (256, 512)]
    viz_stages = [(32, 64)] + stages
    datasets = {
        stage: build_dataset(cfg, root, stage_args(args, *stage))[0]
        for stage in viz_stages
    }
    if args.instances is None:
        selected = restrict_datasets_to_same_instances(
            [datasets[stage] for stage in viz_stages],
            args.seed,
            args.num_samples,
        )
    else:
        selected = [sha256 for _, sha256 in datasets[viz_stages[0]].instances[:args.num_samples]]
        for stage in viz_stages[1:]:
            current = [sha256 for _, sha256 in datasets[stage].instances[:args.num_samples]]
            if current != selected:
                raise ValueError("Explicit instance ordering differs across repeat cascade stage datasets.")
    low_res_viz_dataset = {
        64: datasets[(32, 64)],
        128: datasets[(64, 128)],
        256: datasets[(128, 256)],
    }

    trainer = build_trainer(cfg, datasets[(64, 128)], output_dir)
    apply_conditioning_augmentation_overrides(trainer, args)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)
    latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])

    images = {}
    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        previous_refined = None

        for stage_idx, (low_res, high_res) in enumerate(stages):
            dataset = datasets[(low_res, high_res)]
            data = recursive_to_device(collate_slice(dataset, start, end), trainer.device)

            if stage_idx == 0:
                batch_sha256s = selected[start:end]
                if args.stage0_cond_source == "gt":
                    initial_cond = data["cond"]
                elif args.stage0_cond_source == "zero":
                    initial_cond = support_zero_condition(data)
                elif args.stage0_cond_source == "dit256_downsample64":
                    initial_cond = load_dit_64_condition(
                        dit_generation_dir,
                        batch_sha256s,
                        data["x_0"],
                        not args.dit_disable_distance_transform,
                    )
                else:
                    raise ValueError(f"Unsupported stage0_cond_source: {args.stage0_cond_source}")
                initial_guidance = args.base_guidance_strength
            else:
                initial_cond = sparse_condition_from_low_to_high(
                    previous_refined, data["x_0"], low_res, high_res
                )
                initial_guidance = args.guidance_strength

            prefix = f"stage{low_res}to{high_res}"
            add_visuals(images, dataset, f"{prefix}_gt", data["x_0"])
            cond = initial_cond
            guidance = initial_guidance
            sample = None
            density_cond = None
            if args.oracle_density_conditioning:
                density_cond = oracle_density_from_base_to_high(
                    oracle_density_dir,
                    selected[start:end],
                    data["x_0"],
                    args.oracle_density_base_resolution,
                    high_res,
                    args.oracle_density_channel,
                    args.oracle_density_scale_mode,
                )
            elif args.constant_density_conditioning:
                density_cond = constant_density_to_high(
                    data["x_0"],
                    args.constant_density_value,
                    args.constant_density_base_resolution,
                    high_res,
                    args.constant_density_scale_mode,
                )
            for repeat_idx in range(args.stage_repeats):
                repeat_num = repeat_idx + 1
                sample, pred_last = run_stage(
                    trainer,
                    data,
                    cond,
                    latent_channels,
                    args.steps,
                    guidance,
                    args.apply_conditioning_augmentation,
                    density_cond,
                    high_resolution=high_res,
                )
                add_visuals(images, dataset, f"{prefix}_iter{repeat_num}_cond", cond)
                add_visuals(images, dataset, f"{prefix}_iter{repeat_num}_sample", sample)
                add_visuals(images, dataset, f"{prefix}_iter{repeat_num}_pred_z0_last", pred_last)

                if repeat_idx < args.stage_repeats - 1:
                    feedback_low = average_downsample_to_low(sample, factor=2)
                    add_visuals(
                        images,
                        low_res_viz_dataset[low_res],
                        f"{prefix}_iter{repeat_num}_avg_down_to_{low_res}",
                        feedback_low,
                    )
                    cond = sparse_condition_from_low_to_high(
                        feedback_low, data["x_0"], low_res, high_res
                    )
                    guidance = args.guidance_strength

            previous_refined = sample

    suffix = (
        f"step{ckpt_step:07d}_stage_repeat{args.stage_repeats}_128to512"
        f"_cfg{args.guidance_strength:g}_basecfg{args.base_guidance_strength:g}"
    )
    for name, chunks in images.items():
        save_image_grid(torch.cat(chunks, dim=0)[:args.num_samples], output_dir / f"{name}_{suffix}.jpg")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "root": str(root),
        "split": args.split,
        "selected_instances": selected,
        "stages": stages,
        "stage0_cond_source": args.stage0_cond_source,
        "dit_generation_dir": str(dit_generation_dir) if dit_generation_dir is not None else None,
        "dit_distance_transform": "minus_one_one" if not args.dit_disable_distance_transform else "none",
        "oracle_density_conditioning": args.oracle_density_conditioning,
        "oracle_density_voxel_dir": str(oracle_density_dir) if oracle_density_dir is not None else None,
        "oracle_density_base_resolution": args.oracle_density_base_resolution,
        "oracle_density_channel": args.oracle_density_channel,
        "oracle_density_scale_mode": args.oracle_density_scale_mode,
        "oracle_density_formula": (
            "density_R = density_base"
            if args.oracle_density_scale_mode == "none"
            else "density_R = density_base - 2 * log(R / base_R)"
        ),
        "constant_density_conditioning": args.constant_density_conditioning,
        "constant_density_value": args.constant_density_value,
        "constant_density_base_resolution": args.constant_density_base_resolution,
        "constant_density_scale_mode": args.constant_density_scale_mode,
        "constant_density_formula": (
            "density_R = density_base"
            if args.constant_density_scale_mode == "none"
            else "density_R = density_base - 2 * log(R / base_R)"
        ),
        "stage_repeat": "sample -> average downsample by 2 -> rerun same stage",
        "stage_repeats": args.stage_repeats,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "base_guidance_strength": args.base_guidance_strength,
        "guidance_strength": args.guidance_strength,
        "apply_conditioning_augmentation": args.apply_conditioning_augmentation,
        "conditioning_augmentation": getattr(trainer, "conditioning_augmentation", None),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
