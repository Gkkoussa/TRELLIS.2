import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision import utils as tv_utils

from trellis2.modules import sparse as sp
from trellis2.utils.data_utils import recursive_to_device

from eval_triangle_field_latent_sr_cascade import (
    collate_slice,
    restrict_datasets_to_same_instances,
    sparse_condition_from_low_to_high,
    visualize,
)
from eval_triangle_field_latent_sr_flow import (
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
    parser.add_argument("--stage_repeats", type=int, default=2)
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
    )


def run_stage(
    trainer,
    data,
    cond: sp.SparseTensor,
    latent_channels: int,
    steps: int,
    guidance_strength: float,
    apply_conditioning_augmentation: bool,
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
    )
    sample = trainer._decode_latents_with_cache(sample_z, caches=caches)
    pred_last = trainer._decode_latents_with_cache(pred_z0_last, caches=caches)
    return sample, pred_last


def average_downsample_to_low(x_high: sp.SparseTensor, factor: int = 2) -> sp.SparseTensor:
    downsample = sp.SparseDownsample(factor, mode="mean").to(x_high.device)
    return downsample(x_high)


def main():
    args = parse_args()
    if args.stage_repeats < 1:
        raise ValueError(f"--stage_repeats must be at least 1, got {args.stage_repeats}")
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
                initial_cond = data["cond"]
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
