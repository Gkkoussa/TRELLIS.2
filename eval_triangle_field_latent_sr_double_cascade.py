import argparse
import copy
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
            "Run an iterated latent-space triangle-field SR cascade. Pass 1 "
            "generates 64->512 from an unconditional 64 stage. Each following "
            "pass average-pools the previous generated 512 field back to 64 "
            "and uses it as conditioning for another 64->512 cascade."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--base_guidance_strength", type=float, default=0.0)
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument("--num_passes", type=int, default=2)
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


def clone_stage_args(args, low_resolution: int, high_resolution: int):
    copied = copy.copy(args)
    return SimpleNamespace(
        low_resolution=low_resolution,
        high_resolution=high_resolution,
        latent_name=None,
        no_latents=True,
        split=copied.split,
        render_resolution=copied.render_resolution,
        metadata_filter_csv=copied.metadata_filter_csv,
        no_train_duplicate_csv=copied.no_train_duplicate_csv,
        triangle_filter_csv=copied.triangle_filter_csv,
        disable_default_eval_filters=copied.disable_default_eval_filters,
    )


def run_stage(
    trainer,
    dataset,
    data,
    cond: sp.SparseTensor,
    latent_channels: int,
    steps: int,
    guidance_strength: float,
    high_resolution: int,
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
        high_resolution=high_resolution,
    )
    sample = trainer._decode_latents_with_cache(sample_z, caches=caches)
    pred_last = trainer._decode_latents_with_cache(pred_z0_last, caches=caches)
    return sample, pred_last


def average_downsample_512_to_64(x_512: sp.SparseTensor) -> sp.SparseTensor:
    downsample = sp.SparseDownsample(8, mode="mean").to(x_512.device)
    return downsample(x_512)


def main():
    args = parse_args()
    if args.num_passes < 1:
        raise ValueError(f"--num_passes must be at least 1, got {args.num_passes}")
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
            f"eval_filtered_{args.split}_iter{args.num_passes}_cascade_64to512_support_only"
            f"_step{ckpt_step:07d}_cfg{args.guidance_strength:g}"
            f"_basecfg{args.base_guidance_strength:g}_steps{args.steps}_n{args.num_samples}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pass1_stages = [(32, 64), (64, 128), (128, 256), (256, 512)]
    feedback_stages = [(64, 128), (128, 256), (256, 512)]
    all_stage_pairs = pass1_stages

    datasets = {
        stage: build_dataset(cfg, root, clone_stage_args(args, *stage))[0]
        for stage in all_stage_pairs
    }
    selected = restrict_datasets_to_same_instances(
        [datasets[stage] for stage in all_stage_pairs],
        args.seed,
        args.num_samples,
    )

    trainer = build_trainer(cfg, datasets[(32, 64)], output_dir)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)
    latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])

    images = {}

    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)

        feedback_64 = None
        for pass_idx in range(args.num_passes):
            pass_num = pass_idx + 1
            stages = pass1_stages if pass_idx == 0 else feedback_stages
            previous_sample = None
            final_512 = None

            for stage_idx, (low_res, high_res) in enumerate(stages):
                dataset = datasets[(low_res, high_res)]
                data = recursive_to_device(collate_slice(dataset, start, end), trainer.device)
                if pass_idx == 0 and stage_idx == 0:
                    cond = data["cond"]
                    guidance = args.base_guidance_strength
                elif stage_idx == 0:
                    cond = sparse_condition_from_low_to_high(feedback_64, data["x_0"], low_res, high_res)
                    guidance = args.guidance_strength
                else:
                    cond = sparse_condition_from_low_to_high(previous_sample, data["x_0"], low_res, high_res)
                    guidance = args.guidance_strength

                sample, pred_last = run_stage(
                    trainer,
                    dataset,
                    data,
                    cond,
                    latent_channels,
                    args.steps,
                    guidance,
                    high_res,
                )
                add_visuals(images, dataset, f"pass{pass_num}_gt_{high_res}", data["x_0"])
                add_visuals(images, dataset, f"pass{pass_num}_cond_{high_res}", cond)
                add_visuals(images, dataset, f"pass{pass_num}_sample_{high_res}", sample)
                add_visuals(images, dataset, f"pass{pass_num}_pred_z0_last_{high_res}", pred_last)

                previous_sample = sample
                if high_res == 512:
                    final_512 = sample

            if pass_idx < args.num_passes - 1:
                feedback_64 = average_downsample_512_to_64(final_512)
                add_visuals(images, datasets[(32, 64)], f"feedback_pass{pass_num}_avg512_to64", feedback_64)

    suffix = (
        f"step{ckpt_step:07d}_iter{args.num_passes}cascade64to512"
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
        "pass1_stages": pass1_stages,
        "feedback": "average_downsample_previous_pass_512_to_64",
        "feedback_stages": feedback_stages,
        "num_passes": args.num_passes,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "base_guidance_strength": args.base_guidance_strength,
        "guidance_strength": args.guidance_strength,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
