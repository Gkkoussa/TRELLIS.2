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
        description="Cascade latent-space triangle-field SR flow from 64 to 512."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--base_guidance_strength", type=float, default=0.0)
    parser.add_argument("--guidance_strength", type=float, default=3.0)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_resolution", type=int, default=None)
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.sqrt(images.shape[0])))
    tv_utils.save_image(images, str(path), nrow=nrow)


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


def restrict_datasets_to_same_instances(datasets, seed: int, num_samples: int):
    instance_sets = [
        {sha256 for _, sha256 in dataset.instances}
        for dataset in datasets
    ]
    common = set.intersection(*instance_sets)
    if len(common) < num_samples:
        raise ValueError(f"Only {len(common)} shared instances are available, requested {num_samples}")

    ordered = sorted(common)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ordered), generator=generator).tolist()
    selected = [ordered[i] for i in perm[:num_samples]]
    selected_set = set(selected)
    selected_index = {sha256: i for i, sha256 in enumerate(selected)}

    for dataset in datasets:
        filtered = [(root, sha256) for root, sha256 in dataset.instances if sha256 in selected_set]
        filtered.sort(key=lambda item: selected_index[item[1]])
        dataset.instances = filtered
        if len(dataset.metadata) > 0:
            dataset.metadata = dataset.metadata[dataset.metadata.index.isin(selected_set)]
        if hasattr(dataset, "loads"):
            dataset.loads = [
                dataset.metadata.loc[sha256, dataset.num_voxels_column]
                if dataset.num_voxels_column in dataset.metadata.columns else 1
                for _, sha256 in dataset.instances
            ]
    return selected


def collate_slice(dataset, start: int, end: int):
    batch = [dataset[i] for i in range(start, end)]
    return dataset.collate_fn(batch)


def sparse_condition_from_low_to_high(
    low: sp.SparseTensor,
    high: sp.SparseTensor,
    low_resolution: int,
    high_resolution: int,
) -> sp.SparseTensor:
    if high_resolution % low_resolution != 0:
        raise ValueError(f"high_resolution must be divisible by low_resolution: {high_resolution}/{low_resolution}")
    factor = high_resolution // low_resolution
    cond_feats = []
    cond_coords = []
    missing = []
    missing_counts = []
    for batch_idx, high_layout in enumerate(high.layout):
        high_coords = high.coords[high_layout].clone()
        high_spatial = high_coords[:, 1:].long()
        parent_spatial = torch.div(high_spatial, factor, rounding_mode="floor")

        low_layout = low.layout[batch_idx]
        low_spatial = low.coords[low_layout][:, 1:].long()
        low_feats = low.feats[low_layout]

        low_keys = (
            low_spatial[:, 0] * (low_resolution * low_resolution) +
            low_spatial[:, 1] * low_resolution +
            low_spatial[:, 2]
        )
        parent_keys = (
            parent_spatial[:, 0] * (low_resolution * low_resolution) +
            parent_spatial[:, 1] * low_resolution +
            parent_spatial[:, 2]
        )
        order = torch.argsort(low_keys)
        sorted_keys = low_keys[order]
        sorted_feats = low_feats[order]
        idx = torch.searchsorted(sorted_keys, parent_keys)
        valid = (
            (idx < sorted_keys.numel()) &
            (sorted_keys[idx.clamp_max(sorted_keys.numel() - 1)] == parent_keys)
        )
        feats = torch.zeros((high_spatial.shape[0], low_feats.shape[1]), device=high.device, dtype=low_feats.dtype)
        if valid.any():
            feats[valid] = sorted_feats[idx[valid]]
        cond_feats.append(feats)
        cond_coords.append(high_coords)
        missing_count = int((~valid).sum().item())
        missing_counts.append(missing_count)
        missing.append(missing_count / max(1, valid.numel()))

    if any(missing_counts):
        print(
            "Warning: generated low-res conditioning missed high-res parents; "
            f"max missing={max(missing):.9f} ({max(missing_counts)} voxels)"
        )
    return sp.SparseTensor(torch.cat(cond_feats, dim=0), torch.cat(cond_coords, dim=0))


def visualize(dataset, tensor: sp.SparseTensor):
    return dataset.visualize_sample({"target": tensor})


def main():
    args = parse_args()
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
            f"eval_filtered_{args.split}_cascade_64to512_support_only"
            f"_step{ckpt_step:07d}_cfg{args.guidance_strength:g}_basecfg{args.base_guidance_strength:g}"
            f"_steps{args.steps}_n{args.num_samples}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = [(32, 64), (64, 128), (128, 256), (256, 512)]
    datasets = [
        build_dataset(cfg, root, stage_args(args, low, high))[0]
        for low, high in stages
    ]
    if args.instances is None:
        selected = restrict_datasets_to_same_instances(datasets, args.seed, args.num_samples)
    else:
        selected = [sha256 for _, sha256 in datasets[0].instances[:args.num_samples]]
        for dataset in datasets[1:]:
            current = [sha256 for _, sha256 in dataset.instances[:args.num_samples]]
            if current != selected:
                raise ValueError("Explicit instance ordering differs across cascade stage datasets.")

    trainer = build_trainer(cfg, datasets[0], output_dir)
    apply_conditioning_augmentation_overrides(trainer, args)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)
    latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])

    images = {}
    final_samples = []
    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        previous_sample = None
        for stage_idx, ((low_res, high_res), dataset) in enumerate(zip(stages, datasets)):
            data = recursive_to_device(collate_slice(dataset, start, end), trainer.device)
            z_0, caches = build_support_latents(data["x_0"], latent_channels)
            if stage_idx == 0:
                cond = data["cond"]
                guidance = args.base_guidance_strength
            else:
                cond = sparse_condition_from_low_to_high(previous_sample, data["x_0"], low_res, high_res)
                guidance = args.guidance_strength

            sample_z, pred_z0_last = sample_latent_sr(
                trainer,
                z_0,
                cond,
                caches,
                None,
                args.steps,
                guidance,
                args.apply_conditioning_augmentation,
                high_resolution=high_res,
            )
            sample = trainer._decode_latents_with_cache(sample_z, caches=caches)
            pred_last = trainer._decode_latents_with_cache(pred_z0_last, caches=caches)

            for prefix, tensor in (
                (f"gt_{high_res}", data["x_0"]),
                (f"cond_{high_res}", cond),
                (f"sample_{high_res}", sample),
                (f"pred_z0_last_{high_res}", pred_last),
            ):
                vis = visualize(dataset, tensor)
                for key, value in vis.items():
                    images.setdefault(f"{prefix}_{key}", []).append(value.cpu())

            previous_sample = sample
            if high_res == 512:
                final_samples.append(sample)

    suffix = f"step{ckpt_step:07d}_cascade64to512_cfg{args.guidance_strength:g}_basecfg{args.base_guidance_strength:g}"
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
