import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm import tqdm

from eval_triangle_field_latent_sr_flow import (
    append_visuals,
    build_dataset,
    build_trainer,
    find_ckpt_step,
    load_config,
    load_encoder_checkpoint,
    predict_z0,
    slice_batch,
)
from trellis2.modules import sparse as sp
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-condition unconditional sampling for latent triangle-field SR flow."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--low_resolution", type=int, default=256)
    parser.add_argument("--high_resolution", type=int, default=512)
    parser.add_argument(
        "--latent_name",
        type=str,
        default=None,
        help="Latent directory name under <root>/triangle_field_latents.",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Optional renderer resolution override for saved grids.",
    )
    return parser.parse_args()


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.sqrt(images.shape[0])))
    tv_utils.save_image(images, str(path), nrow=nrow)


@torch.no_grad()
def sample_unconditional_latent_sr(
    trainer,
    z_0: sp.SparseTensor,
    cond: sp.SparseTensor,
    caches,
    cache_paths,
    steps: int,
):
    z_t = z_0.replace(torch.randn_like(z_0.feats))
    zero_cond = cond.replace(torch.zeros_like(cond.feats))
    t_seq = np.linspace(1.0, 0.0, steps + 1).tolist()
    pred_z0_last = None

    for t, t_prev in tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc="Sampling zero-cond latent SR"):
        pred_z0 = predict_z0(trainer, z_t, zero_cond, caches, cache_paths, float(t))
        velocity = (z_t.feats - pred_z0.feats) / max(float(t), 1e-5)
        z_t = z_t.replace(z_t.feats - (float(t) - float(t_prev)) * velocity)
        pred_z0_last = pred_z0

    return z_t, zero_cond, pred_z0_last


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
        else run_dir
        / f"eval_uncond_{args.split}_{args.low_resolution}to{args.high_resolution}_latent_flow_step{ckpt_step:07d}_n{args.num_samples}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, data_dir = build_dataset(cfg, root, args)
    trainer = build_trainer(cfg, dataset, output_dir)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    images = {}
    remaining = args.num_samples
    with torch.no_grad():
        for data in loader:
            batch = min(remaining, args.batch_size)
            data = recursive_to_device(slice_batch(data, batch), trainer.device)
            caches = data.get("triangle_field_slat_cache", None)
            cache_paths = data.get("triangle_field_slat_cache_path", None)

            sample_z, zero_cond, pred_z0_last = sample_unconditional_latent_sr(
                trainer,
                data["z_0"],
                data["cond"],
                caches,
                cache_paths,
                args.steps,
            )
            gt = trainer._decode_latents_with_cache(data["z_0"], caches=caches, cache_paths=cache_paths)
            sample = trainer._decode_latents_with_cache(sample_z, caches=caches, cache_paths=cache_paths)
            pred_last = trainer._decode_latents_with_cache(pred_z0_last, caches=caches, cache_paths=cache_paths)

            append_visuals(dataset, images, "gt", gt, batch)
            append_visuals(dataset, images, "cond", data["cond"], batch)
            append_visuals(dataset, images, "zero_cond", zero_cond, batch)
            append_visuals(dataset, images, "sample", sample, batch)
            append_visuals(dataset, images, "pred_z0_last", pred_last, batch)

            remaining -= batch
            if remaining <= 0:
                break

    suffix = f"step{ckpt_step:07d}_uncond_{args.low_resolution}to{args.high_resolution}"
    for name, chunks in images.items():
        if chunks:
            save_image_grid(torch.cat(chunks, dim=0)[: args.num_samples], output_dir / f"{name}_{suffix}.jpg")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "root": str(root),
        "split": args.split,
        "data_dir": data_dir,
        "dataset_size": len(dataset),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "steps": args.steps,
        "low_resolution": args.low_resolution,
        "high_resolution": args.high_resolution,
        "latent_name": args.latent_name,
        "conditioning": "zeros_like_cond",
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
