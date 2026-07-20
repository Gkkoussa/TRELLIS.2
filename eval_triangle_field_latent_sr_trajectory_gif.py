import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from trellis2.modules import sparse as sp
from trellis2.utils.data_utils import recursive_to_device

from eval_triangle_field_latent_sr_flow import (
    build_dataset,
    build_support_latents,
    build_trainer,
    find_ckpt_step,
    load_config,
    load_encoder_checkpoint,
    predict_z0,
    slice_batch,
)
from eval_metadata_filters import add_eval_metadata_filter_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a GIF of latent SR sampling decoded into triangle-field d_tri frames."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--low_resolution", type=int, default=256)
    parser.add_argument("--high_resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--duration_ms", type=int, default=160)
    parser.add_argument("--panel_width", type=int, default=512)
    parser.add_argument("--render_resolution", type=int, default=None)
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def tensor_to_pil(image: torch.Tensor, width: int) -> Image.Image:
    image = image.detach().float().cpu().clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(image)
    if width > 0 and pil.width != width:
        height = max(1, round(pil.height * width / pil.width))
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return pil


def make_frame(input_tri: torch.Tensor, output_tri: torch.Tensor, step: int, total_steps: int, t: float, panel_width: int):
    left = tensor_to_pil(input_tri, panel_width)
    right = tensor_to_pil(output_tri, panel_width)
    header_h = 34
    frame = Image.new("RGB", (left.width + right.width, left.height + header_h), "white")
    frame.paste(left, (0, header_h))
    frame.paste(right, (left.width, header_h))

    draw = ImageDraw.Draw(frame)
    draw.text((8, 9), f"step {step:02d}/{total_steps}  t={t:.3f}", fill=(0, 0, 0))
    draw.text((left.width // 2 - 70, 9), "decoded z_t input d_tri", fill=(0, 0, 0))
    draw.text((left.width + right.width // 2 - 84, 9), "decoded pred z0 output d_tri", fill=(0, 0, 0))
    draw.line((left.width, 0, left.width, frame.height), fill=(0, 0, 0), width=2)
    return frame


def tensor_l1_delta(current: torch.Tensor | None, previous: torch.Tensor | None) -> float | None:
    if current is None or previous is None:
        return None
    return float((current.float() - previous.float()).abs().mean().item())


def restrict_dataset_to_instances(dataset, instances_path: str):
    with open(instances_path, "r") as f:
        selected = [line.strip() for line in f if line.strip()]
    selected_set = set(selected)
    selected_index = {sha256: idx for idx, sha256 in enumerate(selected)}
    filtered = [(root, sha256) for root, sha256 in dataset.instances if sha256 in selected_set]
    filtered.sort(key=lambda item: selected_index[item[1]])
    if len(filtered) != len(selected):
        found = {sha256 for _, sha256 in filtered}
        missing = [sha256 for sha256 in selected if sha256 not in found]
        raise ValueError(f"Only found {len(filtered)}/{len(selected)} requested instances; first missing: {missing[:5]}")
    dataset.instances = filtered
    if len(dataset.metadata) > 0:
        dataset.metadata = dataset.metadata[dataset.metadata.index.astype(str).isin(selected_set)]
    if hasattr(dataset, "loads"):
        dataset.loads = [
            dataset.metadata.loc[sha256, dataset.num_voxels_column]
            if dataset.num_voxels_column in dataset.metadata.columns else 1
            for _, sha256 in dataset.instances
        ]
    return selected


@torch.no_grad()
def sample_trajectory(
    trainer,
    dataset,
    z_0: sp.SparseTensor,
    cond: sp.SparseTensor,
    caches,
    cache_paths,
    steps: int,
    guidance_strength: float,
    panel_width: int,
):
    z_t = z_0.replace(torch.randn_like(z_0.feats))
    zero_cond = cond.replace(torch.zeros_like(cond.feats))
    t_seq = np.linspace(1.0, 0.0, steps + 1).tolist()
    frames = []
    stats = []
    final_sample = None
    final_pred = None
    prev_z_t_feats = None
    prev_pred_z0_feats = None
    prev_input_d_tri = None
    prev_output_d_tri = None

    for step_idx, (t, t_prev) in enumerate(tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc="Sampling/rendering trajectory")):
        x_t = trainer._decode_latents_with_cache(z_t, caches=caches, cache_paths=cache_paths)
        if guidance_strength == 0.0:
            pred_z0 = predict_z0(trainer, z_t, zero_cond, caches, cache_paths, float(t))
        else:
            pred_pos = predict_z0(trainer, z_t, cond, caches, cache_paths, float(t))
            if guidance_strength == 1.0:
                pred_z0 = pred_pos
            else:
                pred_neg = predict_z0(trainer, z_t, zero_cond, caches, cache_paths, float(t))
                pred_z0 = pred_pos.replace(
                    guidance_strength * pred_pos.feats + (1.0 - guidance_strength) * pred_neg.feats
                )
        y = trainer._decode_latents_with_cache(pred_z0, caches=caches, cache_paths=cache_paths)

        input_vis = dataset.visualize_sample({"target": x_t})["d_tri"][0]
        output_vis = dataset.visualize_sample({"target": y})["d_tri"][0]
        frames.append(make_frame(input_vis, output_vis, step_idx, steps, float(t), panel_width))
        input_d_tri = x_t.feats[:, :1].detach()
        output_d_tri = y.feats[:, :1].detach()

        stats.append({
            "step": step_idx,
            "t": float(t),
            "t_prev": float(t_prev),
            "z_t_abs_mean": float(z_t.feats.float().abs().mean().item()),
            "pred_z0_abs_mean": float(pred_z0.feats.float().abs().mean().item()),
            "decoded_input_d_tri_abs_mean": float(input_d_tri.float().abs().mean().item()),
            "decoded_pred_d_tri_abs_mean": float(output_d_tri.float().abs().mean().item()),
            "z_t_l1_delta_from_prev": tensor_l1_delta(z_t.feats, prev_z_t_feats),
            "pred_z0_l1_delta_from_prev": tensor_l1_delta(pred_z0.feats, prev_pred_z0_feats),
            "decoded_input_d_tri_l1_delta_from_prev": tensor_l1_delta(input_d_tri, prev_input_d_tri),
            "decoded_pred_d_tri_l1_delta_from_prev": tensor_l1_delta(output_d_tri, prev_output_d_tri),
        })
        prev_z_t_feats = z_t.feats.detach().clone()
        prev_pred_z0_feats = pred_z0.feats.detach().clone()
        prev_input_d_tri = input_d_tri.clone()
        prev_output_d_tri = output_d_tri.clone()

        velocity = (z_t.feats - pred_z0.feats) / max(float(t), 1e-5)
        z_t = z_t.replace(z_t.feats - (float(t) - float(t_prev)) * velocity)
        final_sample = z_t
        final_pred = pred_z0

    # Include the final decoded sample as a last, steady frame.
    x_final = trainer._decode_latents_with_cache(final_sample, caches=caches, cache_paths=cache_paths)
    y_final = trainer._decode_latents_with_cache(final_pred, caches=caches, cache_paths=cache_paths)
    input_vis = dataset.visualize_sample({"target": x_final})["d_tri"][0]
    output_vis = dataset.visualize_sample({"target": y_final})["d_tri"][0]
    frames.append(make_frame(input_vis, output_vis, steps, steps, 0.0, panel_width))
    stats.append({
        "step": steps,
        "t": 0.0,
        "t_prev": 0.0,
        "z_t_abs_mean": float(final_sample.feats.float().abs().mean().item()),
        "pred_z0_abs_mean": float(final_pred.feats.float().abs().mean().item()),
        "decoded_input_d_tri_abs_mean": float(x_final.feats[:, :1].float().abs().mean().item()),
        "decoded_pred_d_tri_abs_mean": float(y_final.feats[:, :1].float().abs().mean().item()),
        "z_t_l1_delta_from_prev": tensor_l1_delta(final_sample.feats, prev_z_t_feats),
        "pred_z0_l1_delta_from_prev": tensor_l1_delta(final_pred.feats, prev_pred_z0_feats),
        "decoded_input_d_tri_l1_delta_from_prev": tensor_l1_delta(x_final.feats[:, :1], prev_input_d_tri),
        "decoded_pred_d_tri_l1_delta_from_prev": tensor_l1_delta(y_final.feats[:, :1], prev_output_d_tri),
    })
    return frames, stats


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
        / (
            f"eval_filtered_{args.split}_{args.low_resolution}to{args.high_resolution}"
            f"_support_only_uncond_trajectory_step{ckpt_step:07d}"
            f"_cfg{args.guidance_strength:g}_steps{args.steps}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_args = argparse.Namespace(
        low_resolution=args.low_resolution,
        high_resolution=args.high_resolution,
        no_latents=True,
        latent_name=None,
        split=args.split,
        instances=args.instances,
        render_resolution=args.render_resolution,
        metadata_filter_csv=args.metadata_filter_csv,
        no_train_duplicate_csv=args.no_train_duplicate_csv,
        triangle_filter_csv=args.triangle_filter_csv,
        disable_default_eval_filters=args.disable_default_eval_filters,
    )
    dataset, data_dir = build_dataset(cfg, root, flow_args)
    selected_instances = None
    if args.instances is not None:
        selected_instances = restrict_dataset_to_instances(dataset, args.instances)
    trainer = build_trainer(cfg, dataset, output_dir)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=args.instances is None,
        generator=generator,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )
    iterator = iter(loader)
    data = None
    for _ in range(max(0, args.sample_index) + 1):
        data = next(iterator)
    data = recursive_to_device(slice_batch(data, 1), trainer.device)
    latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])
    z_0, caches = build_support_latents(data["x_0"], latent_channels)

    frames, stats = sample_trajectory(
        trainer,
        dataset,
        z_0,
        data["cond"],
        caches,
        cache_paths=None,
        steps=args.steps,
        guidance_strength=args.guidance_strength,
        panel_width=args.panel_width,
    )

    gif_path = output_dir / f"unconditional_{args.high_resolution}_d_tri_trajectory.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
    )
    frames[0].save(output_dir / "first_frame.jpg")
    frames[-1].save(output_dir / "last_frame.jpg")
    stats_path = output_dir / "trajectory_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "root": str(root),
        "split": args.split,
        "data_dir": data_dir,
        "steps": args.steps,
        "guidance_strength": args.guidance_strength,
        "low_resolution": args.low_resolution,
        "high_resolution": args.high_resolution,
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "stats_path": str(stats_path),
        "instances": args.instances,
        "selected_instance": selected_instances[args.sample_index] if selected_instances is not None else None,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
