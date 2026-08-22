"""Visualize point-density flow trajectories from several starting timesteps."""

import argparse
import glob
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trellis2 import models
from trellis2.datasets.point_density_mesh import PointDensityMeshDataset, render_point_density


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--ema_rate", default="0.9999")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_t", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration_ms", type=int, default=140)
    parser.add_argument("--match_gt_mean", action="store_true")
    return parser.parse_args()


def find_step(run_dir: Path, ckpt: str, ema_rate: str) -> int:
    if ckpt != "latest":
        return int(ckpt)
    paths = glob.glob(str(run_dir / "ckpts" / f"denoiser_ema{ema_rate}_step*.pt"))
    if not paths:
        raise FileNotFoundError(f"No EMA checkpoints found under {run_dir / 'ckpts'}")
    return max(int(Path(path).stem.split("step")[-1]) for path in paths)


def to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(array)


def labeled_panel(image: Image.Image, label: str) -> Image.Image:
    header = 38
    panel = Image.new("RGB", (image.width, image.height + header), "white")
    panel.paste(image, (0, header))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    ImageDraw.Draw(panel).text((image.width // 2, header // 2), label, fill="black", font=font, anchor="mm")
    return panel


def make_frame(gt: Image.Image, states, current_t, step: int, total_steps: int) -> Image.Image:
    panels = [labeled_panel(gt, "GT density")]
    for start_t, state, t in zip(states["start_t"], states["images"], current_t):
        panels.append(labeled_panel(state, f"start={start_t:g} | t={t:.3f}"))
    footer = 34
    frame = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height + footer), "white")
    x = 0
    for panel in panels:
        frame.paste(panel, (x, 0))
        x += panel.width
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    ImageDraw.Draw(frame).text(
        (frame.width // 2, frame.height - footer // 2),
        f"Euler update {step}/{total_steps}",
        fill="black",
        font=font,
        anchor="mm",
    )
    return frame


@torch.no_grad()
def trajectory_frames(model, sample, start_times, steps, sigma_min, device, match_gt_mean=False):
    points = sample["context_points"][None].to(device)
    normals = sample["context_normals"][None].to(device)
    x0 = sample["context_density"][None].to(device)
    generator = torch.Generator(device=device).manual_seed(sample["noise_seed"])
    noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=generator)
    anchor_indices = model.select_anchor_indices(points, random_start=False)
    states = [
        noise.clone()
        if t >= 1.0
        else (1.0 - t) * x0 + (sigma_min + (1.0 - sigma_min) * t) * noise
        for t in start_times
    ]
    gt = to_pil(render_point_density(points, x0)[0])
    frames = []
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for step in range(steps + 1):
        times = [start_t * (1.0 - step / steps) for start_t in start_times]
        rendered = [to_pil(render_point_density(points, state)[0]) for state in states]
        frames.append(make_frame(
            gt,
            {"start_t": start_times, "images": rendered},
            times,
            step,
            steps,
        ))
        if step == steps:
            break
        next_times = [start_t * (1.0 - (step + 1) / steps) for start_t in start_times]
        for index, (state, t, t_prev) in enumerate(zip(states, times, next_times)):
            timestep = torch.full((1,), t, device=device, dtype=torch.float32)
            with autocast():
                pred_x0 = model(
                    state,
                    timestep,
                    points,
                    normals,
                    anchor_indices=anchor_indices,
                )["context"]
            pred_x0 = pred_x0.float()
            if match_gt_mean:
                pred_x0 = pred_x0 + x0.float().mean(dim=1, keepdim=True) - pred_x0.mean(
                    dim=1, keepdim=True
                )
            scale = sigma_min + (1.0 - sigma_min) * t
            velocity = ((1.0 - sigma_min) * state - pred_x0) / scale
            states[index] = state - (t - t_prev) * velocity
    return frames


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    root = Path(args.root)
    config = json.load(open(run_dir / "config.json"))
    step = find_step(run_dir, args.ckpt, args.ema_rate)
    start_tag = "_".join(f"{value:g}".replace(".", "p") for value in args.start_t)
    constraint_tag = "_gtmean" if args.match_gt_mean else ""
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir
        / f"eval_density_trajectories_tstart{start_tag}{constraint_tag}_ema{args.ema_rate}_step{step:07d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_args = dict(config["validation_dataset"]["args"])
    dataset_args["density_stats_path"] = str(root / "point_density_stats/density512_filtered_train.json")
    dataset_args["deterministic_sampling"] = True
    filters = ",".join([
        str(root / "splits/test_triangle_field_512/metadata.csv"),
        str(root / "metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv"),
    ])
    data_dir = {"filtered_test": {"mesh": str(root), "_metadata_filter_csv": filters}}
    dataset = PointDensityMeshDataset(json.dumps(data_dir), **dataset_args)

    model = getattr(models, config["models"]["denoiser"]["name"])(
        **config["models"]["denoiser"]["args"]
    )
    ckpt_path = run_dir / "ckpts" / f"denoiser_ema{args.ema_rate}_step{step:07d}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    sigma_min = float(config["trainer"]["args"]["sigma_min"])

    summaries = []
    for index in range(min(args.num_samples, len(dataset))):
        sample = dataset[index]
        sample["noise_seed"] = args.seed + index
        frames = trajectory_frames(
            model,
            sample,
            args.start_t,
            args.steps,
            sigma_min,
            device,
            match_gt_mean=args.match_gt_mean,
        )
        gif_path = output_dir / f"sample_{index:02d}_density_trajectory.gif"
        gif_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for frame in frames]
        gif_frames.extend([gif_frames[-1]] * 8)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=args.duration_ms,
            loop=0,
            optimize=False,
        )
        frames[0].save(output_dir / f"sample_{index:02d}_first.jpg")
        frames[-1].save(output_dir / f"sample_{index:02d}_last.jpg")
        summaries.append({"index": index, "sha256": str(dataset.instances[index][1]), "gif": str(gif_path)})
        print(f"Wrote {gif_path}", flush=True)

    summary = {
        "checkpoint": str(ckpt_path),
        "steps_per_trajectory": args.steps,
        "start_t": args.start_t,
        "match_gt_mean": args.match_gt_mean,
        "samples": summaries,
    }
    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
