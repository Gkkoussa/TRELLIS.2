import os
import json
import copy
import glob
import argparse
import csv
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import o_voxel
import utils3d

from trellis2 import models, datasets, trainers
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained dense Gaussian patch flow model.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint to evaluate: latest or integer step.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <run_dir>/eval_<split>_step<step>.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional JSON data_dir override.")
    parser.add_argument("--root", type=str, default=None, help="Processed dataset root used when --data_dir is omitted.")
    parser.add_argument("--split", type=str, default="test", help="Split name under <root>/splits/.")
    parser.add_argument(
        "--metadata_filter_csv",
        type=str,
        default=None,
        help="Optional metadata CSV whose sha256 rows define the evaluation subset.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap for a smoke test.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of paired GT/reconstruction renders to save.")
    parser.add_argument("--render_resolution", type=int, default=256, help="Resolution for paired 3D patch renders.")
    parser.add_argument("--render_ssaa", type=int, default=2, help="SSAA for paired 3D patch renders.")
    parser.add_argument("--reconstruction_t", type=float, default=0.5, help="Fallback single noising timestep for recon renders.")
    parser.add_argument(
        "--reconstruction_ts",
        type=str,
        default=None,
        help="Comma-separated noising timesteps for reconstruction sweep, e.g. 0.1,0.3,0.5,0.7,0.9.",
    )
    parser.add_argument("--generated_samples", type=int, default=16, help="Number of unconditional random-noise generations to render.")
    parser.add_argument("--generated_threshold", type=float, default=0.05, help="Display threshold for generated active cells.")
    parser.add_argument("--snapshot_samples", type=int, default=0, help="Optional pure generation snapshots from noise.")
    parser.add_argument("--snapshot_batch_size", type=int, default=1, help="Batch size used by trainer.snapshot().")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Euler sampling steps for snapshots.")
    parser.add_argument("--ema_rate", type=str, default=None, help="Optional EMA rate, e.g. 0.9999.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def apply_metadata_filter(dataset, metadata_filter_csv: str) -> dict:
    with open(metadata_filter_csv, newline="") as f:
        allowed_sha256 = {row["sha256"] for row in csv.DictReader(f)}

    original_size = len(dataset.instances)
    old_instances = dataset.instances
    old_loads = getattr(dataset, "loads", None)

    if old_loads is not None and len(old_loads) == len(old_instances):
        kept = [
            (instance, load)
            for instance, load in zip(old_instances, old_loads)
            if instance[1] in allowed_sha256
        ]
        dataset.instances = [instance for instance, _ in kept]
        dataset.loads = [load for _, load in kept]
    else:
        dataset.instances = [
            instance for instance in old_instances if instance[1] in allowed_sha256
        ]

    if hasattr(dataset, "metadata") and len(dataset.metadata) > 0:
        keep_index = dataset.metadata.index.intersection(allowed_sha256)
        dataset.metadata = dataset.metadata.loc[keep_index]

    return {
        "metadata_filter_csv": str(Path(metadata_filter_csv).resolve()),
        "metadata_filter_allowed_sha256": len(allowed_sha256),
        "metadata_filter_original_size": original_size,
        "metadata_filter_removed": original_size - len(dataset.instances),
    }


def parse_t_values(value: str | None, fallback: float) -> list[float]:
    if value is None or value.strip() == "":
        values = [fallback]
    else:
        values = [float(v.strip()) for v in value.split(",") if v.strip() != ""]
    if not values:
        raise ValueError("At least one reconstruction timestep is required.")
    bad = [v for v in values if v < 0.0 or v > 1.0]
    if bad:
        raise ValueError(f"Reconstruction timesteps must be in [0, 1], got {bad}")
    return values


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        files = glob.glob(str(run_dir / "ckpts" / "misc_*.pt"))
        if files:
            return max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files)
        files = glob.glob(str(run_dir / "ckpts" / "denoiser_step*.pt"))
        if not files:
            raise RuntimeError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files)
    if ckpt == "none":
        raise ValueError("ckpt=none is not valid for evaluation.")
    return int(ckpt)


def build_data_dir(root: Path, split: str) -> dict:
    split_root = root / "splits" / split
    return {
        split: {
            "base": str(split_root),
            "gaussian_distance_voxel": str(split_root / "gaussian_distance_voxels_256"),
        }
    }


def load_denoiser_checkpoint(model, run_dir: Path, step: int, ema_rate: str | None, device: torch.device) -> str:
    if ema_rate is None:
        path = run_dir / "ckpts" / f"denoiser_step{step:07d}.pt"
    else:
        path = run_dir / "ckpts" / f"denoiser_ema{ema_rate}_step{step:07d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Denoiser checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return str(path)


def build_camera(device: torch.device):
    extrinsics = utils3d.extrinsics_look_at(
        eye=torch.tensor([1.2, 0.5, 1.2]),
        look_at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
    ).to(device)
    intrinsics = utils3d.intrinsics_from_fov_xy(
        fov_x=torch.deg2rad(torch.tensor(45.0)),
        fov_y=torch.deg2rad(torch.tensor(45.0)),
    ).to(device)
    return extrinsics, intrinsics


def render_attr(renderer, coords, attrs, patch_size: int, extrinsics, intrinsics, resolution: int):
    if coords.shape[0] == 0:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    position = (coords.float() / patch_size - 0.5).to(attrs.device)
    output = renderer.render(
        position=position,
        attrs=attrs,
        voxel_size=1.0 / patch_size,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    image = output.attr.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def patch_to_render_tensors(patch: torch.Tensor, background_value: float, occupancy_mask: torch.Tensor | None = None):
    patch = patch.detach().float().cpu()
    if occupancy_mask is None:
        occupancy_mask = (patch - background_value).abs().amax(dim=0) > 1e-6
    else:
        occupancy_mask = occupancy_mask.detach().bool().cpu()

    coords = occupancy_mask.nonzero(as_tuple=False)
    if coords.shape[0] == 0:
        empty_attr = torch.empty(0, 3, dtype=torch.float32)
        return coords, empty_attr, empty_attr, occupancy_mask

    feats = patch[:, coords[:, 0], coords[:, 1], coords[:, 2]].t()
    attrs = ((feats + 1.0) * 0.5).clamp(0, 1)
    edge = attrs[:, 0:3]
    vertex = attrs[:, 3:6] if attrs.shape[1] >= 6 else edge
    return coords, edge, vertex, occupancy_mask


def stack_edge_vertex(edge: np.ndarray, vertex: np.ndarray) -> np.ndarray:
    """Stack edge (top) and vertex (bottom) renders into one RGB image."""
    return np.concatenate([edge, vertex], axis=0)


def save_state_render(path: Path, edge: np.ndarray, vertex: np.ndarray):
    Image.fromarray(stack_edge_vertex(edge, vertex), mode="RGB").save(path)


def make_generated_sheet(images: dict, sample_idx: int, stats: dict, panel_resolution: int):
    panels = [
        ("edge", "Generated edge RGB"),
        ("vertex", "Generated vertex RGB"),
    ]
    cols = 2
    label_h = 24
    header_h = 42
    canvas = Image.new(
        "RGB",
        (cols * panel_resolution, header_h + panel_resolution + label_h),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 8),
        (
            f"sample={sample_idx} generated_voxels={stats['generated_voxels']} "
            f"ratio={stats['generated_ratio']:.5f} threshold={stats['threshold']:.3f}"
        ),
        fill=(255, 255, 255),
        font=font,
    )

    for col, (key, label) in enumerate(panels):
        x0 = col * panel_resolution
        y0 = header_h
        canvas.paste(Image.fromarray(images[key], mode="RGB"), (x0, y0))
        draw.rectangle(
            [x0, y0 + panel_resolution, x0 + panel_resolution, y0 + panel_resolution + label_h],
            fill=(20, 20, 20),
        )
        draw.text((x0 + 8, y0 + panel_resolution + 6), label, fill=(255, 255, 255), font=font)

    return canvas


def make_reconstruction_sheet(images: dict, sample_idx: int, stats: dict, panel_resolution: int):
    # Columns: target | noised @ t | reconstruction. Rows: edge, vertex.
    panel_keys = [
        [("gt_edge", "Target edge"), ("noised_edge", "Noised edge"), ("recon_edge", "Recon edge")],
        [("gt_vertex", "Target vertex"), ("noised_vertex", "Noised vertex"), ("recon_vertex", "Recon vertex")],
    ]
    cols = 3
    rows = 2
    label_h = 24
    header_h = 42
    canvas = Image.new(
        "RGB",
        (cols * panel_resolution, header_h + rows * (panel_resolution + label_h)),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 8),
        (
            f"sample={sample_idx} t={stats['t']:.3f} mse={stats['mse']:.6f} "
            f"gt_voxels={stats['gt_voxels']}"
        ),
        fill=(255, 255, 255),
        font=font,
    )

    for row, row_panels in enumerate(panel_keys):
        for col, (key, label) in enumerate(row_panels):
            x0 = col * panel_resolution
            y0 = header_h + row * (panel_resolution + label_h)
            canvas.paste(Image.fromarray(images[key], mode="RGB"), (x0, y0))
            draw.rectangle(
                [x0, y0 + panel_resolution, x0 + panel_resolution, y0 + panel_resolution + label_h],
                fill=(20, 20, 20),
            )
            draw.text((x0 + 8, y0 + panel_resolution + 6), label, fill=(255, 255, 255), font=font)

    return canvas


def make_t_sweep_sheet(
    gt_edge: np.ndarray,
    gt_vertex: np.ndarray,
    recon_entries: list[dict],
    sample_idx: int,
    gt_voxels: int,
    panel_resolution: int,
):
    cols = 1 + len(recon_entries)
    rows = 2
    label_h = 28
    header_h = 48
    canvas = Image.new(
        "RGB",
        (cols * panel_resolution, header_h + rows * (panel_resolution + label_h)),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    mse_text = " ".join([f"t={entry['t']:.2f}:{entry['mse']:.5f}" for entry in recon_entries])
    draw.text(
        (8, 8),
        f"sample={sample_idx} gt_voxels={gt_voxels} MSE {mse_text}",
        fill=(255, 255, 255),
        font=font,
    )

    panels = [
        (0, 0, gt_edge, "Target edge"),
        (1, 0, gt_vertex, "Target vertex"),
    ]
    for col, entry in enumerate(recon_entries, start=1):
        panels.append((0, col, entry["recon_edge"], f"Recon edge t={entry['t']:.2f}"))
        panels.append((1, col, entry["recon_vertex"], f"Recon vertex MSE={entry['mse']:.5f}"))

    for row, col, image, label in panels:
        x0 = col * panel_resolution
        y0 = header_h + row * (panel_resolution + label_h)
        canvas.paste(Image.fromarray(image, mode="RGB"), (x0, y0))
        draw.rectangle(
            [x0, y0 + panel_resolution, x0 + panel_resolution, y0 + panel_resolution + label_h],
            fill=(20, 20, 20),
        )
        draw.text((x0 + 8, y0 + panel_resolution + 7), label, fill=(255, 255, 255), font=font)

    return canvas


def render_generated_visuals(
    trainer,
    dataset,
    num_samples: int,
    batch_size: int,
    output_dir: Path,
    sampling_steps: int,
    render_resolution: int,
    render_ssaa: int,
    generated_threshold: float,
) -> list[dict]:
    if num_samples <= 0:
        return []

    denoiser = trainer.training_models["denoiser"]
    denoiser.eval()
    vis_dir = output_dir / "generated_samples"
    vis_dir.mkdir(parents=True, exist_ok=True)

    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": render_resolution, "ssaa": render_ssaa}
    )
    extrinsics, intrinsics = build_camera(trainer.device)
    sampler = trainer.get_sampler()
    if trainer.mix_precision_mode == "amp":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

    summary = []
    saved = 0
    with torch.no_grad():
        while saved < num_samples:
            batch = min(batch_size, num_samples - saved)
            noise = torch.randn(
                batch,
                denoiser.in_channels,
                dataset.patch_size,
                dataset.patch_size,
                dataset.patch_size,
                device=trainer.device,
            )
            if getattr(dataset, "cond_as_token", False):
                cond = torch.zeros(batch, 1, denoiser.cond_channels, device=trainer.device)
            else:
                cond = torch.zeros(batch, denoiser.cond_channels, device=trainer.device)

            with amp_context():
                generated = sampler.sample(
                    denoiser,
                    noise=noise,
                    cond=cond,
                    steps=sampling_steps,
                    verbose=False,
                ).samples.clamp(-1, 1)

            for i in range(batch):
                generated_mask = (
                    (generated[i].detach().float().cpu() - dataset.background_value).abs().amax(dim=0)
                    > generated_threshold
                )
                gen_coords, gen_edge, gen_vertex, gen_mask = patch_to_render_tensors(
                    generated[i],
                    dataset.background_value,
                    occupancy_mask=generated_mask,
                )
                gen_edge = gen_edge.to(trainer.device)
                gen_vertex = gen_vertex.to(trainer.device)

                edge_img = render_attr(renderer, gen_coords, gen_edge, dataset.patch_size, extrinsics, intrinsics, render_resolution)
                vertex_img = render_attr(renderer, gen_coords, gen_vertex, dataset.patch_size, extrinsics, intrinsics, render_resolution)

                stats = {
                    "sample": saved,
                    "threshold": generated_threshold,
                    "generated_voxels": int(gen_mask.sum().item()),
                    "generated_ratio": float(gen_mask.float().mean().item()),
                }
                image = make_generated_sheet(
                    {"edge": edge_img, "vertex": vertex_img},
                    saved,
                    stats,
                    render_resolution,
                )
                sheet_path = vis_dir / f"generated_{saved:04d}.png"
                state_path = vis_dir / f"generated_{saved:04d}_edge_vertex.png"
                npz_path = vis_dir / f"generated_{saved:04d}.npz"
                image.save(sheet_path)
                save_state_render(state_path, edge_img, vertex_img)
                np.savez_compressed(
                    npz_path,
                    x=generated[i].detach().float().cpu().numpy().astype(np.float32),
                    active_mask=gen_mask.numpy().astype(np.uint8),
                )

                stats["path"] = str(sheet_path)
                stats["edge_vertex_path"] = str(state_path)
                stats["npz_path"] = str(npz_path)
                summary.append(stats)
                saved += 1
                if saved >= num_samples:
                    break

    with open(vis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def render_reconstruction_visuals(
    trainer,
    dataset,
    num_samples: int,
    batch_size: int,
    output_dir: Path,
    reconstruction_ts: list[float],
    render_resolution: int,
    render_ssaa: int,
) -> list[dict]:
    if num_samples <= 0:
        return []

    denoiser = trainer.training_models["denoiser"]
    denoiser.eval()
    vis_dir = output_dir / "reconstructions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )
    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": render_resolution, "ssaa": render_ssaa}
    )
    extrinsics, intrinsics = build_camera(trainer.device)
    if trainer.mix_precision_mode == "amp":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

    t_dirs = {}
    for t_value in reconstruction_ts:
        t_dir = vis_dir / f"t_{t_value:.3f}"
        t_dir.mkdir(parents=True, exist_ok=True)
        t_dirs[t_value] = t_dir

    summary = []
    mse_by_t = {f"{t_value:.6f}": [] for t_value in reconstruction_ts}
    saved = 0
    with torch.no_grad():
        for data in loader:
            if saved >= num_samples:
                break
            data = recursive_to_device(data, trainer.device)
            x_0 = data["x_0"]
            cond = data.get("cond", None)
            kwargs = {k: v for k, v in data.items() if k not in {"x_0", "cond"}}

            noise = torch.randn_like(x_0)
            model_cond = trainer.get_cond(cond, **kwargs)
            recon_by_t = {}
            for t_value in reconstruction_ts:
                t = torch.full((x_0.shape[0],), t_value, device=x_0.device, dtype=torch.float32)
                x_t = trainer.diffuse(x_0, t, noise=noise)
                with amp_context():
                    pred_v = denoiser(x_t, t * 1000, model_cond, **kwargs)
                t_view = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
                pred_x0 = (1 - trainer.sigma_min) * x_t - (
                    trainer.sigma_min + (1 - trainer.sigma_min) * t_view
                ) * pred_v
                recon_by_t[t_value] = {
                    "x_t": x_t,
                    "pred_x0": pred_x0,
                }

            for i in range(x_0.shape[0]):
                if saved >= num_samples:
                    break
                gt_coords, gt_edge, gt_vertex, gt_mask = patch_to_render_tensors(
                    x_0[i],
                    dataset.background_value,
                )
                gt_edge = gt_edge.to(trainer.device)
                gt_vertex = gt_vertex.to(trainer.device)
                gt_edge_img = render_attr(renderer, gt_coords, gt_edge, dataset.patch_size, extrinsics, intrinsics, render_resolution)
                gt_vertex_img = render_attr(renderer, gt_coords, gt_vertex, dataset.patch_size, extrinsics, intrinsics, render_resolution)

                sample_stats = {
                    "sample": saved,
                    "gt_voxels": int(gt_mask.sum().item()),
                    "per_t": [],
                }
                sweep_entries = []
                target_path = vis_dir / f"reconstruction_{saved:04d}_target.png"
                save_state_render(target_path, gt_edge_img, gt_vertex_img)
                sample_stats["target_path"] = str(target_path)

                for t_value in reconstruction_ts:
                    x_t = recon_by_t[t_value]["x_t"]
                    pred_x0 = recon_by_t[t_value]["pred_x0"]
                    noised_coords, noised_edge, noised_vertex, _ = patch_to_render_tensors(
                        x_t[i].clamp(-1, 1),
                        dataset.background_value,
                        occupancy_mask=gt_mask,
                    )
                    recon_coords, recon_edge, recon_vertex, _ = patch_to_render_tensors(
                        pred_x0[i].clamp(-1, 1),
                        dataset.background_value,
                        occupancy_mask=gt_mask,
                    )
                    noised_edge = noised_edge.to(trainer.device)
                    noised_vertex = noised_vertex.to(trainer.device)
                    recon_edge = recon_edge.to(trainer.device)
                    recon_vertex = recon_vertex.to(trainer.device)

                    noised_edge_img = render_attr(renderer, noised_coords, noised_edge, dataset.patch_size, extrinsics, intrinsics, render_resolution)
                    recon_edge_img = render_attr(renderer, recon_coords, recon_edge, dataset.patch_size, extrinsics, intrinsics, render_resolution)
                    noised_vertex_img = render_attr(renderer, noised_coords, noised_vertex, dataset.patch_size, extrinsics, intrinsics, render_resolution)
                    recon_vertex_img = render_attr(renderer, recon_coords, recon_vertex, dataset.patch_size, extrinsics, intrinsics, render_resolution)

                    mse = F.mse_loss(pred_x0[i].float(), x_0[i].float()).item()
                    t_key = f"{t_value:.6f}"
                    mse_by_t[t_key].append(mse)
                    t_dir = t_dirs[t_value]
                    t_tag = f"t{t_value:.3f}".replace(".", "p")

                    images = {
                        "gt_edge": gt_edge_img,
                        "noised_edge": noised_edge_img,
                        "recon_edge": recon_edge_img,
                        "gt_vertex": gt_vertex_img,
                        "noised_vertex": noised_vertex_img,
                        "recon_vertex": recon_vertex_img,
                    }
                    stats = {
                        "sample": saved,
                        "t": t_value,
                        "mse": mse,
                        "gt_voxels": sample_stats["gt_voxels"],
                    }
                    image = make_reconstruction_sheet(images, saved, stats, render_resolution)
                    sheet_path = t_dir / f"reconstruction_{saved:04d}_{t_tag}.png"
                    noised_path = t_dir / f"reconstruction_{saved:04d}_{t_tag}_noised.png"
                    recon_path = t_dir / f"reconstruction_{saved:04d}_{t_tag}_reconstruction.png"
                    image.save(sheet_path)
                    save_state_render(noised_path, noised_edge_img, noised_vertex_img)
                    save_state_render(recon_path, recon_edge_img, recon_vertex_img)

                    sample_stats["per_t"].append({
                        "t": t_value,
                        "mse": mse,
                        "path": str(sheet_path),
                        "noised_path": str(noised_path),
                        "reconstruction_path": str(recon_path),
                    })
                    sweep_entries.append({
                        "t": t_value,
                        "mse": mse,
                        "recon_edge": recon_edge_img,
                        "recon_vertex": recon_vertex_img,
                    })

                sweep_image = make_t_sweep_sheet(
                    gt_edge_img,
                    gt_vertex_img,
                    sweep_entries,
                    saved,
                    sample_stats["gt_voxels"],
                    render_resolution,
                )
                sweep_path = vis_dir / f"reconstruction_{saved:04d}_t_sweep.png"
                sweep_image.save(sweep_path)
                sample_stats["t_sweep_path"] = str(sweep_path)

                summary.append(sample_stats)
                saved += 1

    mse_summary = {
        t_key: {
            "mean_mse": float(np.mean(values)) if values else None,
            "count": len(values),
        }
        for t_key, values in mse_by_t.items()
    }
    with open(vis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(vis_dir / "mse_by_t.json", "w") as f:
        json.dump(mse_summary, f, indent=2)
    return summary


def evaluate_dense_flow_mse(trainer, loader, max_batches: int | None = None) -> dict:
    denoiser = trainer.training_models["denoiser"]
    denoiser.eval()

    total_mse_sum = 0.0
    total_mse_count = 0
    total_instances = 0
    total_grid_tokens = 0
    channel_mse_sum = None
    bin_mse_sum = {i: 0.0 for i in range(10)}
    bin_count = {i: 0 for i in range(10)}

    if trainer.mix_precision_mode == "amp":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc="Evaluating dense patch flow MSE")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            data = recursive_to_device(data, trainer.device)
            x_0 = data["x_0"]
            cond = data.get("cond", None)
            kwargs = {k: v for k, v in data.items() if k not in {"x_0", "cond"}}

            noise = torch.randn_like(x_0)
            t = trainer.sample_t(x_0.shape[0]).to(x_0.device).float()
            x_t = trainer.diffuse(x_0, t, noise=noise)
            model_cond = trainer.get_cond(cond, **kwargs)

            with amp_context():
                pred = denoiser(x_t, t * 1000, model_cond, **kwargs)
                target = trainer.get_v(x_0, noise, t)

            diff = (pred.float() - target.float()).pow(2)
            total_mse_sum += diff.sum().item()
            total_mse_count += diff.numel()
            total_instances += x_0.shape[0]
            total_grid_tokens += x_0.shape[0] * int(np.prod(x_0.shape[2:]))

            channel_sum = diff.sum(dim=(0, 2, 3, 4)).detach().cpu()
            if channel_mse_sum is None:
                channel_mse_sum = torch.zeros_like(channel_sum)
            channel_mse_sum += channel_sum

            time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
            time_bin = np.clip(time_bin, 0, 9)
            for i in range(x_0.shape[0]):
                instance_mse = F.mse_loss(pred[i].float(), target[i].float()).item()
                b = int(time_bin[i])
                bin_mse_sum[b] += instance_mse
                bin_count[b] += 1

    values_per_channel = total_grid_tokens
    if channel_mse_sum is not None and values_per_channel:
        channel_mse = (channel_mse_sum / values_per_channel).tolist()
    else:
        channel_mse = None

    return {
        "num_instances": total_instances,
        "num_grid_tokens": total_grid_tokens,
        "mse": total_mse_sum / total_mse_count if total_mse_count else None,
        "channel_mse": channel_mse,
        "bins": {
            f"bin_{i}": {
                "mse": bin_mse_sum[i] / bin_count[i] if bin_count[i] else None,
                "count": bin_count[i],
            }
            for i in range(10)
        },
    }


def main():
    args = parse_args()
    reconstruction_ts = parse_t_values(args.reconstruction_ts, args.reconstruction_t)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config is not None else run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = json.load(open(config_path, "r"))
    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])

    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / f"eval_{args.split}_step{ckpt_step:07d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        data_dir = json.loads(args.data_dir)
    else:
        if args.root is None:
            raise ValueError("Either --data_dir or --root must be provided.")
        data_dir = build_data_dir(Path(args.root).resolve(), args.split)

    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    metadata_filter_info = None
    if args.metadata_filter_csv is not None:
        metadata_filter_info = apply_metadata_filter(dataset, args.metadata_filter_csv)
        print(
            "Applied metadata filter: "
            f"{metadata_filter_info['metadata_filter_original_size']} -> {len(dataset)} "
            f"instances, removed {metadata_filter_info['metadata_filter_removed']}"
        )

    model_dict = {
        name: getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda()
        for name, model_cfg in cfg["models"].items()
    }

    trainer = getattr(trainers, cfg["trainer"]["name"])(
        model_dict,
        dataset,
        **trainer_args,
        output_dir=str(output_dir),
        load_dir=None,
        step=None,
    )

    ckpt_path = load_denoiser_checkpoint(
        trainer.models["denoiser"],
        run_dir,
        ckpt_step,
        args.ema_rate,
        trainer.device,
    )
    trainer.models["denoiser"].eval()

    batch_size = args.batch_size or trainer_args["batch_size_per_gpu"]
    num_workers = args.num_workers if args.num_workers is not None else trainer_args.get("num_workers", 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    metrics = evaluate_dense_flow_mse(trainer, loader, max_batches=args.max_batches)
    metrics.update({
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "dataset_size": len(dataset),
        "max_batches": args.max_batches,
    })
    if metadata_filter_info is not None:
        metrics.update(metadata_filter_info)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    recon_summary = render_reconstruction_visuals(
        trainer,
        dataset,
        args.num_samples,
        args.snapshot_batch_size,
        output_dir,
        reconstruction_ts,
        args.render_resolution,
        args.render_ssaa,
    )
    if recon_summary:
        print(
            f"Saved reconstruction renders to {output_dir / 'reconstructions'} "
            f"(sheet + target/noised/reconstruction per sample)"
        )

    generated_summary = render_generated_visuals(
        trainer,
        dataset,
        args.generated_samples,
        args.snapshot_batch_size,
        output_dir,
        args.sampling_steps,
        args.render_resolution,
        args.render_ssaa,
        args.generated_threshold,
    )
    if generated_summary:
        print(f"Saved random-noise generated renders to {output_dir / 'generated_samples'}")

    if args.snapshot_samples > 0:
        suffix = f"{args.split}_step{ckpt_step:07d}"
        trainer.snapshot(
            suffix=suffix,
            num_samples=args.snapshot_samples,
            batch_size=args.snapshot_batch_size,
            steps=args.sampling_steps,
        )
        print(f"Saved visualization samples to {output_dir / 'samples' / suffix}")


if __name__ == "__main__":
    main()
