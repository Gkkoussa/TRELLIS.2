import argparse
import copy
import glob
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from trellis2 import datasets, models, trainers
from trellis2.renderers import VoxelRenderer
from trellis2.utils.data_utils import recursive_to_device

from eval_gaussian_patch_sparse_flow import (
    apply_metadata_filter,
    build_camera,
    find_ckpt_step,
    get_amp_context,
    load_denoiser_checkpoint,
    make_sheet,
    render_sparse_views,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a GIF of sparse Gaussian-grid flow sampling from pure noise "
            "to predicted 6-channel voxel features on GT sparse coordinates."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint: latest or integer step.")
    parser.add_argument("--ema_rate", type=str, default=None, help="Optional EMA rate, e.g. 0.9999.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <run_dir>/sampling_gif_step<step>.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional JSON data_dir override.")
    parser.add_argument("--root", type=str, default=None, help="Processed dataset root used when --data_dir is omitted.")
    parser.add_argument("--split", type=str, default="test", help="Split name under <root>/splits/.")
    parser.add_argument("--metadata_filter_csv", type=str, default=None, help="Optional CSV with sha256 rows to keep.")
    parser.add_argument("--sample_index", type=int, default=0, help="Dataset index after filtering.")
    parser.add_argument("--steps", type=int, default=50, help="Euler sampling steps.")
    parser.add_argument("--frames", type=int, default=26, help="Number of GIF frames including initial and final state.")
    parser.add_argument("--duration_ms", type=int, default=160, help="GIF frame duration in milliseconds.")
    parser.add_argument("--render_resolution", type=int, default=256, help="Per-view render resolution.")
    parser.add_argument("--render_ssaa", type=int, default=2, help="Render supersampling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_gt", action="store_true", help="Do not include GT edge/vertex panels for reference.")
    return parser.parse_args()


def build_grid64_data_dir(root: Path, split: str) -> dict:
    return {
        split: {
            "base": str(root / "splits" / split),
            "gaussian_distance_voxel": str(root / "gaussian_distance_voxels_64"),
        }
    }


def selected_state_indices(steps: int, frames: int) -> list[int]:
    if steps < 1:
        raise ValueError("--steps must be >= 1")
    if frames < 2:
        raise ValueError("--frames must be >= 2")
    return sorted(set(np.linspace(0, steps, min(frames, steps + 1)).round().astype(int).tolist()))


def slice_batch(data: dict, start: int, end: int) -> dict:
    sliced = {}
    for key, value in data.items():
        if hasattr(value, "__getitem__") and hasattr(value, "shape"):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int):
    if not frames:
        raise RuntimeError("No frames were rendered.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


@torch.no_grad()
def render_sampling_gif(
    trainer,
    dataset,
    data: dict,
    output_dir: Path,
    steps: int,
    frames: int,
    duration_ms: int,
    render_resolution: int,
    render_ssaa: int,
    include_gt: bool,
) -> dict:
    x_0 = data["x_0"]
    kwargs = {k: v for k, v in data.items() if k not in {"x_0", "cond"}}
    inference = trainer.get_inference_cond(**{k: v for k, v in data.items() if k != "x_0"})
    cond = inference.pop("cond", None)
    kwargs.update(inference)

    sampler = trainer.get_sampler()
    noise = x_0.replace(torch.randn_like(x_0.feats))
    sample = noise * sampler.noise_scale

    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = render_ssaa
    extrinsics, intrinsics = build_camera(trainer.device)

    frame_dir = output_dir / "frames"
    sample_frame_dir = frame_dir / "sample_state"
    pred_x0_frame_dir = frame_dir / "pred_x0"
    sample_frame_dir.mkdir(parents=True, exist_ok=True)
    pred_x0_frame_dir.mkdir(parents=True, exist_ok=True)
    state_indices = selected_state_indices(steps, frames)
    state_index_set = set(state_indices)
    pred_x0_state_indices = sorted(set(1 if idx == 0 else idx for idx in state_indices))
    pred_x0_state_index_set = set(pred_x0_state_indices)
    t_seq = np.linspace(1.0, 0.0, steps + 1).tolist()
    amp_context = get_amp_context(trainer)
    rendered_frames = []
    rendered_pred_x0_frames = []
    frame_records = []
    pred_x0_frame_records = []

    def render_state(state, state_idx: int, label_prefix: str, frame_list: list[Image.Image], records: list[dict], out_dir: Path):
        t_value = t_seq[state_idx]
        mse = F.mse_loss(state.feats.float(), x_0.feats.float()).item() if state.feats.numel() else None
        panels = [
            (f"{label_prefix} edge", render_sparse_views(renderer, state, 0, slice(0, 3), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
            (f"{label_prefix} vertex", render_sparse_views(renderer, state, 0, slice(3, 6), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
        ]
        if include_gt:
            panels.extend([
                ("gt edge", render_sparse_views(renderer, x_0, 0, slice(0, 3), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("gt vertex", render_sparse_views(renderer, x_0, 0, slice(3, 6), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
            ])
        title = f"step={state_idx:03d}/{steps:03d} t={t_value:.4f} feature_mse={mse if mse is not None else float('nan'):.6f}"
        frame = make_sheet(panels, title)
        png_path = out_dir / f"frame_{len(frame_list):03d}_step{state_idx:03d}.png"
        frame.save(png_path)
        frame_list.append(frame)
        records.append({
            "frame": len(frame_list) - 1,
            "step": state_idx,
            "t": t_value,
            "feature_mse": mse,
            "png": str(png_path),
        })

    if 0 in state_index_set:
        render_state(sample, 0, "sample", rendered_frames, frame_records, sample_frame_dir)

    for step_idx, (t_cur, t_prev) in enumerate(
        tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc="Sampling GIF states"),
        start=1,
    ):
        with amp_context:
            out = sampler.sample_once(
                trainer.models["denoiser"],
                sample,
                t_cur,
                t_prev,
                cond,
                **kwargs,
            )
        sample = out.pred_x_prev
        if step_idx in state_index_set:
            render_state(sample, step_idx, "sample", rendered_frames, frame_records, sample_frame_dir)
        if step_idx in pred_x0_state_index_set:
            render_state(out.pred_x_0, step_idx, "pred x0", rendered_pred_x0_frames, pred_x0_frame_records, pred_x0_frame_dir)

    gif_path = output_dir / "sampling_noise_to_voxel_features.gif"
    pred_x0_gif_path = output_dir / "pred_x0_noise_to_voxel_features.gif"
    save_gif(rendered_frames, gif_path, duration_ms)
    save_gif(rendered_pred_x0_frames, pred_x0_gif_path, duration_ms)

    return {
        "gif": str(gif_path),
        "pred_x0_gif": str(pred_x0_gif_path),
        "frames_dir": str(frame_dir),
        "sample_frames_dir": str(sample_frame_dir),
        "pred_x0_frames_dir": str(pred_x0_frame_dir),
        "num_rendered_frames": len(rendered_frames),
        "num_rendered_pred_x0_frames": len(rendered_pred_x0_frames),
        "state_indices": state_indices,
        "pred_x0_state_indices": pred_x0_state_indices,
        "final_feature_mse": frame_records[-1]["feature_mse"] if frame_records else None,
        "final_pred_x0_feature_mse": pred_x0_frame_records[-1]["feature_mse"] if pred_x0_frame_records else None,
        "frames": frame_records,
        "pred_x0_frames": pred_x0_frame_records,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config is not None else run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = json.load(open(config_path, "r"))

    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / f"sampling_gif_step{ckpt_step:07d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        data_dir = json.loads(args.data_dir)
    else:
        if args.root is None:
            raise ValueError("Either --data_dir or --root must be provided.")
        data_dir = build_grid64_data_dir(Path(args.root).resolve(), args.split)

    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    dataset_args["return_origin"] = True
    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    metadata_filter_info = None
    if args.metadata_filter_csv:
        metadata_filter_info = apply_metadata_filter(dataset, args.metadata_filter_csv)
        print(f"Applied metadata filter: {metadata_filter_info['metadata_filter_original_size']} -> {len(dataset)}")

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"--sample_index {args.sample_index} is outside dataset size {len(dataset)}")

    instance_root, sha256 = dataset.instances[args.sample_index]
    item = dataset[args.sample_index]
    data = dataset.collate_fn([item])
    data = recursive_to_device(data, "cuda")

    model_dict = {
        name: getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda()
        for name, model_cfg in cfg["models"].items()
    }
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    trainer_args["skip_startup_snapshot"] = True
    trainer = getattr(trainers, cfg["trainer"]["name"])(
        model_dict,
        dataset,
        **trainer_args,
        output_dir=str(output_dir),
        load_dir=None,
        step=None,
    )
    ckpt_path = load_denoiser_checkpoint(trainer.models["denoiser"], run_dir, ckpt_step, args.ema_rate, trainer.device)
    trainer.models["denoiser"].eval()

    result = render_sampling_gif(
        trainer,
        dataset,
        data,
        output_dir,
        args.steps,
        args.frames,
        args.duration_ms,
        args.render_resolution,
        args.render_ssaa,
        include_gt=not args.no_gt,
    )

    manifest = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "dataset_size": len(dataset),
        "sample_index": args.sample_index,
        "sample_root": instance_root,
        "sha256": sha256,
        "num_tokens": int(data["x_0"].feats.shape[0]),
        "patch_origin": data["patch_origin"][0].detach().cpu().tolist() if "patch_origin" in data else None,
        "steps": args.steps,
        "requested_frames": args.frames,
        "duration_ms": args.duration_ms,
        "render_resolution": args.render_resolution,
        "render_ssaa": args.render_ssaa,
        "include_gt": not args.no_gt,
        **result,
    }
    if metadata_filter_info is not None:
        manifest.update(metadata_filter_info)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({
        "gif": result["gif"],
        "pred_x0_gif": result["pred_x0_gif"],
        "manifest": str(manifest_path),
        "checkpoint_step": ckpt_step,
        "sha256": sha256,
        "num_tokens": manifest["num_tokens"],
        "final_feature_mse": result["final_feature_mse"],
        "final_pred_x0_feature_mse": result["final_pred_x0_feature_mse"],
    }, indent=2))


if __name__ == "__main__":
    main()
