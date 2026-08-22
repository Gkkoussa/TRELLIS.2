import argparse
import copy
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from trellis2 import datasets, models, trainers
from trellis2.renderers import VoxelRenderer
from trellis2.representations import Voxel
from trellis2.utils.data_utils import recursive_to_device

from eval_gaussian_patch_sparse_flow import (
    build_camera,
    get_amp_context,
    make_sheet,
)
from eval_triangle_field_flow_l1_export import (
    compare_decoded,
    decoded_feats_01,
    find_ckpt_step,
    load_denoiser_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render GIFs of triangle-field flow sampling decoded through the triangle-field VAE."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint: latest, stepXXXXXXX, or integer step.")
    parser.add_argument("--ema_rate", type=str, default=None, help="Optional EMA rate, e.g. 0.9999.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <run_dir>/sampling_gif_step<step>.")
    parser.add_argument("--data_dir", type=str, required=True, help="JSON data_dir with canonical latent roots and filters.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_index", type=int, default=0, help="Dataset index after filtering.")
    parser.add_argument("--steps", type=int, default=50, help="Euler sampling steps.")
    parser.add_argument("--frames", type=int, default=26, help="Number of GIF frames including initial and final state.")
    parser.add_argument("--duration_ms", type=int, default=160, help="GIF frame duration in milliseconds.")
    parser.add_argument("--render_resolution", type=int, default=256, help="Per-view render resolution.")
    parser.add_argument("--render_ssaa", type=int, default=2, help="Render supersampling.")
    parser.add_argument("--guidance_strength", type=float, default=1.0, help="Classifier-free guidance strength.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_gt", action="store_true", help="Do not include GT d_tri/d_vert panels for reference.")
    return parser.parse_args()


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


def selected_state_indices(steps: int, frames: int) -> list[int]:
    if steps < 1:
        raise ValueError("--steps must be >= 1")
    if frames < 2:
        raise ValueError("--frames must be >= 2")
    return sorted(set(np.linspace(0, steps, min(frames, steps + 1)).round().astype(int).tolist()))


def render_triangle_field_panel(
    renderer,
    dataset,
    voxel,
    attr_slice: slice,
    extrinsics,
    intrinsics,
    resolution: int,
) -> np.ndarray:
    coords = voxel.coords[:, 1:].contiguous()
    image = torch.zeros(3, resolution * 2, resolution * 2, device=voxel.feats.device)
    if coords.shape[0] == 0:
        return np.zeros((resolution * 2, resolution * 2, 3), dtype=np.uint8)

    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / dataset.resolution,
        coords=coords,
        attrs=None,
        layout={"color": slice(0, 3)},
    )
    feats = decoded_feats_01(dataset, voxel).clamp(0, 1)
    attr = feats[:, attr_slice]
    if attr.shape[1] == 1:
        attr = attr.expand(-1, 3)
    else:
        attr = attr[:, :3]

    for j, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        with torch.autocast(device_type="cuda", enabled=False):
            res = renderer.render(rep, ext.float(), intr.float(), colors_overwrite=attr.float())
        row = j // 2
        col = j % 2
        image[:, resolution * row:resolution * (row + 1), resolution * col:resolution * (col + 1)] = res["color"]
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


@torch.no_grad()
def decode_one(dataset, z, shape_z):
    return dataset.decode_latent(z, shape_z=shape_z, batch_size=1)[0]


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
    guidance_strength: float,
    include_gt: bool,
) -> dict:
    x_0 = data["x_0"]
    shape_z = data["concat_cond"]
    inference = trainer.get_inference_cond(**{k: v for k, v in data.items() if k != "x_0"})
    cond = inference.pop("cond", None)
    kwargs = inference
    if "neg_cond" in kwargs:
        kwargs["guidance_strength"] = guidance_strength

    sampler = trainer.get_sampler()
    noise = x_0.replace(torch.randn_like(x_0.feats))
    sample = noise * sampler.noise_scale

    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = render_ssaa
    extrinsics, intrinsics = build_camera(trainer.device)

    gt_decoded = decode_one(dataset, x_0, shape_z)

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
        decoded = decode_one(dataset, state, shape_z)
        metrics = compare_decoded(dataset, gt_decoded, decoded)
        panels = [
            (f"{label_prefix} d_tri", render_triangle_field_panel(renderer, dataset, decoded, slice(0, 1), extrinsics, intrinsics, render_resolution)),
            (f"{label_prefix} d_vert", render_triangle_field_panel(renderer, dataset, decoded, slice(1, 2), extrinsics, intrinsics, render_resolution)),
        ]
        if include_gt:
            panels.extend([
                ("gt d_tri", render_triangle_field_panel(renderer, dataset, gt_decoded, slice(0, 1), extrinsics, intrinsics, render_resolution)),
                ("gt d_vert", render_triangle_field_panel(renderer, dataset, gt_decoded, slice(1, 2), extrinsics, intrinsics, render_resolution)),
            ])
        t_value = t_seq[state_idx]
        l1 = metrics["l1"] if metrics["l1"] is not None else float("nan")
        title = (
            f"step={state_idx:03d}/{steps:03d} t={t_value:.4f} "
            f"d_tri_l1={metrics['d_tri_l1'] if metrics['d_tri_l1'] is not None else float('nan'):.6f} "
            f"d_vert_l1={metrics['d_vert_l1'] if metrics['d_vert_l1'] is not None else float('nan'):.6f} "
            f"l1={l1:.6f}"
        )
        frame = make_sheet(panels, title)
        png_path = out_dir / f"frame_{len(frame_list):03d}_step{state_idx:03d}.png"
        frame.save(png_path)
        frame_list.append(frame)
        records.append({
            "frame": len(frame_list) - 1,
            "step": state_idx,
            "t": t_value,
            "d_tri_l1": metrics["d_tri_l1"],
            "d_vert_l1": metrics["d_vert_l1"],
            "l1": metrics["l1"],
            "coord_exact": metrics["coord_exact"],
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

    gif_path = output_dir / "sampling_noise_to_triangle_fields.gif"
    pred_x0_gif_path = output_dir / "pred_x0_noise_to_triangle_fields.gif"
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
        "final_l1": frame_records[-1]["l1"] if frame_records else None,
        "final_pred_x0_l1": pred_x0_frame_records[-1]["l1"] if pred_x0_frame_records else None,
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

    data_dir = json.loads(args.data_dir)
    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    print(f"Dataset size after filters: {len(dataset)}")
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
    trainer.p_uncond = 0.0
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
        args.guidance_strength,
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
        "steps": args.steps,
        "requested_frames": args.frames,
        "duration_ms": args.duration_ms,
        "render_resolution": args.render_resolution,
        "render_ssaa": args.render_ssaa,
        "guidance_strength": args.guidance_strength,
        "include_gt": not args.no_gt,
        **result,
    }

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
        "final_l1": result["final_l1"],
        "final_pred_x0_l1": result["final_pred_x0_l1"],
    }, indent=2))


if __name__ == "__main__":
    main()
