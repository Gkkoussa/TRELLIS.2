import argparse
import copy
import csv
import glob
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils3d

from trellis2 import datasets, models, trainers
from trellis2.representations import Voxel
from trellis2.renderers import VoxelRenderer
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a sparse Gaussian patch flow model.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint to evaluate: latest or integer step.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <run_dir>/eval_<split>_step<step>.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional JSON data_dir override.")
    parser.add_argument("--root", type=str, default=None, help="Processed dataset root used when --data_dir is omitted.")
    parser.add_argument("--split", type=str, default="test", help="Split name under <root>/splits/.")
    parser.add_argument("--metadata_filter_csv", type=str, default=None, help="Optional CSV with sha256 rows to keep.")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Evaluation dataloader workers.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of reconstruction samples to render.")
    parser.add_argument("--generated_samples", type=int, default=8, help="Number of pure-noise samples to render on GT coords.")
    parser.add_argument("--reconstruction_ts", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Euler steps for reconstruction/generation.")
    parser.add_argument("--render_resolution", type=int, default=256, help="Per-view render resolution.")
    parser.add_argument("--render_ssaa", type=int, default=2, help="Render supersampling.")
    parser.add_argument("--ema_rate", type=str, default=None, help="Optional EMA rate, e.g. 0.9999.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_t_values(value: str | None) -> list[float]:
    if value is None or value.strip().lower() in {"", "none", "skip", "off"}:
        return []
    values = [float(v.strip()) for v in value.split(",") if v.strip()]
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


def apply_metadata_filter(dataset, metadata_filter_csv: str) -> dict:
    with open(metadata_filter_csv, newline="") as f:
        allowed_sha256 = {row["sha256"] for row in csv.DictReader(f)}

    original_size = len(dataset.instances)
    old_instances = dataset.instances
    old_loads = getattr(dataset, "loads", None)
    if old_loads is not None and len(old_loads) == len(old_instances):
        kept = [(instance, load) for instance, load in zip(old_instances, old_loads) if instance[1] in allowed_sha256]
        dataset.instances = [instance for instance, _ in kept]
        dataset.loads = [load for _, load in kept]
    else:
        dataset.instances = [instance for instance in old_instances if instance[1] in allowed_sha256]

    if hasattr(dataset, "metadata") and len(dataset.metadata) > 0:
        keep_index = dataset.metadata.index.intersection(allowed_sha256)
        dataset.metadata = dataset.metadata.loc[keep_index]

    return {
        "metadata_filter_csv": str(Path(metadata_filter_csv).resolve()),
        "metadata_filter_allowed_sha256": len(allowed_sha256),
        "metadata_filter_original_size": original_size,
        "metadata_filter_removed": original_size - len(dataset.instances),
    }


def get_amp_context(trainer):
    if trainer.mix_precision_mode == "amp":
        return torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    return nullcontext()


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


def evaluate_flow_mse(trainer, loader, max_batches: int | None = None) -> dict:
    denoiser = trainer.training_models["denoiser"]
    denoiser.eval()
    total_mse_sum = 0.0
    total_mse_count = 0
    total_instances = 0
    total_tokens = 0
    bin_mse_sum = {i: 0.0 for i in range(10)}
    bin_count = {i: 0 for i in range(10)}

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc="Evaluating flow MSE")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = recursive_to_device(data, trainer.device)
            x_0 = data["x_0"]
            cond = data.get("cond", None)
            kwargs = {k: v for k, v in data.items() if k not in {"x_0", "cond"}}
            noise = x_0.replace(torch.randn_like(x_0.feats))
            t = trainer.sample_t(x_0.shape[0]).to(x_0.device).float()
            x_t = trainer.diffuse(x_0, t, noise=noise)
            model_cond = trainer.get_cond(cond, **kwargs)
            with get_amp_context(trainer):
                pred = denoiser(x_t, t * 1000, model_cond, **kwargs)
                target = trainer.get_v(x_0, noise, t)

            diff = (pred.feats.float() - target.feats.float()).pow(2)
            total_mse_sum += diff.sum().item()
            total_mse_count += diff.numel()
            total_instances += x_0.shape[0]
            total_tokens += pred.feats.shape[0]

            time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
            time_bin = np.clip(time_bin, 0, 9)
            for i in range(x_0.shape[0]):
                instance_mse = F.mse_loss(
                    pred.feats[x_0.layout[i]].float(),
                    target.feats[x_0.layout[i]].float(),
                ).item()
                b = int(time_bin[i])
                bin_mse_sum[b] += instance_mse
                bin_count[b] += 1

    return {
        "num_instances": total_instances,
        "num_tokens": total_tokens,
        "mse": total_mse_sum / total_mse_count if total_mse_count else None,
        "bins": {
            f"bin_{i}": {
                "mse": bin_mse_sum[i] / bin_count[i] if bin_count[i] else None,
                "count": bin_count[i],
            }
            for i in range(10)
        },
    }


@torch.no_grad()
def evaluate_reconstruction_feature_mse(
    trainer,
    loader,
    t_values: list[float],
    steps: int,
    max_batches: int | None = None,
) -> dict:
    denoiser = trainer.training_models["denoiser"]
    denoiser.eval()
    sampler = trainer.get_sampler()
    amp_context = get_amp_context(trainer)
    metrics = {}

    for t_value in t_values:
        total_mse_sum = 0.0
        total_mse_count = 0
        edge_mse_sum = 0.0
        edge_mse_count = 0
        vertex_mse_sum = 0.0
        vertex_mse_count = 0
        total_instances = 0
        total_tokens = 0

        for batch_idx, data in enumerate(tqdm(loader, desc=f"Evaluating reconstruction feature MSE t={t_value:.3f}")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = recursive_to_device(data, trainer.device)
            x_0 = data["x_0"]
            cond = data.get("cond", None)
            kwargs = {k: v for k, v in data.items() if k not in {"x_0", "cond"}}
            model_cond = trainer.get_cond(cond, **kwargs)

            noise = x_0.replace(torch.randn_like(x_0.feats))
            t = torch.full((x_0.shape[0],), t_value, device=x_0.device, dtype=torch.float32)
            sample = trainer.diffuse(x_0, t, noise=noise)

            t_seq = np.linspace(t_value, 0.0, steps + 1).tolist()
            for t_cur, t_prev in zip(t_seq[:-1], t_seq[1:]):
                with amp_context:
                    out = sampler.sample_once(
                        trainer.models["denoiser"],
                        sample,
                        t_cur,
                        t_prev,
                        model_cond,
                        **kwargs,
                    )
                sample = out.pred_x_prev

            diff = (sample.feats.float() - x_0.feats.float()).pow(2)
            total_mse_sum += diff.sum().item()
            total_mse_count += diff.numel()
            total_instances += x_0.shape[0]
            total_tokens += sample.feats.shape[0]

            if diff.shape[1] >= 3:
                edge = diff[:, :3]
                edge_mse_sum += edge.sum().item()
                edge_mse_count += edge.numel()
            if diff.shape[1] >= 6:
                vertex = diff[:, 3:6]
                vertex_mse_sum += vertex.sum().item()
                vertex_mse_count += vertex.numel()

        metrics[f"t_{t_value:.3f}"] = {
            "mse": total_mse_sum / total_mse_count if total_mse_count else None,
            "edge_mse": edge_mse_sum / edge_mse_count if edge_mse_count else None,
            "vertex_mse": vertex_mse_sum / vertex_mse_count if vertex_mse_count else None,
            "num_instances": total_instances,
            "num_tokens": total_tokens,
            "sampling_steps": steps,
        }

    return metrics


@torch.no_grad()
def evaluate_pure_noise_feature_mse(
    trainer,
    loader,
    steps: int,
    max_batches: int | None = None,
) -> dict:
    denoiser = trainer.training_models["denoiser"]
    denoiser.eval()
    sampler = trainer.get_sampler()
    total_mse_sum = 0.0
    total_mse_count = 0
    edge_mse_sum = 0.0
    edge_mse_count = 0
    vertex_mse_sum = 0.0
    vertex_mse_count = 0
    total_instances = 0
    total_tokens = 0

    for batch_idx, data in enumerate(tqdm(loader, desc="Evaluating pure-noise feature MSE")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data = recursive_to_device(data, trainer.device)
        x_0 = data["x_0"]
        cond = data.get("cond", None)
        kwargs = {k: v for k, v in data.items() if k not in {"x_0", "cond"}}
        model_cond = trainer.get_cond(cond, **kwargs)

        noise = x_0.replace(torch.randn_like(x_0.feats))
        sample = sampler.sample(
            trainer.models["denoiser"],
            noise=noise,
            cond=model_cond,
            steps=steps,
            verbose=False,
            **kwargs,
        ).samples

        diff = (sample.feats.float() - x_0.feats.float()).pow(2)
        total_mse_sum += diff.sum().item()
        total_mse_count += diff.numel()
        total_instances += x_0.shape[0]
        total_tokens += sample.feats.shape[0]

        if diff.shape[1] >= 3:
            edge = diff[:, :3]
            edge_mse_sum += edge.sum().item()
            edge_mse_count += edge.numel()
        if diff.shape[1] >= 6:
            vertex = diff[:, 3:6]
            vertex_mse_sum += vertex.sum().item()
            vertex_mse_count += vertex.numel()

    return {
        "mse": total_mse_sum / total_mse_count if total_mse_count else None,
        "edge_mse": edge_mse_sum / edge_mse_count if edge_mse_count else None,
        "vertex_mse": vertex_mse_sum / vertex_mse_count if vertex_mse_count else None,
        "num_instances": total_instances,
        "num_tokens": total_tokens,
        "sampling_steps": steps,
    }


def build_camera(device: torch.device):
    yaws = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    pitch = np.deg2rad(20.0)
    extrinsics = []
    intrinsics = []
    fov = torch.deg2rad(torch.tensor(30.0, device=device))
    for yaw in yaws:
        eye = torch.tensor(
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)],
            device=device,
            dtype=torch.float32,
        ) * 2.0
        extrinsics.append(
            utils3d.torch.extrinsics_look_at(
                eye,
                torch.tensor([0, 0, 0], device=device, dtype=torch.float32),
                torch.tensor([0, 0, 1], device=device, dtype=torch.float32),
            )
        )
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return extrinsics, intrinsics


def render_sparse_views(
    renderer,
    sparse_tensor,
    sample_idx: int,
    attr_slice: slice,
    patch_size: int,
    extrinsics,
    intrinsics,
    resolution: int,
) -> np.ndarray:
    x = sparse_tensor[sample_idx]
    coords = x.coords[:, 1:].contiguous()
    feats = x.feats.float()
    image = torch.zeros(3, resolution * 2, resolution * 2, device=feats.device)
    if coords.shape[0] == 0:
        return np.zeros((resolution * 2, resolution * 2, 3), dtype=np.uint8)

    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / patch_size,
        coords=coords,
        attrs=None,
        layout={"color": slice(0, 3)},
    )
    attr = ((feats[:, attr_slice] + 1.0) * 0.5).clamp(0, 1)
    if attr.shape[1] == 1:
        attr = attr.expand(-1, 3)
    elif attr.shape[1] > 3:
        attr = attr[:, :3]

    for j, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        res = renderer.render(rep, ext, intr, colors_overwrite=attr)
        row = j // 2
        col = j % 2
        image[:, resolution * row:resolution * (row + 1), resolution * col:resolution * (col + 1)] = res["color"]
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def make_sheet(panels: list[tuple[str, np.ndarray]], title: str) -> Image.Image:
    panel_h, panel_w = panels[0][1].shape[:2]
    label_h = 24
    header_h = 36
    cols = len(panels)
    canvas = Image.new("RGB", (cols * panel_w, header_h + panel_h + label_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)
    for col, (label, image) in enumerate(panels):
        x0 = col * panel_w
        canvas.paste(Image.fromarray(image, mode="RGB"), (x0, header_h))
        draw.text((x0 + 8, header_h + panel_h + 6), label, fill=(255, 255, 255), font=font)
    return canvas


@torch.no_grad()
def sample_from_t(trainer, x_t, start_t: float, cond, kwargs: dict, steps: int, desc: str):
    sampler = trainer.get_sampler()
    sample = x_t
    t_seq = np.linspace(start_t, 0.0, steps + 1).tolist()
    amp_context = get_amp_context(trainer)
    for t, t_prev in tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc=desc, leave=False):
        with amp_context:
            out = sampler.sample_once(trainer.models["denoiser"], sample, t, t_prev, cond, **kwargs)
        sample = out.pred_x_prev
    return sample


@torch.no_grad()
def save_visualizations(trainer, dataset, output_dir: Path, num_samples: int, generated_samples: int, t_values: list[float], steps: int, render_resolution: int, render_ssaa: int):
    loader = DataLoader(
        copy.deepcopy(dataset),
        batch_size=max(num_samples, generated_samples),
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    data = recursive_to_device(next(iter(loader)), trainer.device)
    x_0 = data["x_0"]
    kwargs = {k: v for k, v in data.items() if k not in {"x_0", "cond"}}
    cond = trainer.get_inference_cond(**{k: v for k, v in data.items() if k != "x_0"})["cond"]

    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = render_ssaa
    extrinsics, intrinsics = build_camera(trainer.device)

    recon_dir = output_dir / "reconstructions"
    gen_dir = output_dir / "generated_samples"
    recon_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    recon_metrics = {}
    for t_value in t_values:
        take = min(num_samples, x_0.shape[0])
        x = x_0[:take]
        batch_cond = cond[:take] if isinstance(cond, torch.Tensor) else cond
        batch_kwargs = {k: v[:take] if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        noise = x.replace(torch.randn_like(x.feats))
        t = torch.full((x.shape[0],), t_value, device=x.device, dtype=torch.float32)
        x_t = trainer.diffuse(x, t, noise=noise)
        recon = sample_from_t(trainer, x_t, t_value, batch_cond, batch_kwargs, steps, f"Recon t={t_value:.2f}")
        mse = F.mse_loss(recon.feats.float(), x.feats.float()).item() if recon.feats.numel() else None
        recon_metrics[f"t_{t_value:.3f}"] = {"mse": mse, "num_tokens": int(recon.feats.shape[0])}

        for i in range(take):
            panels = [
                ("gt edge", render_sparse_views(renderer, x, i, slice(0, 3), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("gt vertex", render_sparse_views(renderer, x, i, slice(3, 6), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("noisy edge", render_sparse_views(renderer, x_t, i, slice(0, 3), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("noisy vertex", render_sparse_views(renderer, x_t, i, slice(3, 6), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("recon edge", render_sparse_views(renderer, recon, i, slice(0, 3), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("recon vertex", render_sparse_views(renderer, recon, i, slice(3, 6), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
            ]
            title = f"sample={i} t={t_value:.3f} mse={mse if mse is not None else float('nan'):.6f}"
            make_sheet(panels, title).save(recon_dir / f"sample_{i:03d}_t{t_value:.3f}.png")

    if generated_samples > 0:
        take = min(generated_samples, x_0.shape[0])
        x = x_0[:take]
        batch_cond = cond[:take] if isinstance(cond, torch.Tensor) else cond
        batch_kwargs = {k: v[:take] if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        noise = x.replace(torch.randn_like(x.feats))
        with get_amp_context(trainer):
            generated = trainer.get_sampler().sample(
                trainer.models["denoiser"],
                noise=noise,
                cond=batch_cond,
                steps=steps,
                verbose=True,
                tqdm_desc="Generating on GT sparse coords",
                **batch_kwargs,
            ).samples
        for i in range(take):
            panels = [
                ("gt edge", render_sparse_views(renderer, x, i, slice(0, 3), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("gt vertex", render_sparse_views(renderer, x, i, slice(3, 6), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("generated edge", render_sparse_views(renderer, generated, i, slice(0, 3), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
                ("generated vertex", render_sparse_views(renderer, generated, i, slice(3, 6), dataset.patch_size, extrinsics, intrinsics, render_resolution)),
            ]
            make_sheet(panels, f"generated sample={i} coords=GT sparse support").save(gen_dir / f"sample_{i:03d}.png")

    return recon_metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config is not None else run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = json.load(open(config_path, "r"))

    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    dataset_args["return_origin"] = True
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    trainer_args["skip_startup_snapshot"] = True

    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / f"eval_{args.split}_step{ckpt_step:07d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        data_dir = json.loads(args.data_dir)
    else:
        if args.root is None:
            raise ValueError("Either --data_dir or --root must be provided.")
        data_dir = build_data_dir(Path(args.root).resolve(), args.split)

    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    metadata_filter_info = None
    if args.metadata_filter_csv:
        metadata_filter_info = apply_metadata_filter(dataset, args.metadata_filter_csv)
        print(f"Applied metadata filter: {metadata_filter_info['metadata_filter_original_size']} -> {len(dataset)}")

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
    ckpt_path = load_denoiser_checkpoint(trainer.models["denoiser"], run_dir, ckpt_step, args.ema_rate, trainer.device)
    trainer.models["denoiser"].eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=dataset.collate_fn,
    )
    metrics = evaluate_flow_mse(trainer, loader, max_batches=args.max_batches)
    reconstruction_t_values = parse_t_values(args.reconstruction_ts)
    if reconstruction_t_values:
        metrics["full_reconstruction_feature_mse"] = evaluate_reconstruction_feature_mse(
            trainer,
            loader,
            reconstruction_t_values,
            args.sampling_steps,
            max_batches=args.max_batches,
        )
    else:
        metrics["full_reconstruction_feature_mse"] = {}
    metrics["pure_noise_feature_mse"] = evaluate_pure_noise_feature_mse(
        trainer,
        loader,
        args.sampling_steps,
        max_batches=args.max_batches,
    )
    metrics.update({
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "dataset_size": len(dataset),
        "dataset_stats": getattr(dataset, "_stats", None),
        "max_batches": args.max_batches,
    })
    if metadata_filter_info is not None:
        metrics.update(metadata_filter_info)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved flow/pure-noise metrics to {metrics_path}")

    recon_metrics = save_visualizations(
        trainer,
        dataset,
        output_dir,
        args.num_samples,
        args.generated_samples,
        reconstruction_t_values,
        args.sampling_steps,
        args.render_resolution,
        args.render_ssaa,
    )
    metrics["reconstructions"] = recon_metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved reconstructions to {output_dir / 'reconstructions'}")
    print(f"Saved generated samples to {output_dir / 'generated_samples'}")


if __name__ == "__main__":
    main()
