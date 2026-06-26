import argparse
import copy
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm import tqdm

from trellis2 import datasets, models, trainers
from trellis2.modules import sparse as sp
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a latent-space triangle-field super-resolution flow model."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance_strength", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--low_resolution", type=int, default=64)
    parser.add_argument("--high_resolution", type=int, default=128)
    parser.add_argument(
        "--latent_name",
        type=str,
        default=None,
        help="Latent directory name under <root>/triangle_field_latents. Defaults to config data_dir.",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Optional renderer resolution override for saved grids.",
    )
    return parser.parse_args()


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        files = glob.glob(str(run_dir / "ckpts" / "misc_step*.pt"))
        if not files:
            files = glob.glob(str(run_dir / "ckpts" / "encoder_step*.pt"))
        if not files:
            raise RuntimeError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files)
    if ckpt == "none":
        raise ValueError("ckpt=none is not valid for evaluation.")
    return int(ckpt)


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.sqrt(images.shape[0])))
    tv_utils.save_image(images, str(path), nrow=nrow)


def build_data_dir(cfg: dict, root: Path, low_resolution: int, high_resolution: int, latent_name: str | None) -> dict:
    if latent_name is None:
        configured = cfg.get("data_dir", {})
        if isinstance(configured, str):
            configured = json.loads(configured)
        try:
            latent_root = Path(configured["objxl4k_filtered"]["triangle_field_latent"])
        except KeyError as exc:
            raise ValueError("Could not infer latent root from config data_dir; pass --latent_name.") from exc
    else:
        latent_root = root / "triangle_field_latents" / latent_name

    return {
        "objxl4k_filtered": {
            "low_triangle_field_voxel": str(root / f"triangle_field_voxels_{low_resolution}"),
            "high_triangle_field_voxel": str(root / f"triangle_field_voxels_{high_resolution}"),
            "triangle_field_latent": str(latent_root),
        }
    }


def build_dataset(cfg: dict, root: Path, args):
    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    dataset_args["low_resolution"] = args.low_resolution
    dataset_args["high_resolution"] = args.high_resolution
    dataset_args["return_area_offsets"] = False
    dataset_args.pop("conditioning_augmentation", None)
    if args.render_resolution is not None:
        dataset_args["snapshot_render_resolution"] = args.render_resolution

    data_dir = build_data_dir(cfg, root, args.low_resolution, args.high_resolution, args.latent_name)
    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    split_instances = root / "splits" / args.split / "instances.txt"
    if split_instances.exists():
        keep = {line.strip() for line in split_instances.read_text().splitlines() if line.strip()}
        dataset.instances = [(item_root, sha256) for item_root, sha256 in dataset.instances if sha256 in keep]
        if len(dataset.metadata) > 0:
            dataset.metadata = dataset.metadata[dataset.metadata.index.isin(keep)]
        if hasattr(dataset, "loads"):
            dataset.loads = [
                dataset.metadata.loc[sha256, dataset.num_voxels_column]
                if dataset.num_voxels_column in dataset.metadata.columns else 1
                for _, sha256 in dataset.instances
            ]
        for stats in getattr(dataset, "_stats", {}).values():
            stats[f"Restricted to split/{args.split}"] = len(dataset.instances)
    else:
        raise FileNotFoundError(f"Split instances file not found: {split_instances}")
    return dataset, data_dir


def build_trainer(cfg: dict, dataset, output_dir: Path):
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    trainer_args["finetune_ckpt"] = None
    trainer_args["num_workers"] = 1
    trainer_args["parallel_mode"] = None
    trainer_args["prefetch_data"] = False
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
    trainer.models["encoder"].eval()
    trainer.decoder.eval()
    return trainer


def load_encoder_checkpoint(trainer, run_dir: Path, step: int, ema_rate: str | None) -> str:
    if ema_rate is None:
        path = run_dir / "ckpts" / f"encoder_step{step:07d}.pt"
    else:
        path = run_dir / "ckpts" / f"encoder_ema{ema_rate}_step{step:07d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    state = torch.load(path, map_location=trainer.device, weights_only=True)
    trainer.models["encoder"].load_state_dict(state)
    trainer.models["encoder"].eval()
    return str(path)


def slice_batch(data: dict, batch: int) -> dict:
    sliced = {}
    for key, value in data.items():
        if isinstance(value, str):
            sliced[key] = value
        elif isinstance(value, list):
            sliced[key] = value[:batch]
        elif hasattr(value, "__getitem__"):
            sliced[key] = value[:batch]
        else:
            sliced[key] = value
    return sliced


@torch.no_grad()
def predict_z0(
    trainer,
    z_t: sp.SparseTensor,
    cond: sp.SparseTensor,
    caches,
    cache_paths,
    t: float,
) -> sp.SparseTensor:
    x_t = trainer._decode_latents_with_cache(z_t, caches=caches, cache_paths=cache_paths)
    if not torch.equal(x_t.coords, cond.coords):
        raise ValueError(f"Decoded z_t coords must match cond coords: {x_t.coords.shape} vs {cond.coords.shape}")
    enc_in = trainer._make_encoder_input(x_t, cond)
    batch_t = torch.full((z_t.shape[0],), t * 1000.0, device=z_t.feats.device, dtype=torch.float32)
    pred_z0 = trainer.models["encoder"](enc_in, batch_t, sample_posterior=False)
    if not torch.equal(pred_z0.coords, z_t.coords):
        raise ValueError(f"Predicted latent coords must match z_t coords: {pred_z0.coords.shape} vs {z_t.coords.shape}")
    return pred_z0


@torch.no_grad()
def sample_latent_sr(
    trainer,
    z_0: sp.SparseTensor,
    cond: sp.SparseTensor,
    caches,
    cache_paths,
    steps: int,
    guidance_strength: float,
):
    z_t = z_0.replace(torch.randn_like(z_0.feats))
    zero_cond = cond.replace(torch.zeros_like(cond.feats))
    t_seq = np.linspace(1.0, 0.0, steps + 1).tolist()
    pred_z0_last = None
    for t, t_prev in tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc="Sampling latent SR"):
        pred_pos = predict_z0(trainer, z_t, cond, caches, cache_paths, float(t))
        if guidance_strength == 1.0:
            pred_z0 = pred_pos
        else:
            pred_neg = predict_z0(trainer, z_t, zero_cond, caches, cache_paths, float(t))
            pred_z0 = pred_pos.replace(
                guidance_strength * pred_pos.feats + (1.0 - guidance_strength) * pred_neg.feats
            )
        velocity = (z_t.feats - pred_z0.feats) / max(float(t), 1e-5)
        z_t = z_t.replace(z_t.feats - (float(t) - float(t_prev)) * velocity)
        pred_z0_last = pred_z0
    return z_t, pred_z0_last


def append_visuals(dataset, store: dict, prefix: str, tensor: sp.SparseTensor, batch: int) -> None:
    vis = dataset.visualize_sample({"target": tensor})
    for key, value in vis.items():
        store.setdefault(f"{prefix}_{key}", []).append(value[:batch].cpu())


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
        / f"eval_filtered_{args.split}_{args.low_resolution}to{args.high_resolution}_latent_flow_sampling_step{ckpt_step:07d}_cfg{args.guidance_strength:g}_n{args.num_samples}"
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
    for data in loader:
        batch = min(remaining, args.batch_size)
        data = recursive_to_device(slice_batch(data, batch), trainer.device)
        caches = data.get("triangle_field_slat_cache", None)
        cache_paths = data.get("triangle_field_slat_cache_path", None)

        sample_z, pred_z0_last = sample_latent_sr(
            trainer,
            data["z_0"],
            data["cond"],
            caches,
            cache_paths,
            args.steps,
            args.guidance_strength,
        )
        gt = trainer._decode_latents_with_cache(data["z_0"], caches=caches, cache_paths=cache_paths)
        sample = trainer._decode_latents_with_cache(sample_z, caches=caches, cache_paths=cache_paths)
        pred_last = trainer._decode_latents_with_cache(pred_z0_last, caches=caches, cache_paths=cache_paths)

        append_visuals(dataset, images, "gt", gt, batch)
        append_visuals(dataset, images, "cond", data["cond"], batch)
        append_visuals(dataset, images, "sample", sample, batch)
        append_visuals(dataset, images, "pred_z0_last", pred_last, batch)

        remaining -= batch
        if remaining <= 0:
            break

    suffix = f"step{ckpt_step:07d}_cfg{args.guidance_strength:g}_{args.low_resolution}to{args.high_resolution}"
    for name, chunks in images.items():
        save_image_grid(torch.cat(chunks, dim=0)[: args.num_samples], output_dir / f"{name}_{suffix}.jpg")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "root": str(root),
        "split": args.split,
        "data_dir": data_dir,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "steps": args.steps,
        "guidance_strength": args.guidance_strength,
        "low_resolution": args.low_resolution,
        "high_resolution": args.high_resolution,
        "latent_name": args.latent_name,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
