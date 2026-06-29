import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm import tqdm

from trellis2 import datasets, models
from trellis2.modules import sparse as sp
from trellis2.utils.data_utils import recursive_to_device
from eval_metadata_filters import add_eval_metadata_filter_args, attach_eval_metadata_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample the triangle-field SR flow with a custom Euler start timestep."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=90000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--t_start", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--low_resolution", type=int, default=64)
    parser.add_argument("--high_resolution", type=int, default=128)
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.sqrt(images.shape[0])))
    tv_utils.save_image(images, str(path), nrow=nrow)


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def build_dataset(cfg: dict, root: Path, split: str, low_resolution: int, high_resolution: int, args=None):
    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    dataset_args["low_resolution"] = low_resolution
    dataset_args["high_resolution"] = high_resolution
    dataset_args["return_area_offsets"] = False
    dataset_args.pop("conditioning_augmentation", None)

    data_dir = {
        "objxl4k_filtered": {
            "low_triangle_field_voxel": str(root / f"triangle_field_voxels_{low_resolution}"),
            "high_triangle_field_voxel": str(root / f"triangle_field_voxels_{high_resolution}"),
        }
    }
    data_dir = attach_eval_metadata_filter(data_dir, root, split, args)
    return getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)


def load_model(cfg: dict, run_dir: Path, ckpt: int, device: torch.device):
    encoder_cfg = cfg["models"]["encoder"]
    encoder = getattr(models, encoder_cfg["name"])(**encoder_cfg["args"]).to(device).eval()
    encoder_path = run_dir / "ckpts" / f"encoder_step{ckpt:07d}.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))

    trainer_args = cfg["trainer"]["args"]
    decoder_cfg = trainer_args["decoder_model"]
    decoder = getattr(models, decoder_cfg["name"])(**decoder_cfg["args"]).to(device).eval()
    decoder_path = Path(trainer_args["decoder_ckpt"])
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_path}")
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))

    return encoder, decoder, float(trainer_args.get("sigma_min", 1e-5))


def predict_x0(encoder, decoder, x_t: sp.SparseTensor, cond: sp.SparseTensor, t: float):
    if not torch.equal(x_t.coords, cond.coords):
        raise ValueError(f"x_t and cond coords must match: {x_t.coords.shape} vs {cond.coords.shape}")
    batch_t = torch.full((x_t.shape[0],), t * 1000.0, device=x_t.feats.device, dtype=torch.float32)
    enc_in = x_t.replace(torch.cat([x_t.feats, cond.feats], dim=-1))
    z = encoder(enc_in, batch_t, sample_posterior=False)
    y = decoder(z)
    if not torch.equal(y.coords, x_t.coords):
        raise ValueError(f"Decoded coords must match x_t coords: {y.coords.shape} vs {x_t.coords.shape}")
    return y


@torch.no_grad()
def sample_custom_t_start(
    encoder,
    decoder,
    x_0: sp.SparseTensor,
    cond: sp.SparseTensor,
    sigma_min: float,
    steps: int,
    t_start: float,
):
    sample = x_0.replace(torch.randn_like(x_0.feats))
    zero_cond = cond.replace(torch.zeros_like(cond.feats))
    t_seq = np.linspace(t_start, 0.0, steps + 1).tolist()
    pred_x0_last = None
    for t, t_prev in tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc=f"Sampling t={t_start:g}->0"):
        pred_x0 = predict_x0(encoder, decoder, sample, zero_cond, float(t))
        denom = sigma_min + (1.0 - sigma_min) * float(t)
        pred_v = ((1.0 - sigma_min) * sample.feats - pred_x0.feats) / denom
        sample = sample.replace(sample.feats - (float(t) - float(t_prev)) * pred_v)
        pred_x0_last = pred_x0
    return sample, zero_cond, pred_x0_last


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    run_dir = Path(args.run_dir).resolve()
    root = Path(args.root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / f"eval_filtered_uncond{args.high_resolution}_tstart{args.t_start:g}_step{args.ckpt:07d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(run_dir)
    dataset = build_dataset(cfg, root, args.split, args.low_resolution, args.high_resolution, args=args)
    encoder, decoder, sigma_min = load_model(cfg, run_dir, args.ckpt, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    gt_images = {}
    zero_cond_images = {}
    sample_images = {}
    pred_x0_images = {}
    remaining = args.num_samples
    for data in loader:
        batch = min(remaining, args.batch_size)
        data = {k: v[:batch] if hasattr(v, "__getitem__") and not isinstance(v, str) else v for k, v in data.items()}
        data = recursive_to_device(data, device)
        sample, zero_cond, pred_x0 = sample_custom_t_start(
            encoder,
            decoder,
            data["x_0"],
            data["cond"],
            sigma_min,
            args.steps,
            args.t_start,
        )

        for name, tensor, store in (
            ("gt", data["x_0"], gt_images),
            ("zero_cond", zero_cond, zero_cond_images),
            ("sample", sample, sample_images),
            ("pred_x0_last", pred_x0, pred_x0_images),
        ):
            vis = dataset.visualize_sample({"target": tensor})
            for key, value in vis.items():
                store.setdefault(key, []).append(value[:batch].cpu())

        remaining -= batch
        if remaining <= 0:
            break

    suffix = f"step{args.ckpt:07d}_uncond{args.high_resolution}_tstart{args.t_start:g}"
    for prefix, images in (
        ("gt", gt_images),
        ("zero_cond", zero_cond_images),
        ("sample", sample_images),
        ("pred_x0_last", pred_x0_images),
    ):
        for key, chunks in images.items():
            save_image_grid(torch.cat(chunks, dim=0)[:args.num_samples], output_dir / f"{prefix}_{key}_{suffix}.jpg")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": args.ckpt,
        "root": str(root),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "t_start": args.t_start,
        "low_resolution": args.low_resolution,
        "high_resolution": args.high_resolution,
        "sigma_min": sigma_min,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
