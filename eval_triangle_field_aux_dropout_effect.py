import argparse
import copy
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trellis2 import datasets, models
from trellis2.modules import sparse as sp
from trellis2.utils.data_utils import recursive_to_device
from eval_metadata_filters import add_eval_metadata_filter_args, attach_eval_metadata_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare triangle-field VAE behavior with full input features vs "
            "zeroed auxiliary features for before/after checkpoints."
        )
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--before_run_dir", type=str, required=True)
    parser.add_argument("--before_ckpt", type=str, default="latest")
    parser.add_argument("--after_run_dir", type=str, required=True)
    parser.add_argument("--after_ckpt", type=str, default="latest")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero_from_channel", type=int, default=2)
    parser.add_argument("--dataset_resolution", type=int, default=None)
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def find_ckpt_step(run_dir: Path, ckpt: str) -> str:
    if ckpt != "latest":
        return ckpt
    files = glob.glob(str(run_dir / "ckpts" / "encoder_step*.pt"))
    if not files:
        raise FileNotFoundError(f"No encoder_step*.pt checkpoints found in {run_dir / 'ckpts'}")
    return max(
        os.path.basename(path).split("encoder_")[-1].split(".")[0]
        for path in files
    )


def build_data_dir(root: Path, split: str, dataset_args: dict, args=None) -> dict:
    voxel_root_key = dataset_args["voxel_root_key"]
    voxel_dirname = dataset_args["voxel_dirname"]
    resolution = dataset_args["resolution"]

    split_root = root / "splits" / split
    filtered_base = root / "splits" / f"{split}_triangle_field_{resolution}"
    base_root = filtered_base if (filtered_base / "metadata.csv").exists() else split_root
    voxel_root = split_root / f"{voxel_dirname}_{resolution}"

    data_dir = {
        split: {
            "base": str(base_root),
            voxel_root_key: str(voxel_root),
        }
    }
    return attach_eval_metadata_filter(data_dir, root, split, args)


def load_vae(run_dir: Path, ckpt: str):
    cfg = json.load(open(run_dir / "config.json", "r"))
    step = find_ckpt_step(run_dir, ckpt)
    encoder = getattr(models, cfg["models"]["encoder"]["name"])(**cfg["models"]["encoder"]["args"]).cuda().eval()
    decoder = getattr(models, cfg["models"]["decoder"]["name"])(**cfg["models"]["decoder"]["args"]).cuda().eval()
    encoder.load_state_dict(
        torch.load(run_dir / "ckpts" / f"encoder_{step}.pt", map_location="cpu", weights_only=True),
        strict=False,
    )
    decoder.load_state_dict(
        torch.load(run_dir / "ckpts" / f"decoder_{step}.pt", map_location="cpu", weights_only=True),
        strict=False,
    )
    return cfg, step, encoder, decoder


def zero_aux_features(x: sp.SparseTensor, zero_from_channel: int) -> sp.SparseTensor:
    feats = x.feats.clone()
    feats[:, zero_from_channel:] = 0
    return x.replace(feats)


def maybe_inverse_distance_transform(values: torch.Tensor, distance_transform: str) -> torch.Tensor:
    if distance_transform == "minus_one_one":
        return values * 0.5 + 0.5
    if distance_transform == "none":
        return values
    raise ValueError(f"Unsupported distance_transform: {distance_transform}")


def reconstruction_metrics(pred: sp.SparseTensor, target: sp.SparseTensor, distance_transform: str):
    if not torch.equal(pred.coords, target.coords):
        raise RuntimeError("Predicted and target sparse supports differ; cannot compare row-wise.")
    pred_values = maybe_inverse_distance_transform(pred.feats.float(), distance_transform)
    target_values = maybe_inverse_distance_transform(target.feats.float(), distance_transform)
    err = pred_values - target_values
    return {
        "l1_d_tri_sum": err[:, 0].abs().sum().detach().cpu().double(),
        "l1_d_vert_sum": err[:, 1].abs().sum().detach().cpu().double(),
        "l2_d_tri_sum": err[:, 0].square().sum().detach().cpu().double(),
        "l2_d_vert_sum": err[:, 1].square().sum().detach().cpu().double(),
        "tokens": int(err.shape[0]),
    }


def latent_metrics(z_full: sp.SparseTensor, z_no_aux: sp.SparseTensor):
    if not torch.equal(z_full.coords, z_no_aux.coords):
        raise RuntimeError("Full and no-aux latent supports differ; cannot compare row-wise.")
    full = z_full.feats.float()
    no_aux = z_no_aux.feats.float()
    diff = no_aux - full
    full_norm = torch.linalg.norm(full, dim=1)
    no_aux_norm = torch.linalg.norm(no_aux, dim=1)
    cosine = torch.nn.functional.cosine_similarity(full, no_aux, dim=1)
    return {
        "latent_l1_sum": diff.abs().sum().detach().cpu().double(),
        "latent_l2_sum": diff.square().sum().detach().cpu().double(),
        "latent_values": int(diff.numel()),
        "latent_token_rmse_sum": diff.square().mean(dim=1).sqrt().sum().detach().cpu().double(),
        "latent_tokens": int(diff.shape[0]),
        "latent_cosine_sum": cosine.sum().detach().cpu().double(),
        "latent_full_norm_sum": full_norm.sum().detach().cpu().double(),
        "latent_no_aux_norm_sum": no_aux_norm.sum().detach().cpu().double(),
    }


def add_sums(dst: dict, src: dict):
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            value = float(value.item())
        dst[key] = dst.get(key, 0.0) + value


def finalize_model_stats(acc: dict):
    out = {}
    for variant in ("full", "no_aux"):
        tokens = max(acc[variant].get("tokens", 0), 1)
        out[variant] = {
            "l1_d_tri": acc[variant]["l1_d_tri_sum"] / tokens,
            "l1_d_vert": acc[variant]["l1_d_vert_sum"] / tokens,
            "rmse_d_tri": np.sqrt(acc[variant]["l2_d_tri_sum"] / tokens),
            "rmse_d_vert": np.sqrt(acc[variant]["l2_d_vert_sum"] / tokens),
            "tokens": int(acc[variant].get("tokens", 0)),
        }
    latent_values = max(acc["latent"].get("latent_values", 0), 1)
    latent_tokens = max(acc["latent"].get("latent_tokens", 0), 1)
    out["latent_full_vs_no_aux"] = {
        "mean_abs_diff_per_value": acc["latent"]["latent_l1_sum"] / latent_values,
        "rmse_per_value": np.sqrt(acc["latent"]["latent_l2_sum"] / latent_values),
        "mean_token_rmse": acc["latent"]["latent_token_rmse_sum"] / latent_tokens,
        "mean_cosine_similarity": acc["latent"]["latent_cosine_sum"] / latent_tokens,
        "mean_full_norm": acc["latent"]["latent_full_norm_sum"] / latent_tokens,
        "mean_no_aux_norm": acc["latent"]["latent_no_aux_norm_sum"] / latent_tokens,
        "latent_tokens": int(acc["latent"].get("latent_tokens", 0)),
    }
    out["no_aux_minus_full"] = {
        "l1_d_tri": out["no_aux"]["l1_d_tri"] - out["full"]["l1_d_tri"],
        "l1_d_vert": out["no_aux"]["l1_d_vert"] - out["full"]["l1_d_vert"],
        "rmse_d_tri": out["no_aux"]["rmse_d_tri"] - out["full"]["rmse_d_tri"],
        "rmse_d_vert": out["no_aux"]["rmse_d_vert"] - out["full"]["rmse_d_vert"],
    }
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    before_run_dir = Path(args.before_run_dir)
    after_run_dir = Path(args.after_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_cfg, before_step, before_encoder, before_decoder = load_vae(before_run_dir, args.before_ckpt)
    after_cfg, after_step, after_encoder, after_decoder = load_vae(after_run_dir, args.after_ckpt)

    dataset_args = copy.deepcopy(before_cfg["dataset"]["args"])
    if args.dataset_resolution is not None:
        dataset_args["resolution"] = args.dataset_resolution
    data_dir = json.loads(args.data_dir) if args.data_dir is not None else build_data_dir(Path(args.root), args.split, dataset_args, args=args)
    dataset = getattr(datasets, before_cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    distance_transform = dataset_args.get("distance_transform", "none")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    acc = {
        "before": {"full": {}, "no_aux": {}, "latent": {}},
        "after": {"full": {}, "no_aux": {}, "latent": {}},
    }
    processed_samples = 0

    model_pairs = {
        "before": (before_encoder, before_decoder),
        "after": (after_encoder, after_decoder),
    }

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Evaluating aux-feature removal")
        for batch in loader:
            if processed_samples >= args.num_samples:
                break
            take = min(args.num_samples - processed_samples, batch["x"].shape[0])
            batch = {key: value[:take] if hasattr(value, "__getitem__") else value for key, value in batch.items()}
            batch = recursive_to_device(batch, torch.device("cuda"))
            x_full = batch["x"]
            x_no_aux = zero_aux_features(x_full, args.zero_from_channel)
            target = batch["target"]

            for name, (encoder, decoder) in model_pairs.items():
                z_full = encoder(x_full, sample_posterior=False)
                z_no_aux = encoder(x_no_aux, sample_posterior=False)
                y_full = decoder(z_full)
                y_no_aux = decoder(z_no_aux)
                add_sums(acc[name]["full"], reconstruction_metrics(y_full, target, distance_transform))
                add_sums(acc[name]["no_aux"], reconstruction_metrics(y_no_aux, target, distance_transform))
                add_sums(acc[name]["latent"], latent_metrics(z_full, z_no_aux))

            processed_samples += take
            pbar.update(take)
        pbar.close()

    summary = {
        "root": str(args.root),
        "split": args.split,
        "num_samples": processed_samples,
        "zero_from_channel": args.zero_from_channel,
        "before": {
            "run_dir": str(before_run_dir),
            "ckpt": before_step,
            "metrics": finalize_model_stats(acc["before"]),
        },
        "after": {
            "run_dir": str(after_run_dir),
            "ckpt": after_step,
            "metrics": finalize_model_stats(acc["after"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
