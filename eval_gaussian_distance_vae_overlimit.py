import os
import json
import math
import copy
import glob
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm import tqdm

from trellis2 import models, datasets, trainers
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Gaussian-distance VAE on assets above the VAE training active-voxel cap."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint to evaluate: latest or an integer step.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <run_dir>/eval_<split>_over<min>_step<step>.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional JSON data_dir override.")
    parser.add_argument("--root", type=str, default=None, help="Processed dataset root used when --data_dir is omitted.")
    parser.add_argument("--split", type=str, default="train", help="Split name under <root>/splits/.")
    parser.add_argument("--min_active_voxels", type=int, default=1_000_000, help="Keep assets with voxel count strictly above this.")
    parser.add_argument("--max_active_voxels", type=int, default=100_000_000, help="Temporary upper cap used to bypass the training cap.")
    parser.add_argument("--max_instances", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size. Large over-limit assets usually need 1.")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--num_samples", type=int, default=16, help="Visualization samples to save via trainer.run_snapshot().")
    parser.add_argument("--snapshot_batch_size", type=int, default=1, help="Batch size used by trainer.run_snapshot().")
    parser.add_argument("--render_resolution", type=int, default=None, help="Override snapshot render resolution.")
    parser.add_argument("--deterministic_posterior", action="store_true", help="Use posterior mean instead of sampling.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        files = glob.glob(str(run_dir / "ckpts" / "misc_*.pt"))
        if not files:
            raise RuntimeError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files)
    return int(ckpt.removeprefix("step"))


def build_data_dir(root: Path, split: str, dataset_args: dict) -> dict:
    voxel_root_key = dataset_args["voxel_root_key"]
    voxel_dirname = dataset_args["voxel_dirname"]
    resolution = dataset_args["resolution"]
    split_root = root / "splits" / split
    voxel_root = split_root / f"{voxel_dirname}_{resolution}"
    return {
        split: {
            "base": str(split_root),
            voxel_root_key: str(voxel_root),
        }
    }


def restrict_to_overlimit(dataset, count_column: str, min_active_voxels: int, max_instances: int | None):
    kept = []
    for root, sha256 in dataset.instances:
        count = int(dataset.metadata.loc[sha256, count_column])
        if count > min_active_voxels:
            kept.append((root, sha256))
    if max_instances is not None:
        kept = kept[:max_instances]
    dataset.instances = kept
    keep_sha = [sha256 for _, sha256 in kept]
    dataset.metadata = dataset.metadata.loc[keep_sha]
    dataset.loads = [int(dataset.metadata.loc[sha256, count_column]) for sha256 in keep_sha]
    return keep_sha


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().float()
    images = (images * 0.5 + 0.5).clamp(0.0, 1.0)
    nrow = max(1, int(math.ceil(math.sqrt(images.shape[0]))))
    tv_utils.save_image(images, str(path), nrow=nrow)


def save_snapshot_outputs(sample_dict: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in sample_dict.items():
        if payload.get("type") != "image":
            continue
        save_image_grid(payload["value"], out_dir / f"{key}.jpg")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config is not None else run_dir / "config.json"
    cfg = json.load(open(config_path, "r"))

    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    dataset_args["max_active_voxels"] = args.max_active_voxels
    if args.render_resolution is not None:
        trainer_args["render_resolution"] = args.render_resolution

    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / f"eval_{args.split}_over{args.min_active_voxels}_step{ckpt_step}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        data_dir = json.loads(args.data_dir)
    else:
        if args.root is None:
            raise ValueError("Either --data_dir or --root must be provided.")
        data_dir = build_data_dir(Path(args.root).resolve(), args.split, dataset_args)

    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    count_column = dataset_args["num_voxels_column"]
    keep_sha = restrict_to_overlimit(dataset, count_column, args.min_active_voxels, args.max_instances)
    if len(dataset) == 0:
        raise RuntimeError(f"No assets found with {count_column} > {args.min_active_voxels}")

    with open(output_dir / "instances.txt", "w") as f:
        f.write("\n".join(keep_sha) + "\n")

    model_dict = {
        name: getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda()
        for name, model_cfg in cfg["models"].items()
    }

    trainer = getattr(trainers, cfg["trainer"]["name"])(
        model_dict,
        dataset,
        **trainer_args,
        output_dir=str(output_dir),
        load_dir=str(run_dir),
        step=ckpt_step,
    )

    encoder = trainer.training_models["encoder"]
    decoder = trainer.training_models["decoder"]
    encoder.eval()
    decoder.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    l1_sum = edge_sum = vertex_sum = kl_sum = 0.0
    l1_count = edge_count = vertex_count = kl_count = 0
    rows = []
    offset = 0

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {args.split} assets over {args.min_active_voxels} voxels"):
            data = recursive_to_device(data, trainer.device)
            x = data["x"]

            z, mean, logvar = encoder(
                x,
                sample_posterior=not args.deterministic_posterior,
                return_raw=True,
            )
            y = decoder(z)

            diff = (x.feats - y.feats).abs()
            edge = diff[:, 0:3]
            vertex = diff[:, 3:6]
            kl_term = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)

            l1_sum += diff.sum().item()
            l1_count += diff.numel()
            edge_sum += edge.sum().item()
            edge_count += edge.numel()
            vertex_sum += vertex.sum().item()
            vertex_count += vertex.numel()
            kl_sum += kl_term.sum().item()
            kl_count += kl_term.numel()

            for batch_idx in range(x.shape[0]):
                mask = x.coords[:, 0] == batch_idx
                z_mask = z.coords[:, 0] == batch_idx
                sha256 = keep_sha[offset + batch_idx]
                rows.append({
                    "sha256": sha256,
                    "num_gaussian_distance_voxels": int(dataset.metadata.loc[sha256, count_column]),
                    "latent_tokens": int(z_mask.sum().item()),
                    "l1": float(diff[mask].mean().item()),
                    "edge_l1": float(edge[mask].mean().item()),
                    "vertex_l1": float(vertex[mask].mean().item()),
                    "kl": float(kl_term[z_mask].mean().item()),
                })
            offset += x.shape[0]

    lambda_kl = trainer_args["lambda_kl"]
    metrics = {
        "checkpoint_step": ckpt_step,
        "split": args.split,
        "num_instances": len(dataset),
        "min_active_voxels_exclusive": args.min_active_voxels,
        "max_active_voxels": args.max_active_voxels,
        "posterior_mode": "mean" if args.deterministic_posterior else "sampled",
        "l1": l1_sum / l1_count,
        "edge_l1": edge_sum / edge_count,
        "vertex_l1": vertex_sum / vertex_count,
        "kl": kl_sum / kl_count,
    }
    metrics["total"] = metrics["l1"] + lambda_kl * metrics["kl"]

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(rows).to_csv(output_dir / "per_instance_metrics.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {output_dir}")

    if args.num_samples > 0:
        sample_dict = trainer.run_snapshot(
            num_samples=args.num_samples,
            batch_size=args.snapshot_batch_size,
        )
        save_snapshot_outputs(sample_dict, output_dir / "samples")


if __name__ == "__main__":
    main()
