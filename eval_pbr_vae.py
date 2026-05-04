import os
import json
import math
import copy
import glob
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm import tqdm

from trellis2 import models, datasets, trainers
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PBR VAE on a held-out split.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config JSON to use. Defaults to <run_dir>/config.json.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latest",
        help="Checkpoint to evaluate: latest or an integer step.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Evaluation output directory. Defaults to <run_dir>/eval_<split>_step<step>.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional JSON data_dir override. If omitted, --root and the config dataset args are used.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Processed dataset root used to build a split-specific data_dir when --data_dir is omitted.",
    )
    parser.add_argument("--split", type=str, default="test", help="Split name under <root>/splits/.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size. Defaults to config trainer batch_size_per_gpu.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader worker count. Defaults to config trainer num_workers.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of visualization samples to save via trainer.run_snapshot().",
    )
    parser.add_argument(
        "--snapshot_batch_size",
        type=int,
        default=4,
        help="Batch size used by trainer.run_snapshot().",
    )
    parser.add_argument(
        "--deterministic_posterior",
        action="store_true",
        help="Use posterior mean for reconstruction instead of sampling, for more stable metrics.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        files = glob.glob(str(run_dir / "ckpts" / "misc_*.pt"))
        if not files:
            raise RuntimeError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files)
    if ckpt == "none":
        raise ValueError("ckpt=none is not valid for evaluation.")
    return int(ckpt)


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
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = json.load(open(config_path, "r"))
    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])

    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / f"eval_{args.split}_step{ckpt_step}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        data_dir = json.loads(args.data_dir)
    else:
        if args.root is None:
            raise ValueError("Either --data_dir or --root must be provided.")
        data_dir = build_data_dir(Path(args.root).resolve(), args.split, dataset_args)

    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)

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

    batch_size = args.batch_size or trainer_args["batch_size_per_gpu"]
    num_workers = args.num_workers if args.num_workers is not None else trainer_args.get("num_workers", 0)
    lambda_kl = trainer_args["lambda_kl"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    l1_sum = 0.0
    l1_count = 0
    edge_sum = 0.0
    edge_count = 0
    vertex_sum = 0.0
    vertex_count = 0
    kl_sum = 0.0
    kl_count = 0

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {args.split} split"):
            data = recursive_to_device(data, trainer.device)
            x = data["x"]

            z, mean, logvar = encoder(
                x,
                sample_posterior=not args.deterministic_posterior,
                return_raw=True,
            )
            y = decoder(z)

            diff = (x.feats - y.feats).abs()
            l1_sum += diff.sum().item()
            l1_count += diff.numel()

            edge = diff[:, 0:3]
            edge_sum += edge.sum().item()
            edge_count += edge.numel()

            vertex = diff[:, 3:6]
            vertex_sum += vertex.sum().item()
            vertex_count += vertex.numel()

            kl_term = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
            kl_sum += kl_term.sum().item()
            kl_count += kl_term.numel()

    metrics = {
        "checkpoint_step": ckpt_step,
        "split": args.split,
        "num_instances": len(dataset),
        "posterior_mode": "mean" if args.deterministic_posterior else "sampled",
        "l1": l1_sum / l1_count,
        "edge_l1": edge_sum / edge_count,
        "vertex_l1": vertex_sum / vertex_count,
        "kl": kl_sum / kl_count,
    }
    metrics["total"] = metrics["l1"] + lambda_kl * metrics["kl"]

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    if args.num_samples > 0:
        sample_dict = trainer.run_snapshot(
            num_samples=args.num_samples,
            batch_size=args.snapshot_batch_size,
        )
        save_snapshot_outputs(sample_dict, output_dir / "samples")
        print(f"Saved visualization samples to {output_dir / 'samples'}")


if __name__ == "__main__":
    main()
