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
    parser = argparse.ArgumentParser(description="Evaluate a trained occupancy shape VAE on a held-out split.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON to use. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint to evaluate: latest or an integer step.")
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
        "--metadata_filter_csv",
        type=str,
        default=None,
        help="Optional metadata CSV filter. Multiple CSVs can be comma-separated and are applied by the dataset loader.",
    )
    parser.add_argument("--root", type=str, default=None, help="Processed dataset root used when --data_dir is omitted.")
    parser.add_argument("--split", type=str, default="test", help="Split name under <root>/splits/.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of visualization samples to save.")
    parser.add_argument("--snapshot_batch_size", type=int, default=4, help="Batch size used by trainer.run_snapshot().")
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


def build_data_dir(root: Path, split: str, dataset_args: dict, metadata_filter_csv: str | None = None) -> dict:
    voxel_root_key = dataset_args["voxel_root_key"]
    voxel_dirname = dataset_args["voxel_dirname"]
    resolution = dataset_args["resolution"]
    split_root = root / "splits" / split
    voxel_root = root / f"{voxel_dirname}_{resolution}"
    filters = [str(split_root / "metadata.csv")]
    if metadata_filter_csv is not None and str(metadata_filter_csv).strip() != "":
        filters.append(str(Path(metadata_filter_csv).resolve()))
    return {
        split: {
            "base": str(split_root),
            voxel_root_key: str(voxel_root),
            "_metadata_filter_csv": ",".join(filters),
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


def coords_to_set(coords: torch.Tensor) -> set[tuple[int, int, int]]:
    coords = coords.detach().cpu().int()
    return set(map(tuple, coords[:, 1:].tolist()))


def support_metrics(gt, pred) -> dict:
    gt_set = coords_to_set(gt.coords)
    pred_set = coords_to_set(pred.coords)
    intersection = len(gt_set & pred_set)
    union = len(gt_set | pred_set)
    gt_count = len(gt_set)
    pred_count = len(pred_set)
    false_negative = gt_count - intersection
    false_positive = pred_count - intersection
    return {
        "gt_count": gt_count,
        "pred_count": pred_count,
        "intersection": intersection,
        "union": union,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "iou": intersection / union if union else 1.0,
        "recall": intersection / gt_count if gt_count else 1.0,
        "precision": intersection / pred_count if pred_count else 1.0,
        "exact_match": gt_set == pred_set,
        "all_gt_captured": false_negative == 0,
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
        data_dir = build_data_dir(Path(args.root).resolve(), args.split, dataset_args, args.metadata_filter_csv)

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

    totals = {
        "instances": 0,
        "gt_count": 0,
        "pred_count": 0,
        "intersection": 0,
        "union": 0,
        "false_negative": 0,
        "false_positive": 0,
        "exact_match": 0,
        "all_gt_captured": 0,
        "iou_sum": 0.0,
        "recall_sum": 0.0,
        "precision_sum": 0.0,
        "kl_sum": 0.0,
        "kl_count": 0,
    }

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc=f"Evaluating {args.split} split")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            data = recursive_to_device(data, trainer.device)
            x = data["x"]

            z, mean, logvar = encoder(
                x,
                sample_posterior=not args.deterministic_posterior,
                return_raw=True,
            )
            z.clear_spatial_cache()
            y = decoder(z)

            kl_term = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
            totals["kl_sum"] += kl_term.sum().item()
            totals["kl_count"] += kl_term.numel()

            for i in range(x.shape[0]):
                m = support_metrics(x[i], y[i])
                totals["instances"] += 1
                for key in ["gt_count", "pred_count", "intersection", "union", "false_negative", "false_positive"]:
                    totals[key] += m[key]
                totals["exact_match"] += int(m["exact_match"])
                totals["all_gt_captured"] += int(m["all_gt_captured"])
                totals["iou_sum"] += m["iou"]
                totals["recall_sum"] += m["recall"]
                totals["precision_sum"] += m["precision"]

    n = totals["instances"]
    if n == 0:
        raise RuntimeError("No instances were evaluated.")

    metrics = {
        "checkpoint_step": ckpt_step,
        "split": args.split,
        "num_instances": n,
        "dataset_size": len(dataset),
        "max_batches": args.max_batches,
        "posterior_mode": "mean" if args.deterministic_posterior else "sampled",
        "metadata_filter_csv": str(Path(args.metadata_filter_csv).resolve()) if args.metadata_filter_csv else None,
        "mean_iou": totals["iou_sum"] / n,
        "mean_recall": totals["recall_sum"] / n,
        "mean_precision": totals["precision_sum"] / n,
        "exact_support_match_rate": totals["exact_match"] / n,
        "all_gt_locations_captured_rate": totals["all_gt_captured"] / n,
        "total_gt_locations": totals["gt_count"],
        "total_pred_locations": totals["pred_count"],
        "total_intersection_locations": totals["intersection"],
        "total_false_negative_locations": totals["false_negative"],
        "total_false_positive_locations": totals["false_positive"],
        "micro_iou": totals["intersection"] / totals["union"] if totals["union"] else 1.0,
        "micro_recall": totals["intersection"] / totals["gt_count"] if totals["gt_count"] else 1.0,
        "micro_precision": totals["intersection"] / totals["pred_count"] if totals["pred_count"] else 1.0,
        "kl": totals["kl_sum"] / totals["kl_count"],
    }
    metrics["total"] = (1.0 - metrics["mean_iou"]) + lambda_kl * metrics["kl"]

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
