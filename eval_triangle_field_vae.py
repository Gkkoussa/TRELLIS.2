import argparse
import copy
import glob
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm import tqdm

from trellis2 import datasets, models, trainers
from trellis2.utils.data_utils import recursive_to_device
from eval_metadata_filters import add_eval_metadata_filter_args, attach_eval_metadata_filter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a triangle-field VAE.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--snapshot_batch_size", type=int, default=1)
    parser.add_argument("--render_resolution", type=int, default=None)
    parser.add_argument("--dataset_resolution", type=int, default=None)
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--weight_eps", type=float, default=1e-8)
    parser.add_argument("--weight_clamp_max", type=float, default=100.0)
    parser.add_argument("--log_area_min", type=float, default=-12.0)
    parser.add_argument("--log_area_max", type=float, default=0.0)
    parser.add_argument("--num_area_bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        patterns = [
            str(run_dir / "ckpts" / "misc_*.pt"),
            str(run_dir / "ckpts" / "encoder_*.pt"),
        ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files)
    return int(ckpt)


def build_data_dir(root: Path, split: str, dataset_args: dict, args=None) -> dict:
    voxel_root_key = dataset_args["voxel_root_key"]
    voxel_dirname = dataset_args["voxel_dirname"]
    resolution = dataset_args["resolution"]

    split_root = root / "splits" / split
    filtered_base = root / "splits" / f"{split}_triangle_field_{resolution}"
    base_root = filtered_base if (filtered_base / "metadata.csv").exists() else split_root
    split_voxel_root = split_root / f"{voxel_dirname}_{resolution}"
    canonical_voxel_root = root / f"{voxel_dirname}_{resolution}"
    voxel_root = split_voxel_root if (split_voxel_root / "metadata.csv").exists() else canonical_voxel_root

    data_dir = {
        split: {
            "base": str(base_root),
            voxel_root_key: str(voxel_root),
        }
    }
    return attach_eval_metadata_filter(data_dir, root, split, args)


def restrict_dataset_instances(dataset, instances_path: str) -> None:
    with open(instances_path, "r") as f:
        keep = {line.strip() for line in f if line.strip()}
    before = len(dataset.instances)
    dataset.instances = [
        (root, sha256)
        for root, sha256 in dataset.instances
        if str(sha256) in keep
    ]
    if len(dataset.metadata) > 0:
        dataset.metadata = dataset.metadata[dataset.metadata.index.astype(str).isin(keep)]
    if hasattr(dataset, "loads"):
        dataset.loads = [
            dataset.metadata.loc[sha256, dataset.num_voxels_column]
            if getattr(dataset, "num_voxels_column", None) in dataset.metadata.columns else 1
            for _, sha256 in dataset.instances
        ]
    for stats in getattr(dataset, "_stats", {}).values():
        stats["Restricted to instances"] = len(dataset.instances)
        stats["Instances removed"] = before - len(dataset.instances)


def limit_dataset_instances(dataset, max_eval_samples: int) -> None:
    if max_eval_samples is None:
        return
    if max_eval_samples <= 0:
        raise ValueError("--max_eval_samples must be positive when provided.")
    before = len(dataset.instances)
    dataset.instances = dataset.instances[:max_eval_samples]
    keep = {str(sha256) for _, sha256 in dataset.instances}
    if len(dataset.metadata) > 0:
        dataset.metadata = dataset.metadata[dataset.metadata.index.astype(str).isin(keep)]
    if hasattr(dataset, "loads"):
        dataset.loads = dataset.loads[:len(dataset.instances)]
    for stats in getattr(dataset, "_stats", {}).values():
        stats["Max eval samples"] = max_eval_samples
        stats["Eval samples used"] = len(dataset.instances)
        stats["Eval samples skipped"] = max(0, before - len(dataset.instances))


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().float().clamp(0.0, 1.0)
    nrow = max(1, int(math.ceil(math.sqrt(images.shape[0]))))
    tv_utils.save_image(images, str(path), nrow=nrow)


def save_snapshot_outputs(sample_dict: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in sample_dict.items():
        if payload.get("type") != "image":
            continue
        save_image_grid(payload["value"], out_dir / f"{key}.jpg")


def triangle_area_from_features(x_feats: torch.Tensor, eps: float) -> torch.Tensor:
    with torch.autocast(device_type="cuda", enabled=False):
        offset0 = x_feats[:, 2:5].float()
        offset1 = x_feats[:, 5:8].float()
        offset2 = x_feats[:, 8:11].float()
        area = 0.5 * torch.linalg.norm(
            torch.cross(offset1 - offset0, offset2 - offset0, dim=-1),
            dim=-1,
        )
    return area.clamp_min(eps)


def maybe_inverse_distance_transform(values: torch.Tensor, distance_transform: str) -> torch.Tensor:
    if distance_transform == "minus_one_one":
        return values * 0.5 + 0.5
    if distance_transform == "none":
        return values
    raise ValueError(f"Unsupported distance_transform: {distance_transform}")


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
    if args.dataset_resolution is not None:
        dataset_args["resolution"] = args.dataset_resolution
        if cfg["dataset"]["name"] == "MultiResolutionSparseVoxelTriangleFieldDataset":
            resolutions = [int(r) for r in dataset_args.get("resolutions", [])]
            if int(args.dataset_resolution) not in resolutions:
                raise ValueError(
                    f"--dataset_resolution {args.dataset_resolution} is not in config resolutions {resolutions}"
                )
            dataset_args.pop("resolutions", None)
            dataset_args.pop("instances_path", None)
            cfg["dataset"]["name"] = "SparseVoxelTriangleFieldDataset"
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    if args.render_resolution is not None:
        trainer_args["render_resolution"] = args.render_resolution

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
        data_dir = build_data_dir(Path(args.root).resolve(), args.split, dataset_args, args=args)

    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    if args.instances is not None:
        restrict_dataset_instances(dataset, args.instances)
    limit_dataset_instances(dataset, args.max_eval_samples)
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
    distance_transform = dataset_args.get("distance_transform", "none")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    channel_names = ["d_tri", "d_vert"]
    l1_sum = torch.zeros(2, dtype=torch.float64)
    l2_sum = torch.zeros(2, dtype=torch.float64)
    count = 0

    weighted_l1_sum = torch.zeros(2, dtype=torch.float64)
    weighted_l2_sum = torch.zeros(2, dtype=torch.float64)
    weight_sum = 0.0

    area_sum = 0.0
    area_count = 0
    min_area = float("inf")
    max_area = 0.0

    bin_edges = torch.linspace(args.log_area_min, args.log_area_max, args.num_area_bins + 1)
    bin_l1_sum = torch.zeros(args.num_area_bins, 2, dtype=torch.float64)
    bin_count = torch.zeros(args.num_area_bins, dtype=torch.float64)

    kl_sum = 0.0
    kl_count = 0

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {args.split} split"):
            data = recursive_to_device(data, trainer.device)
            x = data["x"]
            target = data["target"]

            z, mean, logvar = encoder(
                x,
                sample_posterior=args.sample_posterior,
                return_raw=True,
            )
            y = decoder(z)
            if y.feats.shape != target.feats.shape:
                raise ValueError(
                    f"Decoder output shape must match target shape, got "
                    f"{tuple(y.feats.shape)} vs {tuple(target.feats.shape)}."
                )

            pred_values = maybe_inverse_distance_transform(y.feats.float(), distance_transform)
            target_values = maybe_inverse_distance_transform(target.feats.float(), distance_transform)
            err = pred_values - target_values
            abs_err = err.abs()
            sq_err = err.square()

            l1_sum += abs_err.sum(dim=0).detach().cpu().double()
            l2_sum += sq_err.sum(dim=0).detach().cpu().double()
            count += abs_err.shape[0]

            area = triangle_area_from_features(x.feats, args.weight_eps)
            weight = area.reciprocal()
            if args.weight_clamp_max is not None:
                weight = weight.clamp_max(args.weight_clamp_max)

            weighted_l1_sum += (abs_err * weight[:, None]).sum(dim=0).detach().cpu().double()
            weighted_l2_sum += (sq_err * weight[:, None]).sum(dim=0).detach().cpu().double()
            weight_sum += float(weight.sum().detach().cpu())

            area_cpu = area.detach().cpu()
            area_sum += float(area_cpu.double().sum())
            area_count += int(area_cpu.numel())
            min_area = min(min_area, float(area_cpu.min()))
            max_area = max(max_area, float(area_cpu.max()))

            log_area = torch.log10(area_cpu.clamp_min(10 ** args.log_area_min))
            bin_idx = torch.bucketize(log_area, bin_edges, right=False) - 1
            bin_idx = bin_idx.clamp(0, args.num_area_bins - 1)
            abs_err_cpu = abs_err.detach().cpu().double()
            for bin_i in range(args.num_area_bins):
                mask = bin_idx == bin_i
                n = int(mask.sum())
                if n == 0:
                    continue
                bin_l1_sum[bin_i] += abs_err_cpu[mask].sum(dim=0)
                bin_count[bin_i] += n

            kl_term = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
            kl_sum += float(kl_term.sum().detach().cpu())
            kl_count += int(kl_term.numel())

    l1 = l1_sum / max(count, 1)
    rmse = torch.sqrt(l2_sum / max(count, 1))
    weighted_l1 = weighted_l1_sum / max(weight_sum, 1e-12)
    weighted_rmse = torch.sqrt(weighted_l2_sum / max(weight_sum, 1e-12))

    area_bins = []
    for bin_i in range(args.num_area_bins):
        n = int(bin_count[bin_i].item())
        area_bins.append({
            "log10_area_min": float(bin_edges[bin_i].item()),
            "log10_area_max": float(bin_edges[bin_i + 1].item()),
            "voxel_count": n,
            "l1": {
                channel_names[c]: (
                    float((bin_l1_sum[bin_i, c] / bin_count[bin_i]).item())
                    if n > 0 else None
                )
                for c in range(2)
            },
            "l1_mean": (
                float((bin_l1_sum[bin_i].sum() / (bin_count[bin_i] * 2)).item())
                if n > 0 else None
            ),
        })

    metrics = {
        "checkpoint_step": ckpt_step,
        "split": args.split,
        "num_instances": len(dataset),
        "num_voxels": count,
        "posterior_mode": "sampled" if args.sample_posterior else "mean",
        "distance_units": "[0, 1]",
        "l1": {channel_names[i]: float(l1[i].item()) for i in range(2)},
        "l1_mean": float(l1.mean().item()),
        "rmse": {channel_names[i]: float(rmse[i].item()) for i in range(2)},
        "rmse_mean": float(rmse.mean().item()),
        "inverse_area_weighted_l1": {channel_names[i]: float(weighted_l1[i].item()) for i in range(2)},
        "inverse_area_weighted_l1_mean": float(weighted_l1.mean().item()),
        "inverse_area_weighted_rmse": {channel_names[i]: float(weighted_rmse[i].item()) for i in range(2)},
        "inverse_area_weighted_rmse_mean": float(weighted_rmse.mean().item()),
        "triangle_area": {
            "min": min_area,
            "max": max_area,
            "mean": area_sum / max(area_count, 1),
            "weight_eps": args.weight_eps,
            "weight_clamp_max": args.weight_clamp_max,
        },
        "area_bins": area_bins,
        "kl": kl_sum / max(kl_count, 1),
        "data_dir": data_dir,
    }

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


if __name__ == "__main__":
    main()
