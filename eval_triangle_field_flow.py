import os
import json
import glob
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from trellis2 import models, datasets, trainers
from trellis2.utils.data_utils import recursive_to_device
from eval_metadata_filters import add_eval_metadata_filter_args, attach_eval_metadata_filter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained triangle-field flow model on a held-out split.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON to use. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint to evaluate: latest or an integer step.")
    parser.add_argument("--output_dir", type=str, default=None, help="Evaluation output directory.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional JSON data_dir override.")
    parser.add_argument("--root", type=str, default=None, help="Processed dataset root used when --data_dir is omitted.")
    parser.add_argument("--split", type=str, default="test", help="Split name under <root>/splits/.")
    parser.add_argument("--instances", type=str, default=None, help="Optional ordered instance list to evaluate/visualize.")
    parser.add_argument(
        "--triangle_field_latent_name",
        type=str,
        default="triangle_field_vae_51483691_step0050000_256",
        help="Latent directory name under split triangle_field_latents/.",
    )
    parser.add_argument(
        "--michelangelo_latent_name",
        type=str,
        default="shapevae256_pretrained",
        help="Latent directory name under split michelangelo_latents/.",
    )
    parser.add_argument(
        "--shape_latent_name",
        type=str,
        default="occupancy_shape_vae_step0110000_256",
        help="Latent directory name under split shape_latents/.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap on validation batches.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of visualization samples.")
    parser.add_argument("--snapshot_batch_size", type=int, default=4, help="Batch size used by trainer.snapshot().")
    parser.add_argument("--dataset_resolution", type=int, default=None, help="Override dataset/decode resolution.")
    parser.add_argument("--render_resolution", type=int, default=None, help="Override flow snapshot render resolution.")
    parser.add_argument("--sampling_steps", type=int, default=12, help="Number of Euler sampling steps.")
    parser.add_argument("--guidance_strength", type=float, default=1.0, help="Classifier-free guidance strength.")
    parser.add_argument("--ema_rate", type=str, default=None, help="Optional EMA rate to evaluate, e.g. 0.9999.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def restrict_dataset_to_instances(dataset, instances_path: str, limit: int | None = None) -> list[str]:
    with open(instances_path, "r") as f:
        ordered = [line.strip() for line in f if line.strip()]
    if limit is not None:
        ordered = ordered[:limit]
    keep = set(ordered)
    by_sha = {str(sha256): (root, sha256) for root, sha256 in dataset.instances}
    missing = [sha256 for sha256 in ordered if sha256 not in by_sha]
    if missing:
        raise ValueError(f"{len(missing)} requested instances are missing from dataset; first missing: {missing[0]}")
    dataset.instances = [by_sha[sha256] for sha256 in ordered]
    if len(dataset.metadata) > 0:
        dataset.metadata = dataset.metadata[dataset.metadata.index.astype(str).isin(keep)]
    if hasattr(dataset, "loads"):
        dataset.loads = [
            dataset.metadata.loc[sha256, dataset.num_voxels_column]
            if getattr(dataset, "num_voxels_column", None) in dataset.metadata.columns else 1
            for _, sha256 in dataset.instances
        ]
    for stats in getattr(dataset, "_stats", {}).values():
        stats["Restricted to explicit instances"] = len(dataset.instances)
    return ordered


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


def build_data_dir(
    root: Path,
    split: str,
    triangle_field_latent_name: str,
    michelangelo_latent_name: str,
    shape_latent_name: str,
    args=None,
) -> dict:
    split_root = root / "splits" / split
    triangle_field_latent = split_root / "triangle_field_latents" / triangle_field_latent_name
    michelangelo_latent = split_root / "michelangelo_latents" / michelangelo_latent_name
    shape_latent = split_root / "shape_latents" / shape_latent_name
    if not triangle_field_latent.exists():
        triangle_field_latent = root / "triangle_field_latents" / triangle_field_latent_name
    if not michelangelo_latent.exists():
        michelangelo_latent = root / "michelangelo_latents" / michelangelo_latent_name
    if not shape_latent.exists():
        shape_latent = root / "shape_latents" / shape_latent_name
    data_dir = {
        split: {
            "metadata": str(split_root),
            "triangle_field_latent": str(triangle_field_latent),
            "michelangelo_latent": str(michelangelo_latent),
            "shape_latent": str(shape_latent),
        }
    }
    return attach_eval_metadata_filter(data_dir, root, split, args)


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

    if trainer.mix_precision_mode == "amp":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

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

            with amp_context():
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


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config is not None else run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = json.load(open(config_path, "r"))
    dataset_args = json.loads(json.dumps(cfg["dataset"]["args"]))
    trainer_args = json.loads(json.dumps(cfg["trainer"]["args"]))
    if args.dataset_resolution is not None:
        dataset_args["resolution"] = args.dataset_resolution
    if args.render_resolution is not None:
        dataset_args["snapshot_render_resolution"] = args.render_resolution

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
        root = Path(args.root).resolve()
        data_dir = build_data_dir(
            root,
            args.split,
            args.triangle_field_latent_name,
            args.michelangelo_latent_name,
            args.shape_latent_name,
            args=args,
        )
        train_norm_path = (
            root
            / "splits"
            / "train"
            / "triangle_field_latents"
            / args.triangle_field_latent_name
            / "normalization.json"
        )
        if train_norm_path.exists():
            dataset_args["triangle_field_slat_normalization_path"] = str(train_norm_path)
        shape_train_norm_path = (
            root
            / "splits"
            / "train"
            / "shape_latents"
            / args.shape_latent_name
            / "normalization.json"
        )
        if shape_train_norm_path.exists():
            dataset_args["shape_slat_normalization_path"] = str(shape_train_norm_path)

    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    selected_instances = None
    if args.instances is not None:
        selected_instances = restrict_dataset_to_instances(dataset, args.instances, args.num_samples if args.num_samples > 0 else None)

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
    trainer.p_uncond = 0.0
    ckpt_path = load_denoiser_checkpoint(
        trainer.models["denoiser"],
        run_dir,
        ckpt_step,
        args.ema_rate,
        trainer.device,
    )
    trainer.models["denoiser"].eval()

    batch_size = args.batch_size or trainer_args["batch_size_per_gpu"]
    num_workers = args.num_workers if args.num_workers is not None else trainer_args.get("num_workers", 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    metrics = evaluate_flow_mse(trainer, loader, max_batches=args.max_batches)
    metrics.update({
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "dataset_size": len(dataset),
        "max_batches": args.max_batches,
        "instances": selected_instances,
    })

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    if args.num_samples > 0:
        suffix = f"{args.split}_step{ckpt_step:07d}"
        trainer.snapshot(
            suffix=suffix,
            num_samples=args.num_samples,
            batch_size=args.snapshot_batch_size,
            steps=args.sampling_steps,
            guidance_strength=args.guidance_strength,
            shuffle=False,
        )
        print(f"Saved visualization samples to {output_dir / 'samples' / suffix}")


if __name__ == "__main__":
    main()
