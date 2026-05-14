import os
import json
import math
import copy
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Gaussian-distance flow model on a held-out split.")
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
        help="Optional JSON data_dir override. If omitted, --root and latent names are used.",
    )
    parser.add_argument("--root", type=str, default=None, help="Processed dataset root used when --data_dir is omitted.")
    parser.add_argument("--split", type=str, default="test", help="Split name under <root>/splits/.")
    parser.add_argument(
        "--gaussian_distance_latent_name",
        type=str,
        default="gaussian_distance_vae_step0350000_256",
        help="Latent directory name under split gaussian_distance_latents/.",
    )
    parser.add_argument(
        "--michelangelo_latent_name",
        type=str,
        default="shapevae256_pretrained",
        help="Latent directory name under split michelangelo_latents/.",
    )
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
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on validation batches for a quick smoke test.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of visualization samples to save via trainer.snapshot().",
    )
    parser.add_argument(
        "--snapshot_batch_size",
        type=int,
        default=4,
        help="Batch size used by trainer.snapshot().",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Override flow snapshot render resolution. Defaults to the dataset config.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=12,
        help="Number of Euler sampling steps used for visualization snapshots.",
    )
    parser.add_argument(
        "--guidance_strength",
        type=float,
        default=3.0,
        help="Classifier-free guidance strength used for visualization snapshots.",
    )
    parser.add_argument(
        "--ema_rate",
        type=str,
        default=None,
        help="Optional EMA rate to evaluate, e.g. 0.9999. Defaults to raw denoiser checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for validation timesteps/noise and snapshots.")
    return parser.parse_args()


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


def build_data_dir(root: Path, split: str, gaussian_distance_latent_name: str, michelangelo_latent_name: str) -> dict:
    split_root = root / "splits" / split
    return {
        split: {
            "metadata": str(split_root),
            "gaussian_distance_latent": str(
                split_root / "gaussian_distance_latents" / gaussian_distance_latent_name
            ),
            "michelangelo_latent": str(
                split_root / "michelangelo_latents" / michelangelo_latent_name
            ),
        }
    }


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

    metrics = {
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
    return metrics


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
            args.gaussian_distance_latent_name,
            args.michelangelo_latent_name,
        )
        train_norm_path = (
            root
            / "splits"
            / "train"
            / "gaussian_distance_latents"
            / args.gaussian_distance_latent_name
            / "normalization.json"
        )
        if train_norm_path.exists():
            dataset_args["gaussian_distance_slat_normalization_path"] = str(train_norm_path)

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
        )
        print(f"Saved visualization samples to {output_dir / 'samples' / suffix}")


if __name__ == "__main__":
    main()
