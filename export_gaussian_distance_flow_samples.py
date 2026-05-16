import argparse
import copy
import glob
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from trellis2 import datasets, models, trainers
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export individual Gaussian-distance flow samples for qualitative and latent-space debugging."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Flow run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint step or latest.")
    parser.add_argument("--ema_rate", type=str, default=None, help="Optional EMA rate, e.g. 0.9999.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for exported samples.")
    parser.add_argument("--root", type=str, required=True, help="Processed dataset root.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split under <root>/splits/.")
    parser.add_argument("--gaussian_distance_latent_name", type=str, required=True)
    parser.add_argument("--michelangelo_latent_name", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated dataset indices to export.")
    parser.add_argument("--sha256s", type=str, default=None, help="Comma-separated sha256 ids to export.")
    parser.add_argument("--random", action="store_true", help="Randomly choose samples instead of sequential indices.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sampling_steps", type=int, default=12)
    parser.add_argument("--guidance_strength", type=float, default=3.0)
    parser.add_argument("--render_resolution", type=int, default=None)
    parser.add_argument("--save_decoded_npz", action="store_true", help="Also decode and save dense-ish voxel sparse outputs.")
    return parser.parse_args()


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        misc_files = glob.glob(str(run_dir / "ckpts" / "misc_*.pt"))
        if misc_files:
            return max(int(os.path.basename(path).split("step")[-1].split(".")[0]) for path in misc_files)
        denoiser_files = glob.glob(str(run_dir / "ckpts" / "denoiser_step*.pt"))
        if not denoiser_files:
            raise FileNotFoundError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(path).split("step")[-1].split(".")[0]) for path in denoiser_files)
    return int(ckpt)


def build_data_dir(root: Path, split: str, gaussian_distance_latent_name: str, michelangelo_latent_name: str) -> dict:
    split_root = root / "splits" / split
    return {
        split: {
            "metadata": str(split_root),
            "gaussian_distance_latent": str(split_root / "gaussian_distance_latents" / gaussian_distance_latent_name),
            "michelangelo_latent": str(split_root / "michelangelo_latents" / michelangelo_latent_name),
        }
    }


def load_denoiser_checkpoint(model, run_dir: Path, step: int, ema_rate: str | None, device: torch.device) -> str:
    if ema_rate is None:
        ckpt_path = run_dir / "ckpts" / f"denoiser_step{step:07d}.pt"
    else:
        ckpt_path = run_dir / "ckpts" / f"denoiser_ema{ema_rate}_step{step:07d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Denoiser checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return str(ckpt_path)


def select_indices(dataset, args) -> list[int]:
    if args.indices:
        return [int(index) for index in args.indices.split(",") if index.strip()]

    if args.sha256s:
        wanted = {sha.strip() for sha in args.sha256s.split(",") if sha.strip()}
        selected = [idx for idx, (_, sha) in enumerate(dataset.instances) if sha in wanted]
        missing = sorted(wanted - {dataset.instances[idx][1] for idx in selected})
        if missing:
            raise ValueError(f"Requested sha256 ids not present after dataset filtering: {missing}")
        return selected

    if args.random:
        rng = np.random.default_rng(args.seed)
        count = min(args.num_samples, len(dataset))
        return rng.choice(len(dataset), size=count, replace=False).tolist()

    stop = min(args.start_index + args.num_samples, len(dataset))
    return list(range(args.start_index, stop))


def sparse_to_npz_dict(tensor, item_index=0, normalization=None, denormalize=False):
    item = tensor[item_index].detach().cpu()
    feats = item.feats.float()
    if denormalize:
        if normalization is None:
            raise ValueError("normalization is required for denormalize=True")
        mean = torch.tensor(normalization["mean"], dtype=feats.dtype).reshape(1, -1)
        std = torch.tensor(normalization["std"], dtype=feats.dtype).reshape(1, -1)
        feats = feats * std + mean
    return {
        "coords": item.coords[:, 1:].numpy().astype(np.int32),
        "feats": feats.numpy().astype(np.float32),
    }


def sparse_stats(tensor, item_index=0, normalization=None, denormalize=False):
    data = sparse_to_npz_dict(tensor, item_index, normalization=normalization, denormalize=denormalize)
    feats = data["feats"]
    coords = data["coords"]
    return {
        "num_tokens": int(feats.shape[0]),
        "feat_mean": feats.mean(axis=0).tolist() if feats.size else [],
        "feat_std": feats.std(axis=0).tolist() if feats.size else [],
        "feat_min": feats.min(axis=0).tolist() if feats.size else [],
        "feat_max": feats.max(axis=0).tolist() if feats.size else [],
        "coord_min": coords.min(axis=0).tolist() if coords.size else [],
        "coord_max": coords.max(axis=0).tolist() if coords.size else [],
        "has_nan": bool(np.isnan(feats).any()),
        "has_inf": bool(np.isinf(feats).any()),
    }


def latent_pair_metrics(gt, pred, item_index=0, normalization=None):
    gt_item = gt[item_index].detach().cpu()
    pred_item = pred[item_index].detach().cpu()
    gt_feats = gt_item.feats.float()
    pred_feats = pred_item.feats.float()
    metrics = {
        "normalized_l1": torch.mean(torch.abs(pred_feats - gt_feats)).item(),
        "normalized_l2": torch.mean((pred_feats - gt_feats) ** 2).item(),
        "coord_exact_match": bool(torch.equal(gt_item.coords[:, 1:], pred_item.coords[:, 1:])),
        "gt_tokens": int(gt_feats.shape[0]),
        "pred_tokens": int(pred_feats.shape[0]),
    }
    if normalization is not None:
        mean = torch.tensor(normalization["mean"], dtype=gt_feats.dtype).reshape(1, -1)
        std = torch.tensor(normalization["std"], dtype=gt_feats.dtype).reshape(1, -1)
        gt_denorm = gt_feats * std + mean
        pred_denorm = pred_feats * std + mean
        metrics["denormalized_l1"] = torch.mean(torch.abs(pred_denorm - gt_denorm)).item()
        metrics["denormalized_l2"] = torch.mean((pred_denorm - gt_denorm) ** 2).item()
    return metrics


def save_visualizations(dataset, sample, out_dir: Path, prefix: str):
    vis = dataset.visualize_sample(sample)
    for key, value in vis.items():
        image = value[0].detach().cpu().clamp(0, 1)
        save_image(image, out_dir / f"{prefix}_{key}.jpg", normalize=False)


def save_decoded_voxels(dataset, sample, out_dir: Path, prefix: str):
    voxels = dataset.decode_latent(sample["x_0"].cuda(), cache_paths=sample["gaussian_distance_slat_cache_path"])
    for idx, voxel in enumerate(voxels):
        item = voxel.detach().cpu()
        np.savez_compressed(
            out_dir / f"{prefix}_decoded_voxel_{idx:02d}.npz",
            coords=item.coords[:, 1:].numpy().astype(np.int32),
            feats=item.feats.float().numpy().astype(np.float32),
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config is not None else run_dir / "config.json"
    cfg = json.load(open(config_path, "r"))

    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    if args.render_resolution is not None:
        dataset_args["snapshot_render_resolution"] = args.render_resolution

    root = Path(args.root).resolve()
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

    data_dir = build_data_dir(
        root,
        args.split,
        args.gaussian_distance_latent_name,
        args.michelangelo_latent_name,
    )
    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)

    model_dict = {
        name: getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda()
        for name, model_cfg in cfg["models"].items()
    }
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    trainer = getattr(trainers, cfg["trainer"]["name"])(
        model_dict,
        dataset,
        **trainer_args,
        output_dir=str(run_dir),
        load_dir=None,
        step=None,
    )
    trainer.p_uncond = 0.0
    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    ckpt_path = load_denoiser_checkpoint(
        trainer.models["denoiser"],
        run_dir,
        ckpt_step,
        args.ema_rate,
        trainer.device,
    )
    trainer.models["denoiser"].eval()

    if trainer.mix_precision_mode == "amp":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / f"sample_debug_{args.split}_step{ckpt_step:07d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = select_indices(dataset, args)
    normalization = getattr(dataset, "gaussian_distance_slat_normalization", None)
    sampler = trainer.get_sampler()

    manifest = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "sampling_steps": args.sampling_steps,
        "guidance_strength": args.guidance_strength,
        "samples": [],
    }

    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        raw_items = [dataset[index] for index in batch_indices]
        batch = dataset.collate_fn(raw_items)
        batch = recursive_to_device(batch, "cuda")

        noise = batch["x_0"].replace(torch.randn_like(batch["x_0"].feats))
        cond_args = trainer.get_inference_cond(
            batch["cond"],
            neg_cond=batch["neg_cond"],
            gaussian_distance_slat_cache_path=batch["gaussian_distance_slat_cache_path"],
        )
        with torch.no_grad():
            with amp_context():
                sample = sampler.sample(
                    trainer.models["denoiser"],
                    noise=noise,
                    **cond_args,
                    steps=args.sampling_steps,
                    guidance_strength=args.guidance_strength,
                    verbose=False,
                ).samples

        for local_idx, dataset_idx in enumerate(batch_indices):
            _, sha256 = dataset.instances[dataset_idx]
            sample_dir = output_dir / f"{dataset_idx:06d}_{sha256}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            gt_one = batch["x_0"][local_idx]
            pred_one = sample[local_idx]
            cache_path = batch["gaussian_distance_slat_cache_path"][local_idx]

            gt_norm = sparse_to_npz_dict(gt_one, normalization=normalization, denormalize=False)
            pred_norm = sparse_to_npz_dict(pred_one, normalization=normalization, denormalize=False)
            np.savez_compressed(sample_dir / "gt_latent_normalized.npz", **gt_norm)
            np.savez_compressed(sample_dir / "pred_latent_normalized.npz", **pred_norm)
            if normalization is not None:
                np.savez_compressed(
                    sample_dir / "gt_latent_denormalized.npz",
                    **sparse_to_npz_dict(gt_one, normalization=normalization, denormalize=True),
                )
                np.savez_compressed(
                    sample_dir / "pred_latent_denormalized.npz",
                    **sparse_to_npz_dict(pred_one, normalization=normalization, denormalize=True),
                )

            np.savez_compressed(
                sample_dir / "michelangelo_cond.npz",
                feats=batch["cond"][local_idx].detach().cpu().float().numpy().astype(np.float32),
            )

            gt_vis_sample = {
                "x_0": gt_one,
                "gaussian_distance_slat_cache_path": [cache_path],
            }
            pred_vis_sample = {
                "x_0": pred_one,
                "gaussian_distance_slat_cache_path": [cache_path],
            }
            save_visualizations(dataset, gt_vis_sample, sample_dir, "gt")
            save_visualizations(dataset, pred_vis_sample, sample_dir, "pred")

            if args.save_decoded_npz:
                save_decoded_voxels(dataset, gt_vis_sample, sample_dir, "gt")
                save_decoded_voxels(dataset, pred_vis_sample, sample_dir, "pred")

            metrics = {
                "dataset_index": dataset_idx,
                "sha256": sha256,
                "cache_path": cache_path,
                "pair_metrics": latent_pair_metrics(gt_one, pred_one, normalization=normalization),
                "gt_normalized_stats": sparse_stats(gt_one, normalization=normalization),
                "pred_normalized_stats": sparse_stats(pred_one, normalization=normalization),
            }
            if normalization is not None:
                metrics["gt_denormalized_stats"] = sparse_stats(gt_one, normalization=normalization, denormalize=True)
                metrics["pred_denormalized_stats"] = sparse_stats(pred_one, normalization=normalization, denormalize=True)

            with open(sample_dir / "metrics.json", "w") as fp:
                json.dump(metrics, fp, indent=2)
            manifest["samples"].append(metrics)
            print(f"Saved {sample_dir}")

    with open(output_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Saved manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
