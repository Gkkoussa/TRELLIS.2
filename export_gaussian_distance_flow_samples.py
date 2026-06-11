import argparse
import copy
import csv
import glob
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
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
    parser.add_argument("--name", type=str, default=None, help="Name for this model in comparison output.")
    parser.add_argument("--compare_run_dir", type=str, default=None, help="Optional second flow run directory to export on the same selected samples.")
    parser.add_argument("--compare_config", type=str, default=None, help="Config JSON for --compare_run_dir. Defaults to <compare_run_dir>/config.json.")
    parser.add_argument("--compare_ckpt", type=str, default=None, help="Checkpoint for --compare_run_dir. Defaults to --ckpt.")
    parser.add_argument("--compare_ema_rate", type=str, default=None, help="EMA rate for --compare_run_dir. Defaults to --ema_rate.")
    parser.add_argument("--compare_name", type=str, default=None, help="Name for the comparison model output subdirectory.")
    parser.add_argument(
        "--compare_gaussian_distance_latent_name",
        type=str,
        default=None,
        help="Gaussian-distance latent name for --compare_run_dir. Defaults to --gaussian_distance_latent_name.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for exported samples.")
    parser.add_argument("--root", type=str, required=True, help="Processed dataset root.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split under <root>/splits/.")
    parser.add_argument(
        "--metadata_filter_csv",
        type=str,
        default=None,
        help="Optional metadata CSV whose sha256 rows define the export subset.",
    )
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


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    config_path: Path
    ckpt: str
    ema_rate: str | None
    gaussian_distance_latent_name: str


@dataclass
class PreparedRun:
    spec: RunSpec
    cfg: dict
    dataset: object
    trainer: object
    sampler: object
    amp_context: object
    ckpt_step: int
    ckpt_path: str
    output_dir: Path
    normalization: object
    metadata_filter_info: dict | None


def apply_metadata_filter(dataset, metadata_filter_csv: str) -> dict:
    with open(metadata_filter_csv, newline="") as f:
        allowed_sha256 = {row["sha256"] for row in csv.DictReader(f)}

    original_size = len(dataset.instances)
    old_instances = dataset.instances
    old_loads = getattr(dataset, "loads", None)

    if old_loads is not None and len(old_loads) == len(old_instances):
        kept = [
            (instance, load)
            for instance, load in zip(old_instances, old_loads)
            if instance[1] in allowed_sha256
        ]
        dataset.instances = [instance for instance, _ in kept]
        dataset.loads = [load for _, load in kept]
    else:
        dataset.instances = [
            instance for instance in old_instances if instance[1] in allowed_sha256
        ]

    if hasattr(dataset, "metadata") and len(dataset.metadata) > 0:
        keep_index = dataset.metadata.index.intersection(allowed_sha256)
        dataset.metadata = dataset.metadata.loc[keep_index]

    return {
        "metadata_filter_csv": str(Path(metadata_filter_csv).resolve()),
        "metadata_filter_allowed_sha256": len(allowed_sha256),
        "metadata_filter_original_size": original_size,
        "metadata_filter_removed": original_size - len(dataset.instances),
    }


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


def indices_for_sha256s(dataset, sha256s: list[str]) -> list[int]:
    index_by_sha = {sha: idx for idx, (_, sha) in enumerate(dataset.instances)}
    missing = [sha for sha in sha256s if sha not in index_by_sha]
    if missing:
        raise ValueError(f"Selected sha256 ids not present in comparison dataset: {missing}")
    return [index_by_sha[sha] for sha in sha256s]


def default_run_name(run_dir: Path, fallback: str) -> str:
    return run_dir.name or fallback


def make_unique_names(specs: list[RunSpec]) -> None:
    seen = {}
    for spec in specs:
        count = seen.get(spec.name, 0)
        seen[spec.name] = count + 1
        if count > 0:
            spec.name = f"{spec.name}_{count + 1}"


def build_run_specs(args) -> list[RunSpec]:
    run_dir = Path(args.run_dir).resolve()
    specs = [
        RunSpec(
            name=args.name or default_run_name(run_dir, "model"),
            run_dir=run_dir,
            config_path=Path(args.config).resolve() if args.config is not None else run_dir / "config.json",
            ckpt=args.ckpt,
            ema_rate=args.ema_rate,
            gaussian_distance_latent_name=args.gaussian_distance_latent_name,
        )
    ]
    if args.compare_run_dir is not None:
        compare_run_dir = Path(args.compare_run_dir).resolve()
        specs.append(
            RunSpec(
                name=args.compare_name or default_run_name(compare_run_dir, "compare"),
                run_dir=compare_run_dir,
                config_path=Path(args.compare_config).resolve()
                if args.compare_config is not None
                else compare_run_dir / "config.json",
                ckpt=args.compare_ckpt if args.compare_ckpt is not None else args.ckpt,
                ema_rate=args.compare_ema_rate if args.compare_ema_rate is not None else args.ema_rate,
                gaussian_distance_latent_name=args.compare_gaussian_distance_latent_name
                or args.gaussian_distance_latent_name,
            )
        )
    make_unique_names(specs)
    return specs


def build_dataset_for_run(spec: RunSpec, cfg: dict, args):
    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    if args.render_resolution is not None:
        dataset_args["snapshot_render_resolution"] = args.render_resolution

    root = Path(args.root).resolve()
    train_norm_path = (
        root
        / "splits"
        / "train"
        / "gaussian_distance_latents"
        / spec.gaussian_distance_latent_name
        / "normalization.json"
    )
    if train_norm_path.exists():
        dataset_args["gaussian_distance_slat_normalization_path"] = str(train_norm_path)

    data_dir = build_data_dir(
        root,
        args.split,
        spec.gaussian_distance_latent_name,
        args.michelangelo_latent_name,
    )
    return getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)


def prepare_run(spec: RunSpec, args, output_dir: Path) -> PreparedRun:
    cfg = json.load(open(spec.config_path, "r"))
    dataset = build_dataset_for_run(spec, cfg, args)
    metadata_filter_info = None
    if args.metadata_filter_csv is not None:
        metadata_filter_info = apply_metadata_filter(dataset, args.metadata_filter_csv)
        print(
            f"[{spec.name}] Applied metadata filter: "
            f"{metadata_filter_info['metadata_filter_original_size']} -> {len(dataset)} "
            f"instances, removed {metadata_filter_info['metadata_filter_removed']}"
        )

    model_dict = {
        name: getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda()
        for name, model_cfg in cfg["models"].items()
    }
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    trainer = getattr(trainers, cfg["trainer"]["name"])(
        model_dict,
        dataset,
        **trainer_args,
        output_dir=str(spec.run_dir),
        load_dir=None,
        step=None,
    )
    trainer.p_uncond = 0.0
    ckpt_step = find_ckpt_step(spec.run_dir, spec.ckpt)
    ckpt_path = load_denoiser_checkpoint(
        trainer.models["denoiser"],
        spec.run_dir,
        ckpt_step,
        spec.ema_rate,
        trainer.device,
    )
    trainer.models["denoiser"].eval()

    if trainer.mix_precision_mode == "amp":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

    output_dir.mkdir(parents=True, exist_ok=True)
    return PreparedRun(
        spec=spec,
        cfg=cfg,
        dataset=dataset,
        trainer=trainer,
        sampler=trainer.get_sampler(),
        amp_context=amp_context,
        ckpt_step=ckpt_step,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        normalization=getattr(dataset, "gaussian_distance_slat_normalization", None),
        metadata_filter_info=metadata_filter_info,
    )


def make_noise_like(x_0, seed: int):
    generator = torch.Generator(device=x_0.feats.device)
    generator.manual_seed(seed)
    feats = torch.randn(
        x_0.feats.shape,
        dtype=x_0.feats.dtype,
        device=x_0.feats.device,
        generator=generator,
    )
    return x_0.replace(feats)


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


def export_run(prepared: PreparedRun, args, indices: list[int], sha256s: list[str]):
    dataset = prepared.dataset
    trainer = prepared.trainer
    normalization = prepared.normalization
    manifest = {
        "name": prepared.spec.name,
        "run_dir": str(prepared.spec.run_dir),
        "config": str(prepared.spec.config_path),
        "gaussian_distance_latent_name": prepared.spec.gaussian_distance_latent_name,
        "checkpoint_step": prepared.ckpt_step,
        "checkpoint_path": prepared.ckpt_path,
        "ema_rate": prepared.spec.ema_rate,
        "split": args.split,
        "sampling_steps": args.sampling_steps,
        "guidance_strength": args.guidance_strength,
        "selected_sha256s": sha256s,
        "samples": [],
    }
    if prepared.metadata_filter_info is not None:
        manifest.update(prepared.metadata_filter_info)

    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        raw_items = [dataset[index] for index in batch_indices]
        batch = dataset.collate_fn(raw_items)
        batch = recursive_to_device(batch, "cuda")

        noise = make_noise_like(batch["x_0"], args.seed + batch_start)
        cond_args = trainer.get_inference_cond(
            batch["cond"],
            neg_cond=batch["neg_cond"],
            gaussian_distance_slat_cache_path=batch["gaussian_distance_slat_cache_path"],
        )
        with torch.no_grad():
            with prepared.amp_context():
                sample = prepared.sampler.sample(
                    trainer.models["denoiser"],
                    noise=noise,
                    **cond_args,
                    steps=args.sampling_steps,
                    guidance_strength=args.guidance_strength,
                    verbose=False,
                ).samples

        for local_idx, dataset_idx in enumerate(batch_indices):
            _, sha256 = dataset.instances[dataset_idx]
            sample_dir = prepared.output_dir / f"{dataset_idx:06d}_{sha256}"
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
            print(f"[{prepared.spec.name}] Saved {sample_dir}")

    with open(prepared.output_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"[{prepared.spec.name}] Saved manifest to {prepared.output_dir / 'manifest.json'}")
    return manifest


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    specs = build_run_specs(args)
    compare_mode = len(specs) > 1
    primary_spec = specs[0]
    primary_cfg = json.load(open(primary_spec.config_path, "r"))
    primary_dataset = build_dataset_for_run(primary_spec, primary_cfg, args)
    primary_metadata_filter_info = None
    if args.metadata_filter_csv is not None:
        primary_metadata_filter_info = apply_metadata_filter(primary_dataset, args.metadata_filter_csv)

    primary_indices = select_indices(primary_dataset, args)
    selected_sha256s = [primary_dataset.instances[index][1] for index in primary_indices]

    if compare_mode:
        default_output_dir = (
            primary_spec.run_dir
            / f"sample_debug_compare_{args.split}_step{find_ckpt_step(primary_spec.run_dir, primary_spec.ckpt):07d}"
        )
        output_base = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir
    else:
        ckpt_step = find_ckpt_step(primary_spec.run_dir, primary_spec.ckpt)
        output_base = (
            Path(args.output_dir).resolve()
            if args.output_dir is not None
            else primary_spec.run_dir / f"sample_debug_{args.split}_step{ckpt_step:07d}"
        )

    compare_manifest = {
        "split": args.split,
        "sampling_steps": args.sampling_steps,
        "guidance_strength": args.guidance_strength,
        "seed": args.seed,
        "selected_sha256s": selected_sha256s,
        "runs": [],
    }

    for spec_idx, spec in enumerate(specs):
        if compare_mode:
            run_output_dir = output_base / spec.name
        else:
            run_output_dir = output_base

        prepared = prepare_run(spec, args, run_output_dir)
        if spec_idx == 0 and primary_metadata_filter_info is not None:
            prepared.metadata_filter_info = primary_metadata_filter_info
        indices = indices_for_sha256s(prepared.dataset, selected_sha256s)
        manifest = export_run(prepared, args, indices, selected_sha256s)
        compare_manifest["runs"].append({
            "name": spec.name,
            "run_dir": str(spec.run_dir),
            "output_dir": str(run_output_dir),
            "checkpoint_step": prepared.ckpt_step,
            "checkpoint_path": prepared.ckpt_path,
            "gaussian_distance_latent_name": spec.gaussian_distance_latent_name,
            "samples": len(manifest["samples"]),
        })
        del prepared
        torch.cuda.empty_cache()

    if compare_mode:
        output_base.mkdir(parents=True, exist_ok=True)
        with open(output_base / "comparison_manifest.json", "w") as fp:
            json.dump(compare_manifest, fp, indent=2)
        print(f"Saved comparison manifest to {output_base / 'comparison_manifest.json'}")


if __name__ == "__main__":
    main()
