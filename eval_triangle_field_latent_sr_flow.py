import argparse
import copy
import glob
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm import tqdm

from trellis2 import datasets, models, trainers
from trellis2.modules import sparse as sp
from trellis2.utils.data_utils import recursive_to_device
from eval_metadata_filters import add_eval_metadata_filter_args, attach_eval_metadata_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a latent-space triangle-field super-resolution flow model."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance_strength", type=float, default=3.0)
    parser.add_argument(
        "--apply_conditioning_augmentation",
        action="store_true",
        help="Apply the trainer's configured conditioning augmentation to the positive conditioning path.",
    )
    parser.add_argument(
        "--conditioning_augmentation_noise_level",
        type=float,
        default=None,
        help="Eval-only override for conditioning_augmentation.noise_level.",
    )
    parser.add_argument(
        "--conditioning_augmentation_blur_sigma",
        type=float,
        default=None,
        help="Eval-only override/default for conditioning_augmentation.blur_sigma.",
    )
    parser.add_argument(
        "--conditioning_augmentation_disable_blur",
        action="store_true",
        help="Eval-only override that applies conditioning noise without sparse blur.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--low_resolution", type=int, default=64)
    parser.add_argument("--high_resolution", type=int, default=128)
    parser.add_argument(
        "--no_latents",
        action="store_true",
        help=(
            "Do not require precomputed high-resolution latents. Instead derive "
            "the latent support/cache from the high-resolution voxel support and "
            "sample unconditionally/conditionally on that support."
        ),
    )
    parser.add_argument(
        "--latent_name",
        type=str,
        default=None,
        help="Latent directory name under <root>/triangle_field_latents. Defaults to config data_dir.",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Optional renderer resolution override for saved grids.",
    )
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
            if dataset.num_voxels_column in dataset.metadata.columns else 1
            for _, sha256 in dataset.instances
        ]
    for stats in getattr(dataset, "_stats", {}).values():
        stats["Restricted to explicit instances"] = len(dataset.instances)
    return ordered


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        files = glob.glob(str(run_dir / "ckpts" / "misc_step*.pt"))
        if not files:
            files = glob.glob(str(run_dir / "ckpts" / "encoder_step*.pt"))
        if not files:
            raise RuntimeError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files)
    if ckpt == "none":
        raise ValueError("ckpt=none is not valid for evaluation.")
    return int(ckpt)


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.sqrt(images.shape[0])))
    tv_utils.save_image(images, str(path), nrow=nrow)


def build_data_dir(
    cfg: dict,
    root: Path,
    split: str,
    low_resolution: int,
    high_resolution: int,
    latent_name: str | None,
    args=None,
    require_latents: bool = True,
) -> dict:
    if require_latents and latent_name is None:
        configured = cfg.get("data_dir", {})
        if isinstance(configured, str):
            configured = json.loads(configured)
        try:
            latent_root = Path(configured["objxl4k_filtered"]["triangle_field_latent"])
        except KeyError as exc:
            raise ValueError("Could not infer latent root from config data_dir; pass --latent_name.") from exc
    elif require_latents:
        latent_root = root / "triangle_field_latents" / latent_name

    def triangle_field_voxel_root(resolution: int) -> Path:
        override = os.environ.get(f"TRIANGLE_FIELD_VOXEL_{resolution}_DIR")
        if override:
            return Path(override)
        return root / f"triangle_field_voxels_{resolution}"

    data_dir = {
        "objxl4k_filtered": {
            "low_triangle_field_voxel": str(triangle_field_voxel_root(low_resolution)),
            "high_triangle_field_voxel": str(triangle_field_voxel_root(high_resolution)),
        }
    }
    if require_latents:
        data_dir["objxl4k_filtered"]["triangle_field_latent"] = str(latent_root)
    return attach_eval_metadata_filter(data_dir, root, split, args)


def build_dataset(cfg: dict, root: Path, args):
    dataset_args = copy.deepcopy(cfg["dataset"]["args"])
    dataset_args["low_resolution"] = args.low_resolution
    dataset_args["high_resolution"] = args.high_resolution
    dataset_args["return_area_offsets"] = False
    dataset_args.pop("conditioning_augmentation", None)
    dataset_args.pop("instances_path", None)
    if getattr(args, "disable_dataset_density_conditioning", False):
        dataset_args["density_conditioning"] = False
    if args.render_resolution is not None:
        dataset_args["snapshot_render_resolution"] = args.render_resolution
    dataset_name = cfg["dataset"]["name"]
    if args.no_latents:
        dataset_name = "TriangleFieldSuperResolutionDataset"
        for key in (
            "latent_root_key",
            "latent_encoded_flag_column",
            "latent_tokens_column",
            "max_latent_tokens",
        ):
            dataset_args.pop(key, None)

    data_dir = build_data_dir(
        cfg,
        root,
        args.split,
        args.low_resolution,
        args.high_resolution,
        args.latent_name,
        args=args,
        require_latents=not args.no_latents,
    )
    dataset = getattr(datasets, dataset_name)(json.dumps(data_dir), **dataset_args)
    split_instances = root / "splits" / args.split / "instances.txt"
    if split_instances.exists():
        keep = {line.strip() for line in split_instances.read_text().splitlines() if line.strip()}
        dataset.instances = [(item_root, sha256) for item_root, sha256 in dataset.instances if sha256 in keep]
        if len(dataset.metadata) > 0:
            dataset.metadata = dataset.metadata[dataset.metadata.index.isin(keep)]
        if hasattr(dataset, "loads"):
            dataset.loads = [
                dataset.metadata.loc[sha256, dataset.num_voxels_column]
                if dataset.num_voxels_column in dataset.metadata.columns else 1
                for _, sha256 in dataset.instances
            ]
        for stats in getattr(dataset, "_stats", {}).values():
            stats[f"Restricted to split/{args.split}"] = len(dataset.instances)
    else:
        raise FileNotFoundError(f"Split instances file not found: {split_instances}")
    if getattr(args, "instances", None) is not None:
        restrict_dataset_to_instances(dataset, args.instances, None)
    return dataset, data_dir


def apply_conditioning_augmentation_overrides(trainer, args) -> None:
    noise_level = getattr(args, "conditioning_augmentation_noise_level", None)
    blur_sigma = getattr(args, "conditioning_augmentation_blur_sigma", None)
    disable_blur = bool(getattr(args, "conditioning_augmentation_disable_blur", False))
    if noise_level is None and blur_sigma is None and not disable_blur:
        return
    cfg = (
        copy.deepcopy(trainer.conditioning_augmentation)
        if trainer.conditioning_augmentation is not None
        else {"type": "sparse_blur_noise", "blur_sigma": 1.0, "noise_level": 0.0, "apply_prob": 1.0}
    )
    if noise_level is not None:
        cfg["noise_level"] = float(noise_level)
    if blur_sigma is not None:
        cfg["blur_sigma"] = float(blur_sigma)
    if disable_blur:
        cfg["disable_blur"] = True
    trainer._validate_conditioning_augmentation(cfg)
    trainer.conditioning_augmentation = cfg
    if hasattr(trainer, "_cond_blur_cache"):
        trainer._cond_blur_cache.clear()


def spatial_cache_scale_key(scale: int) -> str:
    return str((Fraction(scale, 1), Fraction(scale, 1), Fraction(scale, 1)))


def downsample_coords_for_channel_cache(coords: torch.Tensor, factor: int = 2):
    dim = coords.shape[-1] - 1
    coord = list(coords.unbind(dim=-1))
    for i in range(dim):
        coord[i + 1] = coord[i + 1] // factor
    subidx_parts = coords[:, 1:] % factor
    subidx = sum([subidx_parts[..., i] * factor ** i for i in range(dim)]).int()

    spatial_shape = coords[:, 1:].max(0)[0] + 1
    max_shape = [int((int(s.item()) + factor - 1) // factor) for s in spatial_shape]
    offsets = torch.cumprod(
        torch.tensor(max_shape[::-1], device=coords.device, dtype=torch.long),
        0,
    ).tolist()[::-1] + [1]
    code = sum([c.long() * o for c, o in zip(coord, offsets)])
    code, idx = code.unique(return_inverse=True)
    new_coords = torch.stack(
        [code // offsets[0]] +
        [(code // offsets[i + 1]) % max_shape[i] for i in range(dim)],
        dim=-1,
    ).int()
    return new_coords, idx.long(), subidx, torch.Size(max_shape)


def build_support_latents(
    high: sp.SparseTensor,
    latent_channels: int,
    levels: int = 4,
) -> tuple[sp.SparseTensor, list[dict]]:
    z_feats = []
    z_coords = []
    caches = []
    scale_tuple = (Fraction(2 ** levels, 1),) * 3
    for batch_idx, layout in enumerate(high.layout):
        coords = high.coords[layout].clone().int()
        coords[:, 0] = 0
        spatial_cache = {
            spatial_cache_scale_key(1): {
                "shape": torch.Size((coords[:, 1:].max(0)[0] + 1).tolist()),
            }
        }
        child_coords = coords
        for level in range(1, levels + 1):
            parent_coords, idx, subidx, shape = downsample_coords_for_channel_cache(child_coords)
            spatial_cache[spatial_cache_scale_key(2 ** level)] = {
                "channel2spatial_2": (child_coords, idx, subidx),
                "shape": shape,
            }
            child_coords = parent_coords

        sample_z_coords = child_coords.clone()
        sample_z_coords[:, 0] = batch_idx
        z_coords.append(sample_z_coords)
        z_feats.append(torch.zeros(
            (sample_z_coords.shape[0], latent_channels),
            device=high.feats.device,
            dtype=torch.float32,
        ))
        caches.append({"scale": scale_tuple, "spatial_cache": spatial_cache})

    return sp.SparseTensor(torch.cat(z_feats, dim=0), torch.cat(z_coords, dim=0)), caches


def build_trainer(cfg: dict, dataset, output_dir: Path):
    trainer_args = copy.deepcopy(cfg["trainer"]["args"])
    trainer_args["finetune_ckpt"] = None
    trainer_args["num_workers"] = 1
    trainer_args["parallel_mode"] = None
    trainer_args["prefetch_data"] = False
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
    trainer.models["encoder"].eval()
    trainer.decoder.eval()
    return trainer


def load_encoder_checkpoint(trainer, run_dir: Path, step: int, ema_rate: str | None) -> str:
    if ema_rate is None:
        path = run_dir / "ckpts" / f"encoder_step{step:07d}.pt"
    else:
        path = run_dir / "ckpts" / f"encoder_ema{ema_rate}_step{step:07d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    state = torch.load(path, map_location=trainer.device, weights_only=True)
    trainer.models["encoder"].load_state_dict(state)
    trainer.models["encoder"].eval()
    return str(path)


def slice_batch(data: dict, batch: int) -> dict:
    sliced = {}
    for key, value in data.items():
        if isinstance(value, str):
            sliced[key] = value
        elif isinstance(value, list):
            sliced[key] = value[:batch]
        elif hasattr(value, "__getitem__"):
            sliced[key] = value[:batch]
        else:
            sliced[key] = value
    return sliced


@torch.no_grad()
def predict_z0(
    trainer,
    z_t: sp.SparseTensor,
    cond: sp.SparseTensor,
    caches,
    cache_paths,
    t: float,
    density_cond: sp.SparseTensor | None = None,
) -> sp.SparseTensor:
    x_t = trainer._decode_latents_with_cache(z_t, caches=caches, cache_paths=cache_paths)
    if not torch.equal(x_t.coords, cond.coords):
        raise ValueError(f"Decoded z_t coords must match cond coords: {x_t.coords.shape} vs {cond.coords.shape}")
    mode = getattr(trainer, "latent_self_conditioning_mode", "none")
    if mode == "input":
        latent_cond_input, _ = trainer._latent_to_field_support(z_t, x_t)
        enc_in = trainer._make_encoder_input(x_t, cond)
        enc_in = enc_in.replace(torch.cat([enc_in.feats, latent_cond_input.feats], dim=-1))
        if density_cond is not None:
            if not torch.equal(enc_in.coords, density_cond.coords):
                raise ValueError(
                    f"density_cond coords must match encoder input coords: "
                    f"{density_cond.coords.shape} vs {enc_in.coords.shape}"
                )
            enc_in = enc_in.replace(torch.cat([enc_in.feats, density_cond.feats], dim=-1))
        encoder_kwargs = {}
    elif mode == "bottleneck":
        enc_in = trainer._make_encoder_input(x_t, cond, density_cond)
        encoder_kwargs = {"latent_cond": z_t}
    else:
        enc_in = trainer._make_encoder_input(x_t, cond, density_cond)
        encoder_kwargs = {}
    batch_t = torch.full((z_t.shape[0],), t * 1000.0, device=z_t.feats.device, dtype=torch.float32)
    pred_z0 = trainer.models["encoder"](enc_in, batch_t, sample_posterior=False, **encoder_kwargs)
    if not torch.equal(pred_z0.coords, z_t.coords):
        raise ValueError(f"Predicted latent coords must match z_t coords: {pred_z0.coords.shape} vs {z_t.coords.shape}")
    return pred_z0


@torch.no_grad()
def sample_latent_sr(
    trainer,
    z_0: sp.SparseTensor,
    cond: sp.SparseTensor,
    caches,
    cache_paths,
    steps: int,
    guidance_strength: float,
    apply_conditioning_augmentation: bool = False,
    density_cond: sp.SparseTensor | None = None,
):
    z_t = z_0.replace(torch.randn_like(z_0.feats))
    cond_pos = trainer._augment_conditioning(cond) if apply_conditioning_augmentation else cond
    zero_cond = cond.replace(torch.zeros_like(cond.feats))
    t_seq = np.linspace(1.0, 0.0, steps + 1).tolist()
    pred_z0_last = None
    for t, t_prev in tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc="Sampling latent SR"):
        if guidance_strength == 0.0:
            pred_z0 = predict_z0(trainer, z_t, zero_cond, caches, cache_paths, float(t), density_cond)
        else:
            pred_pos = predict_z0(trainer, z_t, cond_pos, caches, cache_paths, float(t), density_cond)
        if guidance_strength == 1.0:
            pred_z0 = pred_pos
        elif guidance_strength != 0.0:
            pred_neg = predict_z0(trainer, z_t, zero_cond, caches, cache_paths, float(t), density_cond)
            pred_z0 = pred_pos.replace(
                guidance_strength * pred_pos.feats + (1.0 - guidance_strength) * pred_neg.feats
            )
        velocity = (z_t.feats - pred_z0.feats) / max(float(t), 1e-5)
        z_t = z_t.replace(z_t.feats - (float(t) - float(t_prev)) * velocity)
        pred_z0_last = pred_z0
    return z_t, pred_z0_last


def append_visuals(dataset, store: dict, prefix: str, tensor: sp.SparseTensor, batch: int) -> None:
    vis = dataset.visualize_sample({"target": tensor})
    for key, value in vis.items():
        store.setdefault(f"{prefix}_{key}", []).append(value[:batch].cpu())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    root = Path(args.root).resolve()
    cfg = load_config(run_dir)
    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir
        / (
            f"eval_filtered_{args.split}_{args.low_resolution}to{args.high_resolution}"
            f"{'_support_only' if args.no_latents else ''}"
            f"_latent_flow_sampling_step{ckpt_step:07d}_cfg{args.guidance_strength:g}_n{args.num_samples}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, data_dir = build_dataset(cfg, root, args)
    trainer = build_trainer(cfg, dataset, output_dir)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    images = {}
    remaining = args.num_samples
    for data in loader:
        batch = min(remaining, args.batch_size)
        data = recursive_to_device(slice_batch(data, batch), trainer.device)
        cache_paths = None
        if args.no_latents:
            latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])
            data["z_0"], caches = build_support_latents(data["x_0"], latent_channels)
        else:
            caches = data.get("triangle_field_slat_cache", None)
            cache_paths = data.get("triangle_field_slat_cache_path", None)

        sample_z, pred_z0_last = sample_latent_sr(
            trainer,
            data["z_0"],
            data["cond"],
            caches,
            cache_paths,
            args.steps,
            args.guidance_strength,
            args.apply_conditioning_augmentation,
            data.get("density_cond", None),
        )
        gt = data["x_0"] if args.no_latents else trainer._decode_latents_with_cache(data["z_0"], caches=caches, cache_paths=cache_paths)
        sample = trainer._decode_latents_with_cache(sample_z, caches=caches, cache_paths=cache_paths)
        pred_last = trainer._decode_latents_with_cache(pred_z0_last, caches=caches, cache_paths=cache_paths)

        append_visuals(dataset, images, "gt", gt, batch)
        append_visuals(dataset, images, "cond", data["cond"], batch)
        append_visuals(dataset, images, "sample", sample, batch)
        append_visuals(dataset, images, "pred_z0_last", pred_last, batch)

        remaining -= batch
        if remaining <= 0:
            break

    suffix = f"step{ckpt_step:07d}_cfg{args.guidance_strength:g}_{args.low_resolution}to{args.high_resolution}"
    for name, chunks in images.items():
        save_image_grid(torch.cat(chunks, dim=0)[: args.num_samples], output_dir / f"{name}_{suffix}.jpg")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "root": str(root),
        "split": args.split,
        "data_dir": data_dir,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "steps": args.steps,
        "guidance_strength": args.guidance_strength,
        "apply_conditioning_augmentation": args.apply_conditioning_augmentation,
        "conditioning_augmentation": getattr(trainer, "conditioning_augmentation", None),
        "low_resolution": args.low_resolution,
        "high_resolution": args.high_resolution,
        "no_latents": args.no_latents,
        "latent_name": args.latent_name,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
