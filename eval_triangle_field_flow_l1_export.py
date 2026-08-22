import argparse
import csv
import glob
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import utils3d
from tqdm import tqdm
from torchvision import utils as tv_utils

from trellis2 import datasets, models, trainers
from trellis2.modules.sparse import sparse_cat
from trellis2.renderers import VoxelRenderer
from trellis2.representations import Voxel
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a triangle-field flow model, decode individual generations, and report d_tri/d_vert L1."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing ckpts/.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON. Defaults to <run_dir>/config.json.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint to evaluate: latest or an integer step.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for metrics and individual exports.")
    parser.add_argument("--data_dir", type=str, required=True, help="JSON data_dir with canonical latent roots and filters.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of dataset items to sample/export. Use <=0 for all.")
    parser.add_argument("--batch_size", type=int, default=4, help="Flow sampling batch size.")
    parser.add_argument("--decode_batch_size", type=int, default=4, help="VAE decoder batch size.")
    parser.add_argument("--snapshot_num_samples", type=int, default=None, help="Old-style visualization grid sample count. Defaults to --num_samples when positive.")
    parser.add_argument("--snapshot_batch_size", type=int, default=None, help="Batch size for old-style trainer.snapshot visualizations.")
    parser.add_argument("--sampling_steps", type=int, default=12, help="Euler sampling steps.")
    parser.add_argument("--guidance_strength", type=float, default=1.0, help="Classifier-free guidance strength.")
    parser.add_argument("--ema_rate", type=str, default=None, help="Optional EMA rate to evaluate, e.g. 0.9999.")
    parser.add_argument("--shuffle", action="store_true", help="Randomly select examples before applying --num_samples.")
    parser.add_argument("--no_individual_visualizations", action="store_true", help="Skip per-sample GT/pred d_tri/d_vert JPG exports.")
    parser.add_argument("--seed", type=int, default=0)
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
    return int(ckpt.removeprefix("step"))


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


def sparse_to_npz(path: Path, tensor, feats: torch.Tensor | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = tensor.coords[:, 1:].detach().cpu().numpy().astype(np.uint16)
    if feats is None:
        feats = tensor.feats
    feats = feats.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, coords=coords, feats=feats)


def triangle_fields_to_npz(path: Path, tensor, feats_01: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = tensor.coords[:, 1:].detach().cpu().numpy().astype(np.uint16)
    feats = feats_01.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        path,
        coords=coords,
        features=feats,
        d_tri=feats[:, 0],
        d_vert=feats[:, 1],
    )


def save_paired_triangle_fields(path: Path, gt, pred, gt_feats_01: torch.Tensor, pred_feats_01: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_coords = gt.coords[:, 1:].int()
    pred_coords = pred.coords[:, 1:].int()

    if gt_coords.shape == pred_coords.shape and torch.equal(gt_coords, pred_coords):
        coords = gt_coords.detach().cpu().numpy().astype(np.uint16)
        gt_feats = gt_feats_01.detach().cpu().numpy().astype(np.float32)
        pred_feats = pred_feats_01.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(
            path,
            coords=coords,
            gt_features=gt_feats,
            pred_features=pred_feats,
            gt_d_tri=gt_feats[:, 0],
            gt_d_vert=gt_feats[:, 1],
            pred_d_tri=pred_feats[:, 0],
            pred_d_vert=pred_feats[:, 1],
            coord_mode=np.array("exact"),
        )
        return

    gt_map = {tuple(c.tolist()): i for i, c in enumerate(gt_coords.cpu())}
    pred_map = {tuple(c.tolist()): i for i, c in enumerate(pred_coords.cpu())}
    common = sorted(set(gt_map) & set(pred_map))
    coords = np.array(common, dtype=np.uint16)
    gt_idx = torch.tensor([gt_map[c] for c in common], device=gt_feats_01.device)
    pred_idx = torch.tensor([pred_map[c] for c in common], device=pred_feats_01.device)
    gt_feats = gt_feats_01[gt_idx].detach().cpu().numpy().astype(np.float32)
    pred_feats = pred_feats_01[pred_idx].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        path,
        coords=coords,
        gt_features=gt_feats,
        pred_features=pred_feats,
        gt_d_tri=gt_feats[:, 0],
        gt_d_vert=gt_feats[:, 1],
        pred_d_tri=pred_feats[:, 0],
        pred_d_vert=pred_feats[:, 1],
        coord_mode=np.array("common"),
    )


def decoded_feats_01(dataset, tensor):
    feats = tensor.feats[:, :2].float()
    if getattr(dataset, "triangle_field_distance_transform", "none") == "minus_one_one":
        feats = feats * 0.5 + 0.5
    return feats


def render_decoded_triangle_fields(dataset, voxels):
    render_resolution = int(getattr(dataset, "snapshot_render_resolution", 512))
    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = 4

    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
    yaws = [y + yaws_offset for y in yaws]
    pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

    exts = []
    ints = []
    for yaw, pitch_i in zip(yaws, pitch):
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch_i),
            np.cos(yaw) * np.cos(pitch_i),
            np.sin(pitch_i),
        ]).float().cuda() * 2
        fov = torch.deg2rad(torch.tensor(30)).cuda()
        extrinsics = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 0, 1]).float().cuda(),
        )
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        exts.append(extrinsics)
        ints.append(intrinsics)

    images = {"d_tri": [], "d_vert": []}
    layout = {"d_tri": slice(0, 1), "d_vert": slice(1, 2)}
    for voxel in voxels:
        rep = Voxel(
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / dataset.resolution,
            coords=voxel.coords[:, 1:].contiguous(),
            attrs=None,
            layout={"color": slice(0, 3)},
        )
        feats = decoded_feats_01(dataset, voxel).clamp(0, 1)
        for key, channel in layout.items():
            image = torch.zeros(3, render_resolution * 2, render_resolution * 2, dtype=torch.float32).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                attr = feats[:, channel].reshape(-1, 1).expand(-1, 3).float()
                with torch.autocast(device_type="cuda", enabled=False):
                    res = renderer.render(rep, ext.float(), intr.float(), colors_overwrite=attr)
                image[
                    :,
                    render_resolution * (j // tile[1]):render_resolution * (j // tile[1] + 1),
                    render_resolution * (j % tile[1]):render_resolution * (j % tile[1] + 1),
                ] = res["color"].float()
            images[key].append(image)

    return {key: torch.stack(value, dim=0) for key, value in images.items()}


def save_individual_visualizations(sample_dir: Path, gt_images: dict, pred_images: dict, index: int):
    sample_dir.mkdir(parents=True, exist_ok=True)
    for key in ("d_tri", "d_vert"):
        tv_utils.save_image(gt_images[key][index].detach().cpu().clamp(0, 1), sample_dir / f"gt_{key}.jpg")
        tv_utils.save_image(pred_images[key][index].detach().cpu().clamp(0, 1), sample_dir / f"pred_{key}.jpg")


def compare_decoded(dataset, gt, pred):
    gt_feats = decoded_feats_01(dataset, gt)
    pred_feats = decoded_feats_01(dataset, pred)
    gt_coords = gt.coords[:, 1:].int()
    pred_coords = pred.coords[:, 1:].int()

    coord_exact = gt_coords.shape == pred_coords.shape and torch.equal(gt_coords, pred_coords)
    if coord_exact:
        diff = (pred_feats - gt_feats).abs()
        return {
            "coord_exact": True,
            "gt_tokens": int(gt_feats.shape[0]),
            "pred_tokens": int(pred_feats.shape[0]),
            "common_tokens": int(gt_feats.shape[0]),
            "d_tri_l1": float(diff[:, 0].mean().item()),
            "d_vert_l1": float(diff[:, 1].mean().item()),
            "l1": float(diff.mean().item()),
        }

    gt_map = {tuple(c.tolist()): i for i, c in enumerate(gt_coords.cpu())}
    pred_map = {tuple(c.tolist()): i for i, c in enumerate(pred_coords.cpu())}
    common = sorted(set(gt_map) & set(pred_map))
    if len(common) == 0:
        return {
            "coord_exact": False,
            "gt_tokens": int(gt_feats.shape[0]),
            "pred_tokens": int(pred_feats.shape[0]),
            "common_tokens": 0,
            "d_tri_l1": None,
            "d_vert_l1": None,
            "l1": None,
        }

    gt_idx = torch.tensor([gt_map[c] for c in common], device=gt_feats.device)
    pred_idx = torch.tensor([pred_map[c] for c in common], device=pred_feats.device)
    diff = (pred_feats[pred_idx] - gt_feats[gt_idx]).abs()
    return {
        "coord_exact": False,
        "gt_tokens": int(gt_feats.shape[0]),
        "pred_tokens": int(pred_feats.shape[0]),
        "common_tokens": int(len(common)),
        "d_tri_l1": float(diff[:, 0].mean().item()),
        "d_vert_l1": float(diff[:, 1].mean().item()),
        "l1": float(diff.mean().item()),
    }


def write_rows_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config is not None else run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = json.load(open(config_path, "r"))
    dataset_args = json.loads(json.dumps(cfg["dataset"]["args"]))
    trainer_args = json.loads(json.dumps(cfg["trainer"]["args"]))
    data_dir = json.loads(args.data_dir)

    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / f"eval_{args.split}_l1_export_step{ckpt_step:07d}"
    )
    export_dir = output_dir / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    dataset = getattr(datasets, cfg["dataset"]["name"])(json.dumps(data_dir), **dataset_args)
    print(f"Dataset size after filters: {len(dataset)}")

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

    indices = np.arange(len(dataset))
    if args.shuffle:
        rng.shuffle(indices)
    if args.num_samples > 0:
        indices = indices[:args.num_samples]
    indices = indices.tolist()

    sampler = trainer.get_sampler()
    if trainer.mix_precision_mode == "amp":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

    rows = []
    d_tri_sum = 0.0
    d_vert_sum = 0.0
    l1_sum = 0.0
    compared = 0
    exact_coords = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(indices), args.batch_size), desc="Sampling/exporting"):
            batch_indices = indices[start:start + args.batch_size]
            samples = []
            sha256s = []
            for idx in batch_indices:
                root, sha256 = dataset.instances[idx]
                samples.append(dataset.get_instance(root, sha256))
                sha256s.append(str(sha256))

            batch = dataset.collate_fn(samples)
            batch = recursive_to_device(batch, trainer.device)
            noise = batch["x_0"].replace(torch.randn_like(batch["x_0"].feats))

            inference_batch = {k: v for k, v in batch.items() if k != "x_0"}
            inference_args = trainer.get_inference_cond(**inference_batch)
            sample_kwargs = {
                "steps": args.sampling_steps,
                "verbose": False,
            }
            if "neg_cond" in inference_args:
                sample_kwargs["guidance_strength"] = args.guidance_strength

            with amp_context():
                res = sampler.sample(
                    trainer.models["denoiser"],
                    noise=noise,
                    **inference_args,
                    **sample_kwargs,
                )
            pred_z = res.samples
            gt_z = batch["x_0"]
            shape_z = batch["concat_cond"]

            z_cat = sparse_cat([gt_z, pred_z], dim=0)
            shape_cat = sparse_cat([shape_z, shape_z], dim=0)
            decoded = dataset.decode_latent(z_cat, shape_z=shape_cat, batch_size=args.decode_batch_size)
            gt_decoded = decoded[:len(batch_indices)]
            pred_decoded = decoded[len(batch_indices):]
            if args.no_individual_visualizations:
                gt_images = None
                pred_images = None
            else:
                paired_images = render_decoded_triangle_fields(dataset, gt_decoded + pred_decoded)
                gt_images = {key: value[:len(batch_indices)] for key, value in paired_images.items()}
                pred_images = {key: value[len(batch_indices):] for key, value in paired_images.items()}

            for i, sha256 in enumerate(sha256s):
                sample_dir = export_dir / f"{start + i:05d}_{sha256}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                gt_dec = gt_decoded[i]
                pred_dec = pred_decoded[i]
                gt_feats_01 = decoded_feats_01(dataset, gt_dec)
                pred_feats_01 = decoded_feats_01(dataset, pred_dec)

                sparse_to_npz(sample_dir / "gt_latent_normalized.npz", gt_z[i])
                sparse_to_npz(sample_dir / "pred_latent_normalized.npz", pred_z[i])
                sparse_to_npz(sample_dir / "gt_decoded_dtri_dvert.npz", gt_dec, gt_feats_01)
                sparse_to_npz(sample_dir / "pred_decoded_dtri_dvert.npz", pred_dec, pred_feats_01)
                triangle_fields_to_npz(sample_dir / "gt_triangle_fields.npz", gt_dec, gt_feats_01)
                triangle_fields_to_npz(sample_dir / "pred_triangle_fields.npz", pred_dec, pred_feats_01)
                save_paired_triangle_fields(
                    sample_dir / "gt_pred_triangle_fields.npz",
                    gt_dec,
                    pred_dec,
                    gt_feats_01,
                    pred_feats_01,
                )
                if gt_images is not None and pred_images is not None:
                    save_individual_visualizations(sample_dir, gt_images, pred_images, i)

                metrics = compare_decoded(dataset, gt_dec, pred_dec)
                metrics.update({
                    "sha256": sha256,
                    "index": int(batch_indices[i]),
                    "export_dir": str(sample_dir),
                })
                with open(sample_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                rows.append(metrics)

                if metrics["l1"] is not None:
                    d_tri_sum += metrics["d_tri_l1"]
                    d_vert_sum += metrics["d_vert_l1"]
                    l1_sum += metrics["l1"]
                    compared += 1
                exact_coords += int(metrics["coord_exact"])

    summary = {
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "ema_rate": args.ema_rate,
        "split": args.split,
        "dataset_size": len(dataset),
        "num_requested": len(indices),
        "num_compared": compared,
        "coord_exact_rate": exact_coords / len(rows) if rows else None,
        "d_tri_l1": d_tri_sum / compared if compared else None,
        "d_vert_l1": d_vert_sum / compared if compared else None,
        "l1": l1_sum / compared if compared else None,
        "sampling_steps": args.sampling_steps,
        "guidance_strength": args.guidance_strength,
        "output_dir": str(output_dir),
        "generation_dir": str(export_dir),
    }

    write_rows_csv(output_dir / "per_sample_metrics.csv", rows)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved per-sample metrics to {output_dir / 'per_sample_metrics.csv'}")
    print(f"Saved individual generations under {export_dir}")

    snapshot_num_samples = args.snapshot_num_samples
    if snapshot_num_samples is None:
        snapshot_num_samples = len(indices) if args.num_samples > 0 else 16
    if snapshot_num_samples > 0:
        snapshot_batch_size = args.snapshot_batch_size or args.batch_size
        suffix = f"{args.split}_step{ckpt_step:07d}_l1_export"
        trainer.snapshot(
            suffix=suffix,
            num_samples=snapshot_num_samples,
            batch_size=snapshot_batch_size,
            steps=args.sampling_steps,
            guidance_strength=args.guidance_strength,
        )
        print(f"Saved old-style visualization grids to {output_dir / 'samples' / suffix}")


if __name__ == "__main__":
    main()
