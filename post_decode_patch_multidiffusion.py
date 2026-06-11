import argparse
import copy
import csv
import glob
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import utils3d

from trellis2 import datasets, models
from trellis2.modules import sparse as sp
from trellis2.pipelines import samplers
from trellis2.renderers import VoxelRenderer
from trellis2.representations import Voxel
from trellis2.utils.data_utils import recursive_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 512 Gaussian-distance voxels from the latent flow, decode them, "
            "then refine the decoded 6-channel features with MultiDiffusion-style "
            "overlapping sparse patch denoising."
        )
    )
    parser.add_argument("--latent_run_dir", type=str, required=True)
    parser.add_argument("--latent_config", type=str, default=None)
    parser.add_argument("--latent_ckpt", type=str, default="latest")
    parser.add_argument("--latent_ema_rate", type=str, default=None)
    parser.add_argument("--latent_sampling_steps", type=int, default=12)
    parser.add_argument("--latent_guidance_strength", type=float, default=3.0)

    parser.add_argument("--patch_run_dir", type=str, required=True)
    parser.add_argument("--patch_config", type=str, default=None)
    parser.add_argument("--patch_ckpt", type=str, default="latest")
    parser.add_argument("--patch_ema_rate", type=str, default=None)
    parser.add_argument("--patch_start_t", type=float, default=0.5)
    parser.add_argument("--patch_steps", type=int, default=12)
    parser.add_argument("--patch_batch_size", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=16)
    parser.add_argument("--max_patches", type=int, default=4096)
    parser.add_argument("--blend_mode", type=str, default="triangular", choices=["uniform", "triangular"])

    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gaussian_distance_latent_name", type=str, required=True)
    parser.add_argument("--michelangelo_latent_name", type=str, required=True)
    parser.add_argument("--metadata_filter_csv", type=str, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--sha256s", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--render_resolution", type=int, default=256)
    parser.add_argument("--render_ssaa", type=int, default=2)
    parser.add_argument("--save_vxz", action="store_true")
    parser.add_argument("--num_read_threads", type=int, default=4)
    return parser.parse_args()


def unique_output_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{idx:03d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create a unique output directory for {path}")


def find_ckpt_step(run_dir: Path, ckpt: str) -> int:
    if ckpt == "latest":
        misc_files = glob.glob(str(run_dir / "ckpts" / "misc_*.pt"))
        if misc_files:
            return max(int(os.path.basename(path).split("step")[-1].split(".")[0]) for path in misc_files)
        denoiser_files = glob.glob(str(run_dir / "ckpts" / "denoiser_step*.pt"))
        if not denoiser_files:
            raise FileNotFoundError(f"No checkpoints found under {run_dir / 'ckpts'}")
        return max(int(os.path.basename(path).split("step")[-1].split(".")[0]) for path in denoiser_files)
    return int(ckpt.removeprefix("step"))


def load_denoiser_checkpoint(model, run_dir: Path, step: int, ema_rate: str | None, device: torch.device) -> str:
    if ema_rate is None:
        path = run_dir / "ckpts" / f"denoiser_step{step:07d}.pt"
    else:
        path = run_dir / "ckpts" / f"denoiser_ema{ema_rate}_step{step:07d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Denoiser checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return str(path)


def build_latent_data_dir(root: Path, split: str, gaussian_distance_latent_name: str, michelangelo_latent_name: str) -> dict:
    split_root = root / "splits" / split
    return {
        split: {
            "metadata": str(split_root),
            "gaussian_distance_latent": str(split_root / "gaussian_distance_latents" / gaussian_distance_latent_name),
            "michelangelo_latent": str(split_root / "michelangelo_latents" / michelangelo_latent_name),
        }
    }


def apply_metadata_filter(dataset, metadata_filter_csv: str) -> dict:
    with open(metadata_filter_csv, newline="") as f:
        allowed_sha256 = {row["sha256"] for row in csv.DictReader(f)}

    original_size = len(dataset.instances)
    old_instances = dataset.instances
    old_loads = getattr(dataset, "loads", None)
    if old_loads is not None and len(old_loads) == len(old_instances):
        kept = [(instance, load) for instance, load in zip(old_instances, old_loads) if instance[1] in allowed_sha256]
        dataset.instances = [instance for instance, _ in kept]
        dataset.loads = [load for _, load in kept]
    else:
        dataset.instances = [instance for instance in old_instances if instance[1] in allowed_sha256]

    if hasattr(dataset, "metadata") and len(dataset.metadata) > 0:
        keep_index = dataset.metadata.index.intersection(allowed_sha256)
        dataset.metadata = dataset.metadata.loc[keep_index]

    return {
        "metadata_filter_csv": str(Path(metadata_filter_csv).resolve()),
        "metadata_filter_allowed_sha256": len(allowed_sha256),
        "metadata_filter_original_size": original_size,
        "metadata_filter_removed": original_size - len(dataset.instances),
    }


def select_indices(dataset, args) -> list[int]:
    if args.indices:
        return [int(index) for index in args.indices.split(",") if index.strip()]
    if args.sha256s:
        wanted = {sha.strip() for sha in args.sha256s.split(",") if sha.strip()}
        selected = [idx for idx, (_, sha) in enumerate(dataset.instances) if sha in wanted]
        missing = sorted(wanted - {dataset.instances[idx][1] for idx in selected})
        if missing:
            raise ValueError(f"Requested sha256 ids not present after filtering: {missing}")
        return selected
    if args.random:
        rng = np.random.default_rng(args.seed)
        count = min(args.num_samples, len(dataset))
        return rng.choice(len(dataset), size=count, replace=False).tolist()
    stop = min(args.start_index + args.num_samples, len(dataset))
    return list(range(args.start_index, stop))


def make_amp_context(config: dict):
    trainer_args = config.get("trainer", {}).get("args", {})
    if trainer_args.get("mix_precision_mode") == "amp":
        dtype_name = trainer_args.get("mix_precision_dtype", "float16")
        dtype = getattr(torch, dtype_name)
        return lambda: torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext


def build_model(config: dict) -> torch.nn.Module:
    model_cfg = config["models"]["denoiser"]
    return getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda().eval()


@torch.no_grad()
def sample_latent_flow(latent_model, latent_config, batch, steps: int, guidance_strength: float, amp_context):
    trainer_args = latent_config["trainer"]["args"]
    sigma_min = trainer_args.get("sigma_min", 1e-5)
    noise_scale = trainer_args.get("noise_scale", 1.0)
    sampler = samplers.FlowEulerCfgSampler(sigma_min, noise_scale=noise_scale)
    noise = batch["x_0"].replace(torch.randn_like(batch["x_0"].feats))
    with amp_context():
        sample = sampler.sample(
            latent_model,
            noise=noise,
            cond=batch["cond"],
            neg_cond=batch["neg_cond"],
            steps=steps,
            guidance_strength=guidance_strength,
            verbose=True,
            tqdm_desc="Sampling 512 latent flow",
            gaussian_distance_slat_cache_path=batch["gaussian_distance_slat_cache_path"],
        ).samples
    return sample


def build_patch_origins(
    coords: torch.Tensor,
    resolution: int,
    patch_size: int,
    stride: int,
    max_patches: int | None,
    seed: int,
) -> list[tuple[int, int, int]]:
    if coords.numel() == 0:
        return []
    if stride <= 0:
        raise ValueError("patch_stride must be positive")
    max_origin = resolution - patch_size
    if max_origin < 0:
        raise ValueError(f"patch_size {patch_size} exceeds resolution {resolution}")

    coords_cpu = coords.detach().cpu().long()
    # Dedupe first so origin generation scales with the coarse stride grid, not
    # with every active voxel in a potentially large 512^3 sparse output.
    base = torch.unique(torch.div(coords_cpu, stride, rounding_mode="floor") * stride, dim=0)
    min_offset = -((patch_size - 1) // stride) * stride
    offsets = list(range(min_offset, 1, stride))
    origins = set()
    for base_origin in base.tolist():
        for ox in offsets:
            for oy in offsets:
                for oz in offsets:
                    origin = (
                        min(max(base_origin[0] + ox, 0), max_origin),
                        min(max(base_origin[1] + oy, 0), max_origin),
                        min(max(base_origin[2] + oz, 0), max_origin),
                    )
                    origins.add(origin)

    origins = sorted(origins)
    if max_patches is not None and len(origins) > max_patches:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(origins), size=max_patches, replace=False)
        origins = [origins[i] for i in sorted(keep.tolist())]
    return origins


def make_sparse_batch(patches: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> sp.SparseTensor:
    feats = []
    coords = []
    for batch_idx, (local_coords, local_feats) in enumerate(patches):
        batch_col = torch.full((local_coords.shape[0], 1), batch_idx, dtype=torch.int32)
        coords.append(torch.cat([batch_col, local_coords.cpu().int()], dim=1))
        feats.append(local_feats.cpu().float())
    return sp.SparseTensor(torch.cat(feats, dim=0).to(device), torch.cat(coords, dim=0).to(device))


def patch_blend_weights(local_coords: torch.Tensor, patch_size: int, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones((local_coords.shape[0], 1), dtype=torch.float32)
    center = patch_size / 2.0
    half = patch_size / 2.0
    pos = local_coords.float() + 0.5
    per_axis = (1.0 - torch.abs(pos - center) / half).clamp(min=1.0e-3)
    return per_axis.prod(dim=1, keepdim=True)


@torch.no_grad()
def multidiffusion_sparse_patch_refine(
    patch_model,
    patch_config: dict,
    coords: torch.Tensor,
    feats: torch.Tensor,
    origins: list[tuple[int, int, int]],
    patch_size: int,
    start_t: float,
    steps: int,
    batch_size: int,
    blend_mode: str,
    amp_context,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if start_t <= 0.0 or start_t > 1.0:
        raise ValueError("patch_start_t must be in (0, 1].")

    device = next(patch_model.parameters()).device
    trainer_args = patch_config["trainer"]["args"]
    sigma_min = trainer_args.get("sigma_min", 1e-5)
    noise_scale = trainer_args.get("noise_scale", 1.0)
    sampler = samplers.FlowEulerSampler(sigma_min, noise_scale=noise_scale)
    dataset_args = patch_config.get("dataset", {}).get("args", {})
    cond_as_token = dataset_args.get("cond_as_token", True)

    coords_cpu = coords.detach().cpu().long()
    feats_cpu = feats.detach().cpu().float()

    noise = torch.randn_like(feats_cpu)
    noise_coeff = noise_scale * (sigma_min + (1.0 - sigma_min) * start_t)
    current_feats = (1.0 - start_t) * feats_cpu + noise_coeff * noise
    coverage_counts = torch.zeros((feats_cpu.shape[0], 1), dtype=torch.float32)

    total_patch_tokens = 0
    used_patch_steps = 0
    skipped_empty = 0
    t_seq = np.linspace(start_t, 0.0, steps + 1).tolist()

    for step_idx, (t_cur, t_prev) in enumerate(tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc="MultiDiffusion patch steps")):
        accum = torch.zeros_like(feats_cpu)
        weights = torch.zeros((feats_cpu.shape[0], 1), dtype=torch.float32)

        for batch_start in tqdm(
            range(0, len(origins), batch_size),
            desc=f"Patches t={t_cur:.3f}->{t_prev:.3f}",
            leave=False,
        ):
            batch_origins = origins[batch_start:batch_start + batch_size]
            patches = []
            patch_indices = []
            patch_weights = []
            for origin_tuple in batch_origins:
                origin = torch.tensor(origin_tuple, dtype=torch.long)
                patch_max = origin + patch_size
                mask = torch.all((coords_cpu >= origin) & (coords_cpu < patch_max), dim=1)
                selected = mask.nonzero(as_tuple=False).flatten()
                if selected.numel() == 0:
                    skipped_empty += 1
                    continue
                local_coords = coords_cpu[selected] - origin
                patches.append((local_coords, current_feats[selected]))
                patch_indices.append(selected)
                patch_weights.append(patch_blend_weights(local_coords, patch_size, blend_mode))
                if step_idx == 0:
                    total_patch_tokens += int(selected.numel())

            if not patches:
                continue

            sample = make_sparse_batch(patches, device)
            if cond_as_token:
                cond = torch.zeros((sample.shape[0], 1, 3), dtype=torch.float32, device=device)
            else:
                cond = torch.zeros((sample.shape[0], 3), dtype=torch.float32, device=device)

            with amp_context():
                out = sampler.sample_once(patch_model, sample, t_cur, t_prev, cond)
            pred_cpu = out.pred_x_prev.detach().cpu()

            for local_idx, selected in enumerate(patch_indices):
                next_feats = pred_cpu.feats[pred_cpu.layout[local_idx]].float()
                weight = patch_weights[local_idx]
                accum[selected] += next_feats * weight
                weights[selected] += weight
                coverage_counts[selected] += 1.0
                used_patch_steps += 1

        covered = weights.squeeze(1) > 0
        current_feats[covered] = accum[covered] / weights[covered]

    refined = current_feats.clamp(-1.0, 1.0)
    covered_once = coverage_counts.squeeze(1) > 0
    stats = {
        "requested_patches": len(origins),
        "used_patch_steps": used_patch_steps,
        "used_patches_per_step": used_patch_steps / max(steps, 1),
        "skipped_empty_patches": skipped_empty,
        "covered_voxels": int(covered_once.sum().item()),
        "total_voxels": int(feats_cpu.shape[0]),
        "total_patch_tokens": total_patch_tokens,
        "start_t": start_t,
        "steps": steps,
        "blend_mode": blend_mode,
    }
    return refined, coverage_counts.squeeze(1), stats


def render_views(coords: torch.Tensor, feats: torch.Tensor, resolution: int, attr_slice: slice, render_resolution: int, render_ssaa: int) -> np.ndarray:
    device = feats.device
    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = render_ssaa

    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / resolution,
        coords=coords.long().contiguous(),
        attrs=None,
        layout={"color": slice(0, 3)},
    )
    attr = ((feats[:, attr_slice].float() + 1.0) * 0.5).clamp(0, 1)
    if attr.shape[1] == 1:
        attr = attr.expand(-1, 3)
    elif attr.shape[1] > 3:
        attr = attr[:, :3]

    yaws = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    pitch = np.deg2rad(20.0)
    fov = torch.deg2rad(torch.tensor(30.0, device=device))
    image = torch.zeros(3, render_resolution * 2, render_resolution * 2, device=device)
    for view_idx, yaw in enumerate(yaws):
        eye = torch.tensor(
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)],
            device=device,
            dtype=torch.float32,
        ) * 2.0
        ext = utils3d.torch.extrinsics_look_at(
            eye,
            torch.tensor([0, 0, 0], device=device, dtype=torch.float32),
            torch.tensor([0, 0, 1], device=device, dtype=torch.float32),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        res = renderer.render(rep, ext, intr, colors_overwrite=attr)
        row = view_idx // 2
        col = view_idx % 2
        image[:, render_resolution * row:render_resolution * (row + 1), render_resolution * col:render_resolution * (col + 1)] = res["color"]

    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def make_comparison_sheet(panels: list[tuple[str, np.ndarray]], title: str) -> Image.Image:
    panel_h, panel_w = panels[0][1].shape[:2]
    header_h = 36
    label_h = 24
    canvas = Image.new("RGB", (panel_w * len(panels), header_h + panel_h + label_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)
    for col, (label, image) in enumerate(panels):
        x0 = col * panel_w
        canvas.paste(Image.fromarray(image, mode="RGB"), (x0, header_h))
        draw.text((x0 + 8, header_h + panel_h + 6), label, fill=(255, 255, 255), font=font)
    return canvas


def sparse_l1_metrics(
    gt_coords: torch.Tensor,
    gt_feats: torch.Tensor,
    pred_coords: torch.Tensor,
    pred_feats: torch.Tensor,
    background_value: float = -1.0,
) -> dict:
    gt_coords = gt_coords.detach().cpu().long()
    pred_coords = pred_coords.detach().cpu().long()
    gt_feats = gt_feats.detach().cpu().float()
    pred_feats = pred_feats.detach().cpu().float()

    if torch.equal(gt_coords, pred_coords):
        diff = torch.abs(gt_feats - pred_feats)
        edge_diff = diff[:, 0:3]
        vertex_diff = diff[:, 3:6]
        return {
            "coord_exact_match": True,
            "gt_voxels": int(gt_coords.shape[0]),
            "pred_voxels": int(pred_coords.shape[0]),
            "intersection_voxels": int(gt_coords.shape[0]),
            "gt_coverage": 1.0,
            "pred_coverage": 1.0,
            "l1": float(diff.mean().item()) if diff.numel() else None,
            "edge_l1": float(edge_diff.mean().item()) if edge_diff.numel() else None,
            "vertex_l1": float(vertex_diff.mean().item()) if vertex_diff.numel() else None,
            "union_l1_with_background": float(diff.mean().item()) if diff.numel() else None,
        }

    gt = {tuple(coord.tolist()): idx for idx, coord in enumerate(gt_coords)}
    pred = {tuple(coord.tolist()): idx for idx, coord in enumerate(pred_coords)}
    intersection = sorted(set(gt.keys()) & set(pred.keys()))
    union = sorted(set(gt.keys()) | set(pred.keys()))

    if intersection:
        gt_idx = torch.tensor([gt[key] for key in intersection], dtype=torch.long)
        pred_idx = torch.tensor([pred[key] for key in intersection], dtype=torch.long)
        diff = torch.abs(gt_feats[gt_idx] - pred_feats[pred_idx])
        edge_diff = diff[:, 0:3]
        vertex_diff = diff[:, 3:6]
        l1 = float(diff.mean().item())
        edge_l1 = float(edge_diff.mean().item())
        vertex_l1 = float(vertex_diff.mean().item())
    else:
        l1 = edge_l1 = vertex_l1 = None

    union_sum = 0.0
    union_count = 0
    bg = torch.full((gt_feats.shape[1],), background_value, dtype=torch.float32)
    for key in union:
        gt_feat = gt_feats[gt[key]] if key in gt else bg
        pred_feat = pred_feats[pred[key]] if key in pred else bg
        union_sum += torch.abs(gt_feat - pred_feat).sum().item()
        union_count += gt_feat.numel()

    return {
        "coord_exact_match": False,
        "gt_voxels": int(gt_coords.shape[0]),
        "pred_voxels": int(pred_coords.shape[0]),
        "intersection_voxels": len(intersection),
        "gt_coverage": len(intersection) / max(int(gt_coords.shape[0]), 1),
        "pred_coverage": len(intersection) / max(int(pred_coords.shape[0]), 1),
        "l1": l1,
        "edge_l1": edge_l1,
        "vertex_l1": vertex_l1,
        "union_l1_with_background": union_sum / union_count if union_count else None,
    }


def decode_two_latents(latent_dataset, gt_latent: sp.SparseTensor, pred_latent: sp.SparseTensor, cache_path: str):
    z = sp.sparse_cat([gt_latent, pred_latent], dim=0)
    decoded = latent_dataset.decode_latent(z.cuda(), cache_paths=[cache_path, cache_path])
    gt = decoded[0].detach().cpu()
    pred = decoded[1].detach().cpu()
    return gt, pred


def save_visualization(
    gt_coords: torch.Tensor,
    gt_feats: torch.Tensor,
    decoded_coords: torch.Tensor,
    decoded_feats: torch.Tensor,
    refined_coords: torch.Tensor,
    refined_feats: torch.Tensor,
    out_path: Path,
    resolution: int,
    render_resolution: int,
    render_ssaa: int,
):
    gt_coords_gpu = gt_coords.cuda()
    gt_feats_gpu = gt_feats.cuda()
    decoded_coords_gpu = decoded_coords.cuda()
    decoded_gpu = decoded_feats.cuda()
    refined_coords_gpu = refined_coords.cuda()
    refined_gpu = refined_feats.cuda()
    panels = [
        ("gt edge", render_views(gt_coords_gpu, gt_feats_gpu, resolution, slice(0, 3), render_resolution, render_ssaa)),
        ("decoded edge", render_views(decoded_coords_gpu, decoded_gpu, resolution, slice(0, 3), render_resolution, render_ssaa)),
        ("multidiff edge", render_views(refined_coords_gpu, refined_gpu, resolution, slice(0, 3), render_resolution, render_ssaa)),
        ("gt vertex", render_views(gt_coords_gpu, gt_feats_gpu, resolution, slice(3, 6), render_resolution, render_ssaa)),
        ("decoded vertex", render_views(decoded_coords_gpu, decoded_gpu, resolution, slice(3, 6), render_resolution, render_ssaa)),
        ("multidiff vertex", render_views(refined_coords_gpu, refined_gpu, resolution, slice(3, 6), render_resolution, render_ssaa)),
    ]
    make_comparison_sheet(panels, out_path.parent.name).save(out_path)


def save_npz(path: Path, coords: torch.Tensor, feats: torch.Tensor):
    np.savez_compressed(
        path,
        coords=coords.detach().cpu().numpy().astype(np.int32),
        feats=feats.detach().cpu().float().numpy().astype(np.float32),
    )


def save_vxz(path: Path, coords: torch.Tensor, feats: torch.Tensor):
    import o_voxel

    attrs = {
        "base_color": (((feats[:, 0:3].detach().cpu().float().clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8)),
        "emissive": (((feats[:, 3:6].detach().cpu().float().clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8)),
    }
    o_voxel.io.write_vxz(str(path), coords.detach().cpu().int(), attrs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    latent_run_dir = Path(args.latent_run_dir).resolve()
    patch_run_dir = Path(args.patch_run_dir).resolve()
    latent_config_path = Path(args.latent_config).resolve() if args.latent_config else latent_run_dir / "config.json"
    patch_config_path = Path(args.patch_config).resolve() if args.patch_config else patch_run_dir / "config.json"
    latent_config = json.load(open(latent_config_path, "r"))
    patch_config = json.load(open(patch_config_path, "r"))

    output_base = Path(args.output_dir).resolve() if args.output_dir else latent_run_dir / "post_decode_patch_multidiffusion"
    output_dir = unique_output_dir(output_base)
    print(f"Writing outputs to {output_dir}")

    root = Path(args.root).resolve()
    latent_dataset_args = copy.deepcopy(latent_config["dataset"]["args"])
    train_norm_path = root / "splits" / "train" / "gaussian_distance_latents" / args.gaussian_distance_latent_name / "normalization.json"
    if train_norm_path.exists():
        latent_dataset_args["gaussian_distance_slat_normalization_path"] = str(train_norm_path)
    latent_dataset_args["snapshot_render_resolution"] = args.render_resolution
    latent_data_dir = build_latent_data_dir(root, args.split, args.gaussian_distance_latent_name, args.michelangelo_latent_name)
    latent_dataset = getattr(datasets, latent_config["dataset"]["name"])(json.dumps(latent_data_dir), **latent_dataset_args)

    metadata_filter_info = None
    if args.metadata_filter_csv:
        metadata_filter_info = apply_metadata_filter(latent_dataset, args.metadata_filter_csv)
        print(f"Applied metadata filter: {metadata_filter_info['metadata_filter_original_size']} -> {len(latent_dataset)}")

    latent_model = build_model(latent_config)
    latent_step = find_ckpt_step(latent_run_dir, args.latent_ckpt)
    latent_ckpt_path = load_denoiser_checkpoint(latent_model, latent_run_dir, latent_step, args.latent_ema_rate, latent_model.device)
    latent_amp = make_amp_context(latent_config)

    patch_model = build_model(patch_config)
    patch_step = find_ckpt_step(patch_run_dir, args.patch_ckpt)
    patch_ckpt_path = load_denoiser_checkpoint(patch_model, patch_run_dir, patch_step, args.patch_ema_rate, patch_model.device)
    patch_amp = make_amp_context(patch_config)

    patch_size = int(patch_config["dataset"]["args"].get("patch_size", 32))
    resolution = int(latent_config["dataset"]["args"].get("resolution", 512))
    indices = select_indices(latent_dataset, args)
    if not indices:
        raise RuntimeError("No samples selected.")

    manifest = {
        "latent_run_dir": str(latent_run_dir),
        "latent_config": str(latent_config_path),
        "latent_checkpoint_step": latent_step,
        "latent_checkpoint_path": latent_ckpt_path,
        "patch_run_dir": str(patch_run_dir),
        "patch_config": str(patch_config_path),
        "patch_checkpoint_step": patch_step,
        "patch_checkpoint_path": patch_ckpt_path,
        "patch_start_t": args.patch_start_t,
        "patch_steps": args.patch_steps,
        "patch_size": patch_size,
        "patch_stride": args.patch_stride,
        "blend_mode": args.blend_mode,
        "resolution": resolution,
        "samples": [],
    }
    if metadata_filter_info is not None:
        manifest.update(metadata_filter_info)

    for dataset_idx in indices:
        root_info, sha256 = latent_dataset.instances[dataset_idx]
        sample_dir = output_dir / f"{dataset_idx:06d}_{sha256}"
        sample_dir.mkdir(parents=False, exist_ok=False)
        print(f"\nProcessing dataset index {dataset_idx}, sha256={sha256}")

        raw_item = latent_dataset[dataset_idx]
        batch = latent_dataset.collate_fn([raw_item])
        batch = recursive_to_device(batch, "cuda")

        pred_latent = sample_latent_flow(
            latent_model,
            latent_config,
            batch,
            args.latent_sampling_steps,
            args.latent_guidance_strength,
            latent_amp,
        )

        cache_path = batch["gaussian_distance_slat_cache_path"][0]
        gt_decoded_one, decoded_one = decode_two_latents(
            latent_dataset,
            batch["x_0"][0],
            pred_latent[0],
            cache_path,
        )
        gt_coords = gt_decoded_one.coords[:, 1:].long()
        gt_feats = gt_decoded_one.feats.float().clamp(-1.0, 1.0)
        coords = decoded_one.coords[:, 1:].long()
        feats = decoded_one.feats.float().clamp(-1.0, 1.0)

        origins = build_patch_origins(coords, resolution, patch_size, args.patch_stride, args.max_patches, args.seed)
        refined_feats, coverage_counts, patch_stats = multidiffusion_sparse_patch_refine(
            patch_model,
            patch_config,
            coords,
            feats,
            origins,
            patch_size,
            args.patch_start_t,
            args.patch_steps,
            args.patch_batch_size,
            args.blend_mode,
            patch_amp,
        )

        save_npz(sample_dir / "gt_decoded_512_gaussian_distance.npz", gt_coords, gt_feats)
        save_npz(sample_dir / "flow_decoded_512_gaussian_distance.npz", coords, feats)
        save_npz(sample_dir / "patch_multidiffusion_refined_512_gaussian_distance.npz", coords, refined_feats)
        np.savez_compressed(
            sample_dir / "patch_coverage.npz",
            coords=coords.numpy().astype(np.int32),
            counts=coverage_counts.numpy().astype(np.float32),
        )
        if args.save_vxz:
            save_vxz(sample_dir / "gt_decoded_512_gaussian_distance.vxz", gt_coords, gt_feats)
            save_vxz(sample_dir / "flow_decoded_512_gaussian_distance.vxz", coords, feats)
            save_vxz(sample_dir / "patch_multidiffusion_refined_512_gaussian_distance.vxz", coords, refined_feats)

        save_visualization(
            gt_coords,
            gt_feats,
            coords,
            feats,
            coords,
            refined_feats,
            sample_dir / "gt_vs_flow_decoded_vs_patch_multidiffusion_refined.png",
            resolution,
            args.render_resolution,
            args.render_ssaa,
        )

        gt_vs_flow = sparse_l1_metrics(gt_coords, gt_feats, coords, feats)
        gt_vs_refined = sparse_l1_metrics(gt_coords, gt_feats, coords, refined_feats)
        sample_metrics = {
            "dataset_index": dataset_idx,
            "sha256": sha256,
            "source_root": str(root_info),
            "cache_path": cache_path,
            "gt_num_voxels": int(gt_coords.shape[0]),
            "flow_decoded_num_voxels": int(coords.shape[0]),
            "patch_multidiffusion_refined_num_voxels": int(coords.shape[0]),
            "l1_gt_vs_flow_decoded": gt_vs_flow["l1"],
            "l1_gt_vs_patch_multidiffusion_refined": gt_vs_refined["l1"],
            "edge_l1_gt_vs_flow_decoded": gt_vs_flow["edge_l1"],
            "edge_l1_gt_vs_patch_multidiffusion_refined": gt_vs_refined["edge_l1"],
            "vertex_l1_gt_vs_flow_decoded": gt_vs_flow["vertex_l1"],
            "vertex_l1_gt_vs_patch_multidiffusion_refined": gt_vs_refined["vertex_l1"],
            "union_l1_gt_vs_flow_decoded": gt_vs_flow["union_l1_with_background"],
            "union_l1_gt_vs_patch_multidiffusion_refined": gt_vs_refined["union_l1_with_background"],
            "l1_improvement": (
                gt_vs_flow["l1"] - gt_vs_refined["l1"]
                if gt_vs_flow["l1"] is not None and gt_vs_refined["l1"] is not None
                else None
            ),
            "feature_l1_patch_delta": float(torch.mean(torch.abs(refined_feats - feats)).item()) if feats.numel() else None,
            "feature_l2_patch_delta": float(torch.mean((refined_feats - feats).pow(2)).item()) if feats.numel() else None,
            "gt_vs_flow_decoded": gt_vs_flow,
            "gt_vs_patch_multidiffusion_refined": gt_vs_refined,
            "coverage": patch_stats,
            "sample_dir": str(sample_dir),
        }
        with open(sample_dir / "metrics.json", "w") as fp:
            json.dump(sample_metrics, fp, indent=2)
        manifest["samples"].append(sample_metrics)
        print(f"Saved {sample_dir}")

    with open(output_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nSaved manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
