import argparse
import glob
import json
import os
from pathlib import Path

from easydict import EasyDict as edict
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from trellis2 import models
from trellis2.datasets.sparse_voxel_triangle_field import find_triangle_field_path, load_triangle_field_npz
from trellis2.modules import sparse as sp
from trellis2.renderers import VoxelRenderer
from trellis2.representations import Voxel
from trellis2.utils.render_utils import snapshot_orbit_cameras


CHANNELS = {"d_tri": 0, "d_vert": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a triangle-field VAE on parent-bin averaged fields "
            "downsampled from 512 to lower resolutions."
        )
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--vae_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--source_resolution", type=int, default=512)
    parser.add_argument("--resolutions", type=str, default="512,256,128,64,32")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--num_metric_samples", type=int, default=64)
    parser.add_argument("--num_visual_samples", type=int, default=4)
    parser.add_argument("--max_source_voxels", type=int, default=1000000)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--render_resolution", type=int, default=320)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--d_tri_colormap", choices=("magma", "gray"), default="magma")
    parser.add_argument("--d_vert_colormap", choices=("magma", "gray"), default="gray")
    parser.add_argument(
        "--flex_gemm_algo",
        type=str,
        default=None,
        choices=(
            "explicit_gemm",
            "implicit_gemm",
            "implicit_gemm_splitk",
            "masked_implicit_gemm",
            "masked_implicit_gemm_splitk",
        ),
        help="Optional eval-only FlexGEMM sparse-conv algorithm override.",
    )
    return parser.parse_args()


def find_ckpt_step(vae_dir: Path, ckpt: str) -> str:
    if ckpt != "latest":
        return ckpt
    ckpts = sorted(glob.glob(str(vae_dir / "ckpts" / "encoder_step*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No encoder_step*.pt checkpoints found in {vae_dir / 'ckpts'}")
    return Path(ckpts[-1]).stem.replace("encoder_", "")


def load_instances(root: Path, split: str, instances: str | None, random: bool, seed: int) -> list[str]:
    if instances is None:
        candidates = [
            root / "splits" / f"{split}_triangle_field_512" / "instances.txt",
            root / "splits" / split / "instances.txt",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No instances.txt found for split {split} under {root / 'splits'}")
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif "," in instances:
        values = [item.strip() for item in instances.split(",") if item.strip()]
    else:
        path = Path(instances)
        if path.exists():
            values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        else:
            values = [instances.strip()]
    if random:
        rng = np.random.default_rng(seed)
        values = list(rng.permutation(values))
    return values


def load_raw_triangle_field(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with load_triangle_field_npz(str(path)) as data:
        coords = torch.from_numpy(data["coords"].astype(np.int32, copy=False))
        features = torch.from_numpy(data["features"].astype(np.float32, copy=False))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} has invalid coords shape {tuple(coords.shape)}")
    if features.ndim != 2 or features.shape[0] != coords.shape[0]:
        raise ValueError(f"{path} has invalid features shape {tuple(features.shape)}")
    return coords, features


def average_downsample(coords: torch.Tensor, features: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    if factor == 1:
        return coords, features.float()
    parent = torch.div(coords, factor, rounding_mode="floor").int()
    unique_parent, inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    sums = torch.zeros((unique_parent.shape[0], features.shape[1]), dtype=torch.float32)
    counts = torch.zeros((unique_parent.shape[0], 1), dtype=torch.float32)
    sums.index_add_(0, inverse, features.float())
    counts.index_add_(0, inverse, torch.ones((features.shape[0], 1), dtype=torch.float32))
    return unique_parent, sums / counts.clamp_min(1.0)


def transform_for_model(features: torch.Tensor, distance_transform: str) -> torch.Tensor:
    if distance_transform == "none":
        return features.float()
    if distance_transform == "minus_one_one":
        out = features.float().clone()
        out[:, :2] = out[:, :2] * 2.0 - 1.0
        return out
    raise ValueError(f"Unsupported distance_transform: {distance_transform}")


def inverse_distance_transform(values: torch.Tensor, distance_transform: str) -> torch.Tensor:
    if distance_transform == "none":
        return values.float()
    if distance_transform == "minus_one_one":
        out = values.float().clone()
        out[:, :2] = out[:, :2] * 0.5 + 0.5
        return out
    raise ValueError(f"Unsupported distance_transform: {distance_transform}")


def make_sparse(coords: torch.Tensor, features: torch.Tensor) -> sp.SparseTensor:
    sparse_coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1).int().cuda()
    return sp.SparseTensor(features.float().cuda(), sparse_coords)


def load_model(config: dict, key: str, ckpt_path: Path):
    model_cfg = config["models"][key]
    model = getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda().eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=False)
    return model


def reconstruct(
    encoder,
    decoder,
    coords: torch.Tensor,
    raw_features: torch.Tensor,
    distance_transform: str,
) -> tuple[torch.Tensor, sp.SparseTensor]:
    model_features = transform_for_model(raw_features, distance_transform)
    x = make_sparse(coords, model_features)
    z = encoder(x, sample_posterior=False)
    y = decoder(z)
    if not torch.equal(y.coords, x.coords):
        raise RuntimeError("Decoder output support differed from averaged input support.")
    pred = inverse_distance_transform(y.feats[:, :2], distance_transform).clamp(0.0, 1.0)
    target = raw_features[:, :2].float().cuda()
    return pred, sp.SparseTensor(pred, y.coords)


def update_metrics(acc: dict, pred: torch.Tensor, target: torch.Tensor):
    err = pred.float() - target.float().cuda()
    acc["l1_sum"] += err.abs().sum(dim=0).detach().cpu().double()
    acc["l2_sum"] += err.square().sum(dim=0).detach().cpu().double()
    acc["tokens"] += int(err.shape[0])


def scalar_to_gray(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1, 1).float().clamp(0, 1)
    return values.expand(-1, 3)


def scalar_to_magma(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1).float().clamp(0, 1)
    stops = torch.tensor(
        [
            [0.001, 0.000, 0.014],
            [0.251, 0.066, 0.430],
            [0.478, 0.125, 0.514],
            [0.741, 0.214, 0.329],
            [0.944, 0.498, 0.145],
            [0.987, 0.991, 0.749],
        ],
        dtype=torch.float32,
        device=values.device,
    )
    scaled = values * (stops.shape[0] - 1)
    idx0 = torch.floor(scaled).long().clamp(0, stops.shape[0] - 1)
    idx1 = (idx0 + 1).clamp(0, stops.shape[0] - 1)
    t = (scaled - idx0.float()).reshape(-1, 1)
    return stops[idx0] * (1.0 - t) + stops[idx1] * t


def colorize(values: torch.Tensor, colormap: str) -> torch.Tensor:
    if colormap == "gray":
        return scalar_to_gray(values)
    if colormap == "magma":
        return scalar_to_magma(values)
    raise ValueError(f"Unsupported colormap: {colormap}")


@torch.no_grad()
def render_channel(
    coords: torch.Tensor,
    features: torch.Tensor,
    resolution: int,
    channel: int,
    colormap: str,
    render_resolution: int,
    ssaa: int,
) -> Image.Image:
    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.ssaa = ssaa
    exts, ints = snapshot_orbit_cameras()

    sparse = make_sparse(coords, features)
    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / resolution,
        coords=sparse.coords[:, 1:].contiguous(),
        attrs=None,
        layout={"color": slice(0, 3)},
    )
    attr = colorize(sparse.feats[:, channel], colormap).float()
    image = torch.zeros(3, render_resolution * 2, render_resolution * 2, dtype=torch.float32, device="cuda")
    for view_idx, (ext, intr) in enumerate(zip(exts, ints)):
        with torch.autocast(device_type="cuda", enabled=False):
            out = renderer.render(rep, ext.float(), intr.float(), colors_overwrite=attr)
        row = view_idx // 2
        col = view_idx % 2
        image[
            :,
            render_resolution * row:render_resolution * (row + 1),
            render_resolution * col:render_resolution * (col + 1),
        ] = out["color"].float()
    arr = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_side_by_side(images: list[Image.Image], labels: list[str], title: str, out_path: Path) -> None:
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    label_h = 42
    title_h = 34
    pad = 8
    canvas = Image.new("RGB", (sum(widths) + pad * (len(images) + 1), max(heights) + label_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(0, 0, 0))
    x = pad
    for image, label in zip(images, labels):
        draw.text((x, title_h + 10), label, fill=(0, 0, 0))
        canvas.paste(image, (x, title_h + label_h))
        x += image.width + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def finalize_metrics(acc: dict) -> dict:
    out = {}
    for resolution, values in acc.items():
        tokens = max(values["tokens"], 1)
        l1 = values["l1_sum"] / tokens
        rmse = torch.sqrt(values["l2_sum"] / tokens)
        out[str(resolution)] = {
            "tokens": int(values["tokens"]),
            "l1": {"d_tri": float(l1[0]), "d_vert": float(l1[1])},
            "l1_mean": float(l1.mean()),
            "rmse": {"d_tri": float(rmse[0]), "d_vert": float(rmse[1])},
            "rmse_mean": float(rmse.mean()),
        }
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root)
    vae_dir = Path(args.vae_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if args.flex_gemm_algo is not None:
        from trellis2.modules.sparse.conv import config as sparse_conv_config

        sparse_conv_config.FLEX_GEMM_ALGO = args.flex_gemm_algo
        print(f"Using eval-only FlexGEMM algorithm override: {args.flex_gemm_algo}")

    resolutions = [int(item) for item in args.resolutions.split(",") if item.strip()]
    if not resolutions:
        raise ValueError("--resolutions must contain at least one resolution.")
    for resolution in resolutions:
        if args.source_resolution % resolution != 0:
            raise ValueError(f"Source resolution {args.source_resolution} is not divisible by {resolution}.")
        if resolution > args.source_resolution:
            raise ValueError(f"Requested resolution {resolution} exceeds source resolution {args.source_resolution}.")

    config = json.load(open(vae_dir / "config.json", "r"))
    distance_transform = config["dataset"]["args"].get("distance_transform", "none")
    ckpt = find_ckpt_step(vae_dir, args.ckpt)
    encoder = load_model(config, "encoder", vae_dir / "ckpts" / f"encoder_{ckpt}.pt")
    decoder = load_model(config, "decoder", vae_dir / "ckpts" / f"decoder_{ckpt}.pt")

    source_root = root / "splits" / args.split / f"triangle_field_voxels_{args.source_resolution}"
    instances = load_instances(root, args.split, args.instances, args.random, args.seed)
    acc = {
        resolution: {
            "l1_sum": torch.zeros(2, dtype=torch.float64),
            "l2_sum": torch.zeros(2, dtype=torch.float64),
            "tokens": 0,
        }
        for resolution in resolutions
    }
    kept = []
    visualized = 0
    skipped_too_large = 0

    with torch.no_grad():
        for instance in tqdm(instances, desc="Evaluating averaged downsample recon"):
            try:
                path = Path(find_triangle_field_path(str(source_root), instance))
            except FileNotFoundError:
                continue

            coords, features = load_raw_triangle_field(path)
            if args.max_source_voxels is not None and coords.shape[0] > args.max_source_voxels:
                skipped_too_large += 1
                continue
            per_resolution = {}
            for resolution in resolutions:
                factor = args.source_resolution // resolution
                ds_coords, ds_features = average_downsample(coords, features, factor)
                pred, pred_sparse = reconstruct(encoder, decoder, ds_coords, ds_features, distance_transform)
                target = ds_features[:, :2]
                update_metrics(acc[resolution], pred, target)
                per_resolution[resolution] = {
                    "coords": ds_coords,
                    "target": target,
                    "pred": pred_sparse.feats.detach().cpu(),
                }

            if visualized < args.num_visual_samples:
                for channel_name, channel_idx in CHANNELS.items():
                    colormap = args.d_tri_colormap if channel_name == "d_tri" else args.d_vert_colormap
                    images = []
                    labels = []
                    for resolution in resolutions:
                        item = per_resolution[resolution]
                        images.append(
                            render_channel(
                                item["coords"],
                                item["target"],
                                resolution,
                                channel_idx,
                                colormap,
                                args.render_resolution,
                                args.ssaa,
                            )
                        )
                        labels.append(f"{resolution} avg")
                        images.append(
                            render_channel(
                                item["coords"],
                                item["pred"],
                                resolution,
                                channel_idx,
                                colormap,
                                args.render_resolution,
                                args.ssaa,
                            )
                        )
                        labels.append(f"{resolution} rec")
                    make_side_by_side(
                        images,
                        labels,
                        f"{instance} {channel_name}: averaged target vs VAE reconstruction",
                        vis_dir / f"{instance}_{channel_name}_avg_vs_rec.jpg",
                    )
                visualized += 1

            kept.append(instance)
            if len(kept) >= args.num_metric_samples:
                break

    if not kept:
        raise RuntimeError(f"No {args.source_resolution} triangle-field voxels found under {source_root}.")

    summary = {
        "root": str(root),
        "split": args.split,
        "vae_dir": str(vae_dir),
        "ckpt": ckpt,
        "source_resolution": args.source_resolution,
        "resolutions": resolutions,
        "num_metric_samples": len(kept),
        "num_visual_samples": visualized,
        "skipped_too_large": skipped_too_large,
        "max_source_voxels": args.max_source_voxels,
        "distance_transform": distance_transform,
        "metrics": finalize_metrics(acc),
        "instances": kept,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary and visualizations to {output_dir}")


if __name__ == "__main__":
    main()
