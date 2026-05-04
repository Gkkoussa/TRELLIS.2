import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import o_voxel
import utils3d


PANEL_ORDER = [
    ("edge_rgb", "Edge RGB"),
    ("edge_ch0", "Edge Ch0"),
    ("edge_ch1", "Edge Ch1"),
    ("edge_ch2", "Edge Ch2"),
    ("vertex_rgb", "Vertex RGB"),
    ("vertex_ch0", "Vertex Ch0"),
    ("vertex_ch1", "Vertex Ch1"),
    ("vertex_ch2", "Vertex Ch2"),
]


def build_camera(device: torch.device):
    extr = utils3d.extrinsics_look_at(
        eye=torch.tensor([1.2, 0.5, 1.2]),
        look_at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
    ).to(device)
    intr = utils3d.intrinsics_from_fov_xy(
        fov_x=torch.deg2rad(torch.tensor(45.0)),
        fov_y=torch.deg2rad(torch.tensor(45.0)),
    ).to(device)
    return extr, intr


def render_attr(
    renderer,
    position: torch.Tensor,
    attrs: torch.Tensor,
    voxel_size: float,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> np.ndarray:
    output = renderer.render(
        position=position,
        attrs=attrs,
        voxel_size=voxel_size,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    image = output.attr.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def grayscale_channel(attr: torch.Tensor, channel_idx: int) -> torch.Tensor:
    channel = attr[:, channel_idx:channel_idx + 1]
    return channel.expand(-1, 3)


def ensure_rgb_attr(data: dict, key: str) -> torch.Tensor:
    if key not in data:
        raise ValueError(f"vxz has no '{key}' attribute")
    if data[key].ndim != 2 or data[key].shape[1] < 3:
        raise ValueError(f"vxz attribute '{key}' must have at least 3 channels")
    return data[key][:, :3].float() / 255.0


def make_contact_sheet(images: dict, panel_resolution: int) -> np.ndarray:
    cols = 4
    rows = 2
    label_height = 28
    canvas = Image.new(
        "RGB",
        (cols * panel_resolution, rows * (panel_resolution + label_height)),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (key, label) in enumerate(PANEL_ORDER):
        row = idx // cols
        col = idx % cols
        x0 = col * panel_resolution
        y0 = row * (panel_resolution + label_height)
        panel = Image.fromarray(images[key], mode="RGB")
        canvas.paste(panel, (x0, y0))
        draw.rectangle(
            [x0, y0 + panel_resolution, x0 + panel_resolution, y0 + panel_resolution + label_height],
            fill=(20, 20, 20),
        )
        draw.text((x0 + 8, y0 + panel_resolution + 7), label, fill=(255, 255, 255), font=font)

    return np.array(canvas)


def render_file(
    vxz_path: Path,
    output_path: Path,
    grid_size: int,
    resolution: int,
    ssaa: int,
    num_threads: int,
):
    coords, data = o_voxel.io.read_vxz(str(vxz_path), num_threads=num_threads)
    if coords.numel() == 0:
        raise ValueError("empty vxz")

    edge_rgb = ensure_rgb_attr(data, "base_color")
    vertex_rgb = ensure_rgb_attr(data, "emissive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    position = (coords.float() / grid_size - 0.5).to(device)
    edge_rgb = edge_rgb.to(device)
    vertex_rgb = vertex_rgb.to(device)

    extrinsics, intrinsics = build_camera(device)
    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": resolution, "ssaa": ssaa}
    )
    voxel_size = 1.0 / grid_size

    images = {
        "edge_rgb": render_attr(renderer, position, edge_rgb, voxel_size, extrinsics, intrinsics),
        "edge_ch0": render_attr(renderer, position, grayscale_channel(edge_rgb, 0), voxel_size, extrinsics, intrinsics),
        "edge_ch1": render_attr(renderer, position, grayscale_channel(edge_rgb, 1), voxel_size, extrinsics, intrinsics),
        "edge_ch2": render_attr(renderer, position, grayscale_channel(edge_rgb, 2), voxel_size, extrinsics, intrinsics),
        "vertex_rgb": render_attr(renderer, position, vertex_rgb, voxel_size, extrinsics, intrinsics),
        "vertex_ch0": render_attr(renderer, position, grayscale_channel(vertex_rgb, 0), voxel_size, extrinsics, intrinsics),
        "vertex_ch1": render_attr(renderer, position, grayscale_channel(vertex_rgb, 1), voxel_size, extrinsics, intrinsics),
        "vertex_ch2": render_attr(renderer, position, grayscale_channel(vertex_rgb, 2), voxel_size, extrinsics, intrinsics),
    }

    sheet = make_contact_sheet(images, resolution)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, sheet)


def collect_vxz_files(input_path: Path, max_num: int | None):
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.vxz"))
    if max_num is not None:
        files = files[:max_num]
    return files


def output_path_for(vxz_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    if input_root.is_file():
        return output_root / f"{vxz_path.stem}{suffix}"
    rel = vxz_path.relative_to(input_root)
    out_path = (output_root / rel).with_suffix("")
    return out_path.parent / f"{out_path.name}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="A .vxz file or directory to scan recursively")
    parser.add_argument("--output_dir", type=Path, default=None, help="Where to save rendered PNGs")
    parser.add_argument("--grid_size", type=int, required=True, help="Voxel grid size used to generate the .vxz files, e.g. 256")
    parser.add_argument("--resolution", type=int, default=512, help="Per-panel output image resolution")
    parser.add_argument("--ssaa", type=int, default=2, help="Supersampling factor")
    parser.add_argument("--num_threads", type=int, default=4, help="Reader threads for read_vxz")
    parser.add_argument("--max_num", type=int, default=None, help="Maximum number of input voxelized meshes to visualize")
    parser.add_argument("--suffix", type=str, default="_channels.png", help="Suffix for rendered images")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    if args.max_num is not None and args.max_num <= 0:
        raise SystemExit("--max_num must be positive when provided")

    default_output_dir = input_path.parent / "channel_renders" if input_path.is_file() else input_path / "channel_renders"
    output_root = args.output_dir.resolve() if args.output_dir is not None else default_output_dir
    files = collect_vxz_files(input_path, args.max_num)
    if not files:
        raise SystemExit(f"No .vxz files found under {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Found {len(files)} file(s)")
    print(f"Using device: {device}")
    print(f"Writing renders to: {output_root}")

    errors = []
    for vxz_path in tqdm(files, desc="Rendering VXZ channels"):
        out_path = output_path_for(vxz_path, input_path, output_root, args.suffix)
        try:
            render_file(vxz_path, out_path, args.grid_size, args.resolution, args.ssaa, args.num_threads)
        except Exception as e:
            errors.append((str(vxz_path), str(e)))

    if errors:
        print(f"Completed with {len(errors)} errors:")
        for path, err in errors[:20]:
            print(f"  {path}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("Completed successfully")


if __name__ == "__main__":
    main()
