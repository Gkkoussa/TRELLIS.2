#!/usr/bin/env python3
"""Render original/QEM meshes with TRELLIS.2's training-time GPU rasterizer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trellis2.renderers import MeshRenderer
from trellis2.representations import Mesh
from trellis2.utils.render_utils import snapshot_orbit_cameras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing resolution_<R>/{original,collapsed}.obj.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--output-subdir", default="training_rasterizer")
    return parser.parse_args()


def load_obj(path: Path) -> Mesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == "f":
                polygon = [int(x.split("/")[0]) - 1 for x in fields[1:]]
                for idx in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[idx], polygon[idx + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ has no renderable triangles: {path}")
    return Mesh(
        vertices=torch.tensor(np.asarray(vertices), device="cuda", dtype=torch.float32),
        faces=torch.tensor(np.asarray(faces), device="cuda", dtype=torch.int32),
    )


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    return Image.fromarray(array.transpose(1, 2, 0), mode="RGB")


def render_training_sheet(
    renderer: MeshRenderer,
    mesh: Mesh,
    extrinsics: list[torch.Tensor],
    intrinsics: list[torch.Tensor],
    image_resolution: int,
) -> Image.Image:
    # This is the same 2x2 normal-image layout used by
    # FlexiDualGridVisMixin.visualize_sample during training/evaluation.
    sheet = Image.new("RGB", (2 * image_resolution, 2 * image_resolution))
    for view_idx, (extrinsic, intrinsic) in enumerate(zip(extrinsics, intrinsics)):
        normal = renderer.render(
            mesh, extrinsic, intrinsic, return_types=["normal"]
        )["normal"]
        sheet.paste(
            tensor_to_image(normal),
            (
                (view_idx % 2) * image_resolution,
                (view_idx // 2) * image_resolution,
            ),
        )
    return sheet


def labeled_comparison(
    original: Image.Image, collapsed: Image.Image, resolution: int
) -> Image.Image:
    header_height = 44
    panel = Image.new(
        "RGB",
        (original.width + collapsed.width, header_height + original.height),
        "white",
    )
    panel.paste(original, (0, header_height))
    panel.paste(collapsed, (original.width, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 13), f"R={resolution}  ORIGINAL", fill="black")
    draw.text((original.width + 12, 13), f"R={resolution}  QEM COLLAPSED", fill="black")
    return panel


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS MeshRenderer/nvdiffrast requires a CUDA GPU.")

    renderer = MeshRenderer({"near": 1, "far": 3})
    renderer.rendering_options.resolution = args.image_resolution
    renderer.rendering_options.ssaa = args.ssaa
    extrinsics, intrinsics = snapshot_orbit_cameras()

    output_dir = args.input_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    collapsed_across_resolutions: list[tuple[int, Image.Image]] = []

    for resolution in args.resolutions:
        resolution_dir = args.input_dir / f"resolution_{resolution}"
        original_path = resolution_dir / "original.obj"
        collapsed_path = resolution_dir / "collapsed.obj"
        if not original_path.is_file() or not collapsed_path.is_file():
            raise FileNotFoundError(
                f"Expected both {original_path} and {collapsed_path}"
            )

        original_sheet = render_training_sheet(
            renderer,
            load_obj(original_path),
            extrinsics,
            intrinsics,
            args.image_resolution,
        )
        collapsed_sheet = render_training_sheet(
            renderer,
            load_obj(collapsed_path),
            extrinsics,
            intrinsics,
            args.image_resolution,
        )

        per_resolution_dir = output_dir / f"resolution_{resolution}"
        per_resolution_dir.mkdir(parents=True, exist_ok=True)
        original_sheet.save(per_resolution_dir / "original_training_normal.png")
        collapsed_sheet.save(per_resolution_dir / "collapsed_training_normal.png")
        comparison = labeled_comparison(original_sheet, collapsed_sheet, resolution)
        comparison.save(per_resolution_dir / "original_vs_collapsed_training_normal.png")
        collapsed_across_resolutions.append((resolution, collapsed_sheet))
        print(f"Rendered resolution {resolution}: {per_resolution_dir}", flush=True)

    thumb_size = args.image_resolution
    overview = Image.new(
        "RGB",
        (len(collapsed_across_resolutions) * thumb_size, thumb_size + 36),
        "white",
    )
    draw = ImageDraw.Draw(overview)
    for column, (resolution, sheet) in enumerate(collapsed_across_resolutions):
        overview.paste(
            sheet.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS),
            (column * thumb_size, 36),
        )
        draw.text((column * thumb_size + 10, 10), f"QEM R={resolution}", fill="black")
    overview.save(output_dir / "collapsed_all_resolutions_training_normal.png")
    print(f"All rasterizer outputs saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
