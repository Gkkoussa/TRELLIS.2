import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Compose one cascade progression sheet per mesh.")
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--channel", default="d_tri", choices=("d_tri", "d_vert"))
    parser.add_argument("--panel_size", type=int, default=384)
    return parser.parse_args()


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def crop_first_view(grid, index, count):
    nrow = max(1, int(math.sqrt(count)))
    padding = 2
    tile_size = (grid.width - padding * (nrow + 1)) // nrow
    col, row = index % nrow, index // nrow
    left = padding + col * (tile_size + padding)
    top = padding + row * (tile_size + padding)
    tile = grid.crop((left, top, left + tile_size, top + tile_size))
    return tile.crop((0, 0, tile_size // 2, tile_size // 2))


def find_render(eval_dir, low, high, iteration, channel):
    matches = list(eval_dir.glob(f"stage{low}to{high}_iter{iteration}_sample_{channel}_*.jpg"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one stage {low}->{high}, iteration {iteration}, {channel} render; "
            f"found {len(matches)}"
        )
    return Image.open(matches[0]).convert("RGB")


def main():
    args = parse_args()
    summary = json.loads((args.eval_dir / "summary.json").read_text())
    meshes = summary["meshes"]
    stages = [tuple(stage) for stage in summary["stages"]]
    repeats = int(summary["stage_repeats"])
    output_dir = args.output_dir or args.eval_dir / f"progression_{args.channel}_first_view"
    output_dir.mkdir(parents=True, exist_ok=True)

    renders = {
        (low, high, iteration): find_render(
            args.eval_dir, low, high, iteration, args.channel
        )
        for low, high in stages
        for iteration in range(1, repeats + 1)
    }
    title_font = load_font(26)
    label_font = load_font(22)
    pad, title_h, header_h, row_label_w = 10, 48, 40, 150

    for mesh_index, mesh in enumerate(meshes):
        width = row_label_w + repeats * (args.panel_size + pad) + pad
        height = title_h + header_h + len(stages) * (args.panel_size + pad) + pad
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        name = Path(mesh["path"]).stem
        draw.text((pad, 8), name, fill="black", font=title_font)
        for iteration in range(1, repeats + 1):
            x = row_label_w + (iteration - 1) * (args.panel_size + pad)
            draw.text((x + pad, title_h + 5), f"Iteration {iteration}", fill="black", font=label_font)
        for row, (low, high) in enumerate(stages):
            y = title_h + header_h + row * (args.panel_size + pad)
            draw.text((pad, y + 10), f"{low} -> {high}", fill="black", font=label_font)
            for iteration in range(1, repeats + 1):
                view = crop_first_view(
                    renders[(low, high, iteration)], mesh_index, len(meshes)
                ).resize((args.panel_size, args.panel_size), Image.Resampling.LANCZOS)
                x = row_label_w + (iteration - 1) * (args.panel_size + pad)
                sheet.paste(view, (x, y))
        sheet.save(output_dir / f"{mesh_index:02d}_{name}_progression.jpg", quality=95)

    print(f"Saved {len(meshes)} progression sheets to {output_dir}")


if __name__ == "__main__":
    main()
