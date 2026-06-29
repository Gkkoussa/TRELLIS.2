import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Compose triangle-field method comparison sheets.")
    parser.add_argument("--dit_dir", type=str, required=True)
    parser.add_argument("--sr_cascade_dir", type=str, required=True)
    parser.add_argument("--sr_repeat_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match {root / pattern}")
    if len(matches) > 1:
        print(f"Warning: multiple files match {root / pattern}; using {matches[0]}")
    return matches[0]


def load_grid(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_single_view(grid: Image.Image, index: int, nrow: int = 4) -> Image.Image:
    tile_w = grid.width // nrow
    tile_h = grid.height // nrow
    col = index % nrow
    row = index // nrow
    tile = grid.crop((col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h))
    return tile.crop((0, 0, tile_w // 2, tile_h // 2))


def draw_label(draw: ImageDraw.ImageDraw, xy, text: str):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def make_sheet(columns, labels, instances, out_path: Path):
    n = len(instances)
    views = [[crop_single_view(grid, i) for grid in columns] for i in range(n)]
    cell_w = max(view.width for row in views for view in row)
    cell_h = max(view.height for row in views for view in row)
    label_h = 38
    id_w = 92
    pad = 8
    canvas = Image.new(
        "RGB",
        (id_w + len(columns) * (cell_w + pad) + pad, label_h + n * (cell_h + pad) + pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for j, label in enumerate(labels):
        draw_label(draw, (id_w + pad + j * (cell_w + pad), 8), label)
    for i, instance in enumerate(instances):
        y = label_h + i * (cell_h + pad)
        draw_label(draw, (8, y + 8), instance[:8])
        for j, view in enumerate(views[i]):
            x = id_w + pad + j * (cell_w + pad)
            canvas.paste(view.resize((cell_w, cell_h)), (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def load_instances(*summary_paths: Path, num_samples: int):
    selected = None
    for path in summary_paths:
        if not path.exists():
            continue
        with open(path, "r") as f:
            summary = json.load(f)
        values = summary.get("selected_instances") or summary.get("instances")
        if values is None:
            continue
        values = [str(v) for v in values[:num_samples]]
        if selected is None:
            selected = values
        elif values != selected:
            raise ValueError(f"Selected instances differ in {path}")
    if selected is None:
        selected = [f"{i:02d}" for i in range(num_samples)]
    return selected[:num_samples]


def main():
    args = parse_args()
    dit_dir = Path(args.dit_dir)
    sr_cascade_dir = Path(args.sr_cascade_dir)
    sr_repeat_dir = Path(args.sr_repeat_dir)
    output_dir = Path(args.output_dir)

    dit_sample_dir = next((dit_dir / "samples").glob("*"))
    instances = load_instances(
        dit_dir / "metrics.json",
        sr_cascade_dir / "summary.json",
        sr_repeat_dir / "summary.json",
        num_samples=args.num_samples,
    )

    for channel in ("d_tri", "d_vert"):
        dit_grid = load_grid(find_one(dit_sample_dir, f"sample_{channel}_*.jpg"))
        sr_256 = load_grid(find_one(sr_cascade_dir, f"sample_256_{channel}_*.jpg"))
        sr_512 = load_grid(find_one(sr_cascade_dir, f"sample_512_{channel}_*.jpg"))
        repeat_256 = load_grid(find_one(sr_repeat_dir, f"stage128to256_iter3_sample_{channel}_*.jpg"))
        repeat_512 = load_grid(find_one(sr_repeat_dir, f"stage256to512_iter3_sample_{channel}_*.jpg"))

        make_sheet(
            [dit_grid, sr_256, repeat_256],
            ["DiT 256", "SR cascade 256", "SR repeat3 256"],
            instances,
            output_dir / f"comparison_256_{channel}.jpg",
        )
        make_sheet(
            [dit_grid, sr_512, repeat_512],
            ["DiT 256", "SR cascade 512", "SR repeat3 512"],
            instances,
            output_dir / f"comparison_dit256_sr512_{channel}.jpg",
        )

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(
            {
                "dit_dir": str(dit_dir),
                "sr_cascade_dir": str(sr_cascade_dir),
                "sr_repeat_dir": str(sr_repeat_dir),
                "instances": instances,
                "outputs": [
                    "comparison_256_d_tri.jpg",
                    "comparison_256_d_vert.jpg",
                    "comparison_dit256_sr512_d_tri.jpg",
                    "comparison_dit256_sr512_d_vert.jpg",
                ],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
