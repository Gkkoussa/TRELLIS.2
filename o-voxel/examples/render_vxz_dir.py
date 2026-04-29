import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm

import o_voxel
import utils3d


def render_file(vxz_path: Path, output_path: Path, grid_size: int, resolution: int, ssaa: int, num_threads: int):
    coords, data = o_voxel.io.read_vxz(str(vxz_path), num_threads=num_threads)
    if coords.numel() == 0:
        raise ValueError('empty vxz')
    if 'base_color' not in data:
        raise ValueError('vxz has no base_color attribute')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    position = (coords.float() / grid_size - 0.5).to(device)
    base_color = (data['base_color'].float() / 255.0).to(device)

    extr = utils3d.extrinsics_look_at(
        eye=torch.tensor([1.2, 0.5, 1.2]),
        look_at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0])
    ).to(device)
    intr = utils3d.intrinsics_from_fov_xy(
        fov_x=torch.deg2rad(torch.tensor(45.0)),
        fov_y=torch.deg2rad(torch.tensor(45.0)),
    ).to(device)

    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={'resolution': resolution, 'ssaa': ssaa}
    )
    output = renderer.render(
        position=position,
        attrs=base_color,
        voxel_size=1.0 / grid_size,
        extrinsics=extr,
        intrinsics=intr,
    )
    image = np.clip(output.attr.permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='Directory to scan recursively for .vxz files')
    parser.add_argument('--output_dir', type=Path, default=None, help='Where to save rendered PNGs')
    parser.add_argument('--grid_size', type=int, required=True, help='Voxel grid size used to generate the .vxz files, e.g. 256')
    parser.add_argument('--resolution', type=int, default=512, help='Output image resolution')
    parser.add_argument('--ssaa', type=int, default=2, help='Supersampling factor')
    parser.add_argument('--num_threads', type=int, default=4, help='Reader threads for read_vxz')
    parser.add_argument('--suffix', type=str, default='_render.png', help='Suffix for rendered images')
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output_dir.resolve() if args.output_dir is not None else input_root / 'renders'
    files = sorted(input_root.rglob('*.vxz'))
    if not files:
        raise SystemExit(f'No .vxz files found under {input_root}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Found {len(files)} files')
    print(f'Using device: {device}')
    print(f'Writing renders to: {output_root}')

    errors = []
    for vxz_path in tqdm(files, desc='Rendering VXZ'):
        rel = vxz_path.relative_to(input_root)
        out_path = (output_root / rel).with_suffix('')
        out_path = out_path.parent / f'{out_path.name}{args.suffix}'
        try:
            render_file(vxz_path, out_path, args.grid_size, args.resolution, args.ssaa, args.num_threads)
        except Exception as e:
            errors.append((str(vxz_path), str(e)))

    if errors:
        print(f'Completed with {len(errors)} errors:')
        for path, err in errors[:20]:
            print(f'  {path}: {err}')
        if len(errors) > 20:
            print(f'  ... and {len(errors) - 20} more')
    else:
        print('Completed successfully')


if __name__ == '__main__':
    main()
