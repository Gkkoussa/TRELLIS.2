"""Lightweight namespace for importing model components from Hunyuan3D-2.1."""

import os


def source_package_dir() -> str:
    root = os.environ.get(
        'HUNYUAN3D_SHAPE_ROOT',
        '/home/koussa/scratch/Hunyuan3D-2.1/hy3dshape',
    )
    package_dir = os.path.join(root, 'hy3dshape')
    if not os.path.isdir(package_dir):
        raise ImportError(
            f'Hunyuan source package not found at {package_dir}. '
            'Set HUNYUAN3D_SHAPE_ROOT to the directory containing hy3dshape/.'
        )
    return package_dir

