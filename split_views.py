import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["ROOT"])
src_root = root / "gaussian_distance_voxels_64"
src_meta = pd.read_csv(src_root / "metadata.csv")
src_meta["sha256"] = src_meta["sha256"].astype(str)
src_meta = src_meta.set_index("sha256")

for split in ["train", "test"]:
    split_dir = root / "splits" / split
    split_meta = pd.read_csv(split_dir / "metadata.csv")
    split_meta["sha256"] = split_meta["sha256"].astype(str)

    out_dir = split_dir / "gaussian_distance_voxels_64"
    out_dir.mkdir(parents=True, exist_ok=True)

    shas = split_meta["sha256"].tolist()
    missing = [s for s in shas if s not in src_meta.index]
    if missing:
        print(split, "missing 64 voxel metadata:", len(missing))
        shas = [s for s in shas if s in src_meta.index]

    voxel_meta = src_meta.loc[shas].reset_index()[
        ["sha256", "gaussian_distance_voxelized", "num_gaussian_distance_voxels"]
    ]
    voxel_meta.to_csv(out_dir / "metadata.csv", index=False)

    linked = 0
    for sha in shas:
        src = src_root / f"{sha}.vxz"
        dst = out_dir / f"{sha}.vxz"
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.relpath(src, out_dir), dst)
        linked += 1

    print(split, "rows:", len(voxel_meta), "links:", linked)
