# Local OBJ Dataset to TRELLIS Edge-Distance VAE Data

This folder contains small helpers for converting a local tree of `.obj` files into TRELLIS-compatible edge-distance voxel data.

The intended layout is:

```text
<ROOT>/
  category_a/example_001/model.obj
  category_b/example_002/model.obj
  metadata.csv
  pbr_dumps/
  edge_distance_voxels_256/
  splits/
    train/
      metadata.csv
      edge_distance_voxels_256/
    test/
      metadata.csv
      edge_distance_voxels_256/
```

The split directories contain filtered metadata and symlinks to the canonical `.vxz` files, so train/test splits do not duplicate the voxel data.

## 0. Setup

Run from the TRELLIS repo root:

```bash
cd /home/koussa/scratch/TRELLIS.2
. ./data_toolkit/setup.sh
export ROOT=/path/to/your/local_obj_dataset
```

For your current dataset, that might be:

```bash
export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
```

## 1. Build Metadata for Local OBJs

```bash
python data_toolkit/dataset_to_trellis/build_local_obj_metadata.py \
  --root "$ROOT" \
  --overwrite
```

By default, `sha256` is a stable hash of each relative OBJ path. This avoids collisions when two different folders contain identical OBJ text but different neighboring materials or textures.

If you specifically want content hashes, use:

```bash
python data_toolkit/dataset_to_trellis/build_local_obj_metadata.py \
  --root "$ROOT" \
  --id-source file_contents \
  --overwrite
```

## 2. Dump PBR Data

This uses TRELLIS' existing Blender PBR dumper. Keep `--download_root "$ROOT"` because the local paths in metadata are relative to the dataset root.

```bash
python data_toolkit/dump_pbr.py ObjaverseXL \
  --root "$ROOT" \
  --download_root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --max_workers 8
```

Then merge the new PBR records:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT"
```

## 3. Choose Edge-Distance Scaling

For linear encoding, `--d_max` maps distances `>= d_max` to 1.0. Start with `0.05` for normalized `[-0.5, 0.5]` object space, then inspect a histogram if needed:

```bash
python data_toolkit/plot_edge_distance_hist.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --resolution 256 \
  --kind linear \
  --d_max 0.05 \
  --output "$ROOT/edge_distance_hist.png"
```

## 4. Voxelize Edge Distances

The edge-distance voxelizer uses the PBR dump geometry, computes nearest mesh-edge distance for active voxel centers, encodes the result as grayscale `base_color`, and writes `.vxz` files.

Start with one worker on a single GPU to avoid GPU memory contention:

```bash
python data_toolkit/voxelize_edge_distance.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --edge_voxel_root "$ROOT" \
  --resolution 256 \
  --kind linear \
  --d_max 0.05 \
  --max_workers 1
```

Merge edge voxel records:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --edge_voxel_root "$ROOT"
```

## 5. Create Train/Test Split Views

This creates split-specific metadata and symlinks. The filtered voxel metadata is important: TRELLIS merges metadata from every root in `data_dir`, so pointing a train run at the full voxel metadata can accidentally reintroduce test assets.

```bash
python data_toolkit/dataset_to_trellis/make_split_views.py \
  --root "$ROOT" \
  --resolution 256 \
  --voxel-kind edge_distance \
  --test-frac 0.10 \
  --seed 42 \
  --overwrite-metadata \
  --overwrite-links
```

Optional validation split:

```bash
python data_toolkit/dataset_to_trellis/make_split_views.py \
  --root "$ROOT" \
  --resolution 256 \
  --voxel-kind edge_distance \
  --test-frac 0.10 \
  --val-frac 0.05 \
  --seed 42 \
  --overwrite-metadata \
  --overwrite-links
```

## 6. Train Edge-Distance VAE on the Train Split

```bash
python train.py \
  --config configs/scvae/edge_distance_vae_next_dc_f16c32_fp16.json \
  --output_dir "$ROOT/outputs/edge_distance_vae" \
  --data_dir "{\"neuframe_train\":{\"base\":\"$ROOT/splits/train\",\"edge_distance_voxel\":\"$ROOT/splits/train/edge_distance_voxels_256\"}}"
```

The test split is available at:

```text
$ROOT/splits/test
$ROOT/splits/test/edge_distance_voxels_256
```

## Notes

- This workflow uses `ObjaverseXL` only as the TRELLIS dataset adapter; it does not download Objaverse assets when `metadata.csv` already exists.
- OBJ sidecar files such as `.mtl` files and textures should remain next to the OBJ paths referenced in `metadata.csv`.
- If the training config filters on `aesthetic_score`, the metadata builder gives every local OBJ a default score of `5.0`.
- If `--kind linear` is used for voxelization, always pass `--d_max`.
- For large datasets, run `dump_pbr.py` and `voxelize_edge_distance.py` with `--rank` and `--world_size` across jobs, then run `build_metadata.py` once all shards finish.
