# Local Mesh Directory to TRELLIS Gaussian-Distance / Triangle-Field Flow Pipeline

This folder contains helpers for converting a local tree of mesh files into TRELLIS-compatible data and training the custom Gaussian-distance and triangle-field branches discussed in this repo.

This README is organized around the full end-to-end path:

1. local mesh directory
2. mesh / PBR preprocessing
3. Gaussian-distance voxel generation
4. train/test split creation
5. Gaussian-distance VAE training + evaluation
6. optional occupancy shape-latent VAE training for TRELLIS-style subdivision conditioning
7. Michelangelo-latent preprocessing for geometry conditioning
8. Gaussian-distance latent preprocessing
9. Michelangelo-conditioned Gaussian-distance flow training
10. optional triangle-field voxel / VAE / latent / flow training

The important boundary is:

- Everything through the Gaussian-distance VAE is implemented in this repo.
- The Michelangelo-conditioned Gaussian-distance flow training path is also implemented, using precomputed Michelangelo latents as conditioning and precomputed Gaussian-distance VAE latents as the flow target.
- The triangle-field path is implemented as a parallel target representation: sparse triangle-field voxels -> triangle-field VAE latents -> Michelangelo + shape conditioned triangle-field latent flow.

## Expected Root Layout

Start with a dataset root that contains meshes somewhere under it:

```text
<ROOT>/
  category_a/example_001/model.obj
  category_b/example_002/model.obj
```

After preprocessing, the layout should grow to something like:

```text
<ROOT>/
  metadata.csv
  mesh_dumps/
  pbr_dumps/
  dual_grid_256/
  gaussian_distance_voxels_256/
  triangle_field_voxels_256/
  michelangelo_latents/
    shapevae256_pretrained/
  gaussian_distance_latents/
    gaussian_distance_vae_step0350000_256/
  triangle_field_latents/
    triangle_field_vae_51483691_step0060000_256/
  splits/
    train/
      metadata.csv
      gaussian_distance_voxels_256/
      triangle_field_voxels_256/
      michelangelo_latents/
        shapevae256_pretrained/
      gaussian_distance_latents/
        gaussian_distance_vae_step0350000_256/
      triangle_field_latents/
        triangle_field_vae_51483691_step0060000_256/
    test/
      metadata.csv
      gaussian_distance_voxels_256/
      triangle_field_voxels_256/
      michelangelo_latents/
        shapevae256_pretrained/
      gaussian_distance_latents/
        gaussian_distance_vae_step0350000_256/
      triangle_field_latents/
        triangle_field_vae_51483691_step0060000_256/
  outputs/
```

For your current dataset, `<ROOT>` is:

```bash
export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
```

## 0. Setup

Run from the TRELLIS repo root:

```bash
cd /home/koussa/scratch/TRELLIS.2
. ./data_toolkit/setup.sh
```

If you are using the `trellis2` conda env directly:

```bash
eval "$(conda shell.bash hook)"
conda activate trellis2
```

## 1. Build Root Metadata for Local Meshes

This creates a TRELLIS-style `metadata.csv` for a local directory tree.

```bash
python data_toolkit/dataset_to_trellis/build_local_obj_metadata.py \
  --root "$ROOT/meshes" \
  --output "$ROOT/metadata.csv" \
  --overwrite
```

By default, `sha256` is a stable hash of each relative mesh path. That is usually better than a content hash for local OBJ trees, because two different folders can reuse identical OBJ text while relying on different sidecar files.

If you explicitly want content hashes:

```bash
python data_toolkit/dataset_to_trellis/build_local_obj_metadata.py \
  --root "$ROOT/meshes" \
  --output "$ROOT/metadata.csv" \
  --id-source file_contents \
  --overwrite
```

The dump scripts below resolve `local_path` relative to `--download_root`. With the commands above, `local_path` is relative to `$ROOT/meshes`, so use `--download_root "$ROOT/meshes"`. If you build metadata with a different metadata root, set `--download_root` to that same directory.

### Validate

```bash
python - <<'PY'
import os, pandas as pd
root = os.environ["ROOT"]
df = pd.read_csv(os.path.join(root, "metadata.csv"))
print("rows:", len(df))
print("columns:", df.columns.tolist())
print(df.head(3).to_string(index=False))
PY
```

You should see at least:

- `sha256`
- `file_identifier`
- `local_path`
- `aesthetic_score`

## 2. Dump Meshes

This stage is required for geometry-side processing such as dual-grid conversion and shape-latent encoding.

```bash
python data_toolkit/dump_mesh.py ObjaverseXL \
  --root "$ROOT" \
  --download_root "$ROOT/meshes" \
  --mesh_dump_root "$ROOT" \
  --max_workers 8
```

Then merge stage metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --mesh_dump_root "$ROOT"
```

### Validate

```bash
python - <<'PY'
import os, pandas as pd
root = os.environ["ROOT"]
stage = os.path.join(root, "mesh_dumps")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
print("mesh_dumped:", int(df["mesh_dumped"].sum()))
print("pickle_files:", len([f for f in os.listdir(stage) if f.endswith(".pickle")]))
PY
```

If the counts differ significantly, inspect failures before moving on.

## 3. Dump PBR Data

This is still needed because the Gaussian-distance voxelizer uses the PBR dump pipeline as its standardized mesh container.

```bash
python data_toolkit/dump_pbr.py ObjaverseXL \
  --root "$ROOT" \
  --download_root "$ROOT/meshes" \
  --pbr_dump_root "$ROOT" \
  --max_workers 8
```

Merge stage metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT"
```

### Validate

```bash
python - <<'PY'
import os, pandas as pd
root = os.environ["ROOT"]
stage = os.path.join(root, "pbr_dumps")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
print("pbr_dumped:", int(df["pbr_dumped"].sum()))
print("pickle_files:", len([f for f in os.listdir(stage) if f.endswith(".pickle")]))
PY
```

## 4. Build Dual Grids at 256 (Optional / Legacy Shape-Latent Path)

This stage is only required if you also want the repo’s legacy sparse shape-latent path (`encode_shape_latent.py`).  
It is not required for the Michelangelo latent path in this README.

```bash
python data_toolkit/dual_grid.py ObjaverseXL \
  --root "$ROOT" \
  --mesh_dump_root "$ROOT" \
  --dual_grid_root "$ROOT" \
  --resolution 256 \
  --max_workers 8
```

Merge stage metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --mesh_dump_root "$ROOT" \
  --dual_grid_root "$ROOT"
```

### Validate

```bash
python - <<'PY'
import os, pandas as pd
root = os.environ["ROOT"]
stage = os.path.join(root, "dual_grid_256")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
print("dual_grid_converted:", int(df["dual_grid_converted"].sum()))
print("vxz_files:", len([f for f in os.listdir(stage) if f.endswith(".vxz")]))
PY
```

Inspect one sample:

```bash
python - <<'PY'
import os, pandas as pd, o_voxel
root = os.environ["ROOT"]
stage = os.path.join(root, "dual_grid_256")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
sha = df[df["dual_grid_converted"] == True].iloc[0]["sha256"]
coords, attr = o_voxel.io.read_vxz(os.path.join(stage, f"{sha}.vxz"))
print("sha:", sha)
print("coords:", coords.shape, coords.dtype)
print("attrs:", {k: (v.shape, str(v.dtype)) for k, v in attr.items()})
PY
```

Expected attr keys:

- `vertices`
- `intersected`

## 5. Optional: Inspect Edge-Distance Scale

This does not affect correctness, but it can help choose sigma multipliers:

```bash
python data_toolkit/plot_edge_distance_hist.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --resolution 256 \
  --kind raw \
  --output "$ROOT/edge_distance_hist.png"
```

## 6. Voxelize 6-Channel Gaussian Distance Features

The Gaussian-distance voxelizer writes 6 channels total:

- `base_color[0:3]` = edge-distance channels
- `emissive[0:3]` = vertex-distance channels

It uses the transform:

```text
1 - exp(-0.5 * (d / sigma)^2)
```

with `sigma` interpreted as a multiple of voxel length.

Start with one worker on one GPU:

```bash
python data_toolkit/voxelize_gaussian_distance.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --resolution 256 \
  --sigma_multipliers 0.5,3.0,10.0 \
  --max_workers 1
```

If you want timing breakdowns:

```bash
python data_toolkit/voxelize_gaussian_distance.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --resolution 256 \
  --sigma_multipliers 0.5,3.0,10.0 \
  --max_workers 1 \
  --benchmark
```

Merge stage metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT"
```

### Validate

```bash
python - <<'PY'
import os, pandas as pd
root = os.environ["ROOT"]
stage = os.path.join(root, "gaussian_distance_voxels_256")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
print("gaussian_distance_voxelized:", int(df["gaussian_distance_voxelized"].sum()))
print("vxz_files:", len([f for f in os.listdir(stage) if f.endswith(".vxz")]))
PY
```

Visualize a few processed samples:

```bash
python o-voxel/examples/render_vxz_channels.py \
  "$ROOT/gaussian_distance_voxels_256" \
  --grid_size 256 \
  --max_num 20
```

That utility renders:

- edge RGB
- edge channel 0/1/2
- vertex RGB
- vertex channel 0/1/2

## 7. Create Train/Test Split Views for Gaussian-Distance Voxels

This creates split-specific metadata and symlinks. The filtered stage metadata is important because TRELLIS merges metadata from every root in `data_dir`.

```bash
python data_toolkit/dataset_to_trellis/make_split_views.py \
  --root "$ROOT" \
  --resolution 256 \
  --voxel-kind gaussian_distance \
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
  --voxel-kind gaussian_distance \
  --test-frac 0.10 \
  --val-frac 0.05 \
  --seed 42 \
  --overwrite-metadata \
  --overwrite-links
```

### Validate

Check that these exist:

```text
$ROOT/splits/train/metadata.csv
$ROOT/splits/train/gaussian_distance_voxels_256/metadata.csv
$ROOT/splits/test/metadata.csv
$ROOT/splits/test/gaussian_distance_voxels_256/metadata.csv
```

Quick check:

```bash
python - <<'PY'
import os, pandas as pd
root = os.environ["ROOT"]
for split in ["train", "test"]:
    p = os.path.join(root, "splits", split, "metadata.csv")
    df = pd.read_csv(p)
    print(split, len(df))
    print(df[["sha256", "gaussian_distance_voxelized", "num_gaussian_distance_voxels", "aesthetic_score"]].head(2).to_string(index=False))
PY
```

## 8. Train the 6-Channel Gaussian-Distance VAE

Single-job Slurm script:

```bash
sbatch train_gaussian_distance_vae.sh
```

That script writes to:

```text
$ROOT/outputs/gaussian_distance_vae
```

To run multiple independent VAE runs, override `RUN_NAME` manually when submitting the job.

If you want to run it directly:

```bash
python train.py \
  --config configs/scvae/gaussian_distance_vae_next_dc_f16c32_fp16.json \
  --output_dir "$ROOT/outputs/gaussian_distance_vae_manual" \
  --data_dir "{\"neuframe_train\":{\"base\":\"$ROOT/splits/train\",\"gaussian_distance_voxel\":\"$ROOT/splits/train/gaussian_distance_voxels_256\"}}" \
  --auto_retry 0
```

### Validate

During training, inspect:

- `log.txt`
- `tb_logs/`
- `samples/`
- `ckpts/`

inside the run directory.

Important note:

- checkpoints only appear after `i_save` steps
- sample images should appear immediately under `samples/init/`

## 9. Evaluate the Gaussian-Distance VAE on the Test Split

Use the 1-GPU Slurm eval script. The script defaults to `train`, so set `EVAL_SPLIT=test` for held-out evaluation:

```bash
EVAL_SPLIT=test sbatch eval_gaussian_distance_vae.sh \
  "$ROOT/outputs/gaussian_distance_vae"
```

Or run directly:

```bash
python eval_pbr_vae.py \
  --run_dir "$ROOT/outputs/gaussian_distance_vae" \
  --root "$ROOT" \
  --split test \
  --num_samples 64 \
  --snapshot_batch_size 4 \
  --deterministic_posterior
```

### Validate

Inspect:

- `metrics.json`
- `samples/`

inside:

```text
$RUN_DIR/eval_test_<...>/
```

The key metrics are:

- `l1`
- `edge_l1`
- `vertex_l1`
- `kl`
- `total`

At this point, the Gaussian-distance VAE branch is fully trained and evaluated.

## 10. Optional: Train an Occupancy Shape-Latent VAE

This is the simplified shape-latent path for later TRELLIS-style conditioning.

Purpose:

- Train a sparse shape latent whose decoder learns subdivision / upsampling coordinates.
- Use constant occupancy features rather than full mesh/surface reconstruction targets.
- Eventually use these shape latents as `concat_cond` for Gaussian-distance flow, and use the shape decoder's `return_subs=True` output as `guide_subs` for the Gaussian-distance decoder.

This is **not** the stock TRELLIS2 `ShapeVaeTrainer` setup. The stock setup uses FlexiDualGrid inputs, mesh reconstruction, render losses, vertex losses, and intersected-voxel losses. This simplified setup uses:

- existing sparse voxel coordinates from `gaussian_distance_voxels_256`
- one feature channel with value `1.0` at every active voxel
- `SparseUnetVaeEncoder`
- `SparseUnetVaeDecoder` with `pred_subdiv=true`
- subdivision BCE loss plus KL loss

Run training:

```bash
sbatch train_occupancy_shape_vae.sh
```

That script writes to:

```text
$ROOT/outputs/occupancy_shape_vae
```

To run a separate experiment:

```bash
RUN_NAME=occupancy_shape_vae_test1 sbatch train_occupancy_shape_vae.sh
```

### Validate

During training, inspect:

- `log.txt`
- `tb_logs/`
- `samples/`
- `ckpts/`

inside:

```text
$ROOT/outputs/occupancy_shape_vae
```

Snapshots are saved every `10000` steps. The important snapshot outputs are:

- `sample_gt_occupancy_*`: original active sparse voxel support
- `sample_rec_occupancy_*`: reconstruction using cached GT subdivision structure
- `sample_pred_subdiv_occupancy_*`: reconstruction after clearing the cache, so the decoder must use predicted subdivisions

The useful diagnostic is whether `pred_subdiv_occupancy` recovers the sparse support well. Occupancy snapshots render active voxels from four viewpoints and color them by normalized distance from the origin to make structure easier to see.

Run standalone eval on the held-out split with:

```bash
EVAL_SPLIT=test sbatch eval_occupancy_shape_vae.sh "$ROOT/outputs/occupancy_shape_vae"
```

This writes `metrics.json` and snapshot images into:

```text
$ROOT/outputs/occupancy_shape_vae/eval_test_<SLURM_JOB_ID>/
```

The main support metrics are:

- `exact_support_match_rate`: fraction of evaluated instances where predicted sparse coordinates exactly match GT sparse coordinates
- `all_gt_locations_captured_rate`: fraction of evaluated instances where every GT coordinate is present in the prediction, even if extra coordinates were also predicted
- `mean_iou`, `mean_recall`, `mean_precision`: per-instance sparse support metrics
- `micro_iou`, `micro_recall`, `micro_precision`: pooled coordinate-count metrics over the whole eval run

### Encode Occupancy Shape Latents

After the occupancy shape VAE is trained, encode its latents so they can be used as sparse `concat_cond` for the Gaussian-distance flow model.

```bash
python data_toolkit/encode_occupancy_shape_latent.py \
  --root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --shape_latent_root "$ROOT" \
  --resolution 256 \
  --model_root "$ROOT/outputs" \
  --enc_model occupancy_shape_vae \
  --ckpt step0090000 \
  --loader_workers 4 \
  --read_threads 1 \
  --saver_workers 4
```

This writes:

```text
$ROOT/shape_latents/occupancy_shape_vae_step0090000_256/
```

Merge metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --shape_latent_root "$ROOT"
```

## 11. Encode Michelangelo Latents at 256

This is the geometry-conditioning side needed for a pointcloud-conditioned distance model.

Encode with the pretrained Michelangelo shape model. This writes all latent `.npz` files into the canonical latent root:

```text
$ROOT/michelangelo_latents/shapevae256_pretrained/
```

Train/test separation is handled later by `make_latent_split_views.py`, using the existing split `instances.txt` files.

```bash
python data_toolkit/encode_michelangelo_latent.py \
  --root "$ROOT" \
  --mesh_dump_root "$ROOT" \
  --michelangelo_latent_root "$ROOT" \
  --ckpt_path "$MICHELANGELO_CKPT" \
  --latent_name shapevae256_pretrained \
  --batch_size 16 \
  --max_workers 4 \
  --saver_workers 4 \
  --coordinate_scale 2.0
```

Merge stage metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --michelangelo_latent_root "$ROOT"
```

### Validate

```bash
python - <<'PY'
import os, pandas as pd
root = os.environ["ROOT"]
stage = os.path.join(root, "michelangelo_latents", "shapevae256_pretrained")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
print("michelangelo_latent_encoded:", int(df["michelangelo_latent_encoded"].sum()))
print("npz_files:", len([f for f in os.listdir(stage) if f.endswith(".npz")]))
PY
```

Inspect one latent:

```bash
python - <<'PY'
import os, numpy as np, pandas as pd
root = os.environ["ROOT"]
stage = os.path.join(root, "michelangelo_latents", "shapevae256_pretrained")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
sha = df[df["michelangelo_latent_encoded"] == True].iloc[0]["sha256"]
z = np.load(os.path.join(stage, f"{sha}.npz"))
print("sha:", sha)
print("feats:", z["feats"].shape, z["feats"].dtype)
PY
```

You should see:

- `feats`
- token count typically `256`
- latent feature width typically `64`

`--coordinate_scale 2.0` is intentional: TRELLIS mesh dumps are roughly in `[-0.5, 0.5]`, while Michelangelo training normalizes pointcloud coordinates to approximately `[-1, 1]`.

## 12. Encode Gaussian-Distance Latents

This is the latent representation that the Michelangelo-conditioned flow model predicts.

The script mirrors `encode_pbr_latent.py`, but reads:

- `base_color`
- `emissive`

from `gaussian_distance_voxels_256/`.

If you are using your own trained Gaussian-distance VAE encoder, you should pass:

- `--model_root`
- `--enc_model`
- `--ckpt`

`--ckpt` is required when `--enc_model` is used.

Example using a trained run. This writes all latent `.npz` files into the canonical latent root:

```text
$ROOT/gaussian_distance_latents/<GAUSSIAN_DISTANCE_LATENT_NAME>/
```

Train/test separation is handled later by `make_latent_split_views.py`, using the existing split `instances.txt` files.

```bash
python data_toolkit/encode_gaussian_distance_latent.py \
  --root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --gaussian_distance_latent_root "$ROOT" \
  --resolution 256 \
  --model_root "$ROOT/outputs" \
  --enc_model gaussian_distance_vae \
  --ckpt step0350000 \
  --loader_workers 4 \
  --read_threads 1 \
  --saver_workers 4
```

The encoder writes both:

- `<sha>.npz`, containing sparse latent `coords` and `feats`
- `<sha>.cache.pt`, containing the VAE spatial cache needed for decoder-backed flow snapshots

### Validate

Check that the stage directory exists:

```text
$ROOT/gaussian_distance_latents/<latent_name>/
```

Quick check:

```bash
python - <<'PY'
import os, numpy as np
root = os.environ["ROOT"]
latent_root = os.path.join(root, "gaussian_distance_latents")
print("latent_models:", sorted(os.listdir(latent_root))[:10])
PY
```

Inspect one latent file:

```bash
python - <<'PY'
import os, numpy as np
root = os.environ["ROOT"]
latent_root = os.path.join(root, "gaussian_distance_latents")
latent_name = sorted(os.listdir(latent_root))[0]
stage = os.path.join(latent_root, latent_name)
npz = sorted([f for f in os.listdir(stage) if f.endswith('.npz')])[0]
z = np.load(os.path.join(stage, npz))
print("file:", npz)
print("coords:", z["coords"].shape, z["coords"].dtype)
print("feats:", z["feats"].shape, z["feats"].dtype)
print("finite:", np.isfinite(z["coords"]).all(), np.isfinite(z["feats"]).all())
PY
```

Check that cache sidecars exist too:

```bash
find "$ROOT/gaussian_distance_latents/gaussian_distance_vae_step0350000_256" \
  -name '*.cache.pt' | head
```

## 13. Create Split-Specific Latent Views

Use the existing split instance files to create split-local latent roots without changing split membership.
This is the step that makes train/test latent paths explicit, because the encoders write canonical latent roots by default.

First make sure latent-stage metadata has been merged:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --michelangelo_latent_root "$ROOT" \
  --gaussian_distance_latent_root "$ROOT"
```

Then create Michelangelo-latent split views:

```bash
python data_toolkit/dataset_to_trellis/make_latent_split_views.py \
  --root "$ROOT" \
  --latent-kind michelangelo_latent \
  --latent-name shapevae256_pretrained \
  --overwrite-metadata \
  --overwrite-links
```

Then create Gaussian-distance latent split views:

```bash
python data_toolkit/dataset_to_trellis/make_latent_split_views.py \
  --root "$ROOT" \
  --latent-kind gaussian_distance_latent \
  --latent-name gaussian_distance_vae_step0350000_256 \
  --overwrite-metadata \
  --overwrite-links
```

For Gaussian-distance latents, this also links the `.cache.pt` sidecars when they exist.

If you are using the occupancy shape-latent conditioning path, also create shape-latent split views:

```bash
python data_toolkit/dataset_to_trellis/make_latent_split_views.py \
  --root "$ROOT" \
  --latent-kind shape_latent \
  --latent-name occupancy_shape_vae_step0090000_256 \
  --overwrite-metadata \
  --overwrite-links
```

### Validate

Check for:

```text
$ROOT/splits/train/michelangelo_latents/<name>/metadata.csv
$ROOT/splits/test/michelangelo_latents/<name>/metadata.csv
$ROOT/splits/train/gaussian_distance_latents/<name>/metadata.csv
$ROOT/splits/test/gaussian_distance_latents/<name>/metadata.csv
$ROOT/splits/train/shape_latents/<name>/metadata.csv
$ROOT/splits/test/shape_latents/<name>/metadata.csv
```

Also check the Gaussian-distance cache sidecars in the split-local view:

```bash
find "$ROOT/splits/train/gaussian_distance_latents/gaussian_distance_vae_step0350000_256" \
  -name '*.cache.pt' | head
```

## 14. Compute Latent Normalization Statistics

The flow dataset normalizes the Gaussian-distance target latents. Compute these stats from the **train split only**.

Gaussian-distance latent stats:

```bash
python data_toolkit/compute_latent_normalization.py \
  --latent-root "$ROOT/splits/train/gaussian_distance_latents/gaussian_distance_vae_step0350000_256" \
  --latent-kind gaussian_distance_latent
```

Shape latent stats for the occupancy shape-conditioned flow path:

```bash
python data_toolkit/compute_latent_normalization.py \
  --latent-root "$ROOT/splits/train/shape_latents/occupancy_shape_vae_step0090000_256" \
  --latent-kind shape_latent
```

This command writes:

```text
<latent-root>/normalization.json
```

You do not need to paste the stats into the flow config. `MichelangeloConditionedGaussianDistanceSLat` auto-loads `normalization.json` from each `gaussian_distance_latent` root in `data_dir` and raises an error if it is missing. The shape-conditioned variant also requires `normalization.json` under the `shape_latent` root.

### Validate

Inspect the JSON and confirm:

- `files_used > 0`
- `total_tokens > 0`
- `mean` and `std` have the expected latent channel length
- no `std` entries are zero or NaN

## 15. Train the Michelangelo-Conditioned Gaussian-Distance Flow

The training path is implemented with:

- dataset: `trellis2/datasets/structured_latent_gaussian_distance.py`
- config: `configs/gen/slat_flow_michelangelo2gaussian_distance_dit_1_3B_256_bf16.json`
- Slurm script: `train_michelangelo2gaussian_distance_flow.sh`

Run flow training with:

```bash
sbatch train_michelangelo2gaussian_distance_flow.sh
```

The script writes to:

```text
$ROOT/outputs/michelangelo2gaussian_distance_flow_<SLURM_JOB_ID>
```

To reuse or resume a specific output directory, pass the same `RUN_NAME`:

```bash
RUN_NAME=michelangelo2gaussian_distance_flow sbatch train_michelangelo2gaussian_distance_flow.sh
```

Resume only works after a checkpoint exists in:

```text
$ROOT/outputs/<RUN_NAME>/ckpts/
```

The current flow config saves every `10000` steps.

Flow snapshots decode generated Gaussian-distance latents through the trained Gaussian-distance VAE decoder. That snapshot path depends on the `.cache.pt` sidecars created by `encode_gaussian_distance_latent.py` and linked into the split view by `make_latent_split_views.py`.

What is still not implemented is a separate standalone flow evaluation/inference script for arbitrary new point clouds. Training-time snapshots are implemented.

### Shape-Latent Conditioned Variant

To train the TRELLIS-style variant that uses occupancy shape latents as sparse `concat_cond`, run:

```bash
sbatch train_michelangelo_shape2gaussian_distance_flow.sh
```

This path uses:

- dataset: `MichelangeloShapeConditionedGaussianDistanceSLat`
- config: `configs/gen/slat_flow_michelangelo_shape2gaussian_distance_kl5e3_dit_1_3B_256_bf16.json`
- Slurm script: `train_michelangelo_shape2gaussian_distance_flow.sh`

The denoiser input channels are `64`:

```text
32 noisy Gaussian-distance latent channels + 32 occupancy shape latent channels
```

The output channels remain `32`, because the flow only predicts the Gaussian-distance latent velocity.

Evaluate this variant with:

```bash
sbatch eval_michelangelo_shape2gaussian_distance_flow.sh \
  "$ROOT/outputs/michelangelo_shape2gaussian_distance_flow_kl5e3"
```

Snapshots for this variant decode with:

```text
occupancy shape latent -> occupancy shape decoder(return_subs=True) -> guide_subs
generated Gaussian-distance latent -> Gaussian-distance decoder(guide_subs=guide_subs)
```

## 16. Triangle-Field Pipeline: OBJ to VAE to Flow

This is the newer triangle-field target path. It starts from the same `metadata.csv`, `mesh_dumps/`, and `pbr_dumps/` produced above.

The sparse voxel file stores 20 input channels:

- `d_tri`: scalar triangle/interior field
- `d_vert`: scalar vertex field
- `offset_to_v0`, `offset_to_v1`, `offset_to_v2`
- `offset_to_centroid`
- `face_normal`
- `offset_to_projection`

The triangle-field VAE uses all 20 channels as encoder input, but only reconstructs two target channels:

```text
d_tri, d_vert
```

The flow model predicts triangle-field VAE latents. Its conditioning matches the TRELLIS-style setup:

```text
Michelangelo latent -> cross-attention cond
occupancy shape latent -> sparse concat_cond and decoder guide_subs
triangle-field latent -> flow target x_0
```

### 16.1 Voxelize Triangle-Field Features

Run from the TRELLIS repo root in the `trellis2` environment:

```bash
export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
```

Voxelize from PBR dumps:

```bash
python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --resolution 256 \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 3 \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers 8
```

For timing/debugging:

```bash
python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --resolution 256 \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 3 \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers 1 \
  --benchmark
```

Merge stage metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$ROOT"
```

Validate the processed feature statistics:

```bash
python data_toolkit/check_triangle_field_dataset.py \
  --voxel_root "$ROOT/triangle_field_voxels_256" \
  --num_workers 8 \
  --output_json "$ROOT/triangle_field_voxels_256/check_stats.json"
```

### 16.2 Create Train/Test Split Views for Triangle-Field Voxels

If this is the first split you are creating, make split-local triangle-field voxel views:

```bash
python data_toolkit/dataset_to_trellis/make_split_views.py \
  --root "$ROOT" \
  --resolution 256 \
  --voxel-kind triangle_field \
  --test-frac 0.10 \
  --seed 42 \
  --overwrite-metadata \
  --overwrite-links
```

This creates:

```text
$ROOT/splits/train/triangle_field_voxels_256/
$ROOT/splits/test/triangle_field_voxels_256/
```

If you already created split membership for another representation and need identical membership, verify that the resulting `splits/train/instances.txt` and `splits/test/instances.txt` still match your intended split before training.

### 16.3 Train the Triangle-Field VAE

Use the Slurm wrapper:

```bash
sbatch train_triangle_field_vae.sh
```

The script uses:

- config: `configs/scvae/triangle_field_vae_next_dc_f16c32_fp16.json`
- dataset: `SparseVoxelTriangleFieldDataset`
- trainer: `TriangleFieldVaeTrainer`
- train data: `$ROOT/splits/train/triangle_field_voxels_256`

The output directory defaults to:

```text
$ROOT/outputs/triangle_field_vae_<SLURM_JOB_ID>
```

To force a specific output name:

```bash
RUN_NAME=triangle_field_vae_test sbatch train_triangle_field_vae.sh
```

Inspect:

```text
$ROOT/outputs/<RUN_NAME>/log.txt
$ROOT/outputs/<RUN_NAME>/samples/
$ROOT/outputs/<RUN_NAME>/ckpts/
```

For the current flow defaults in this repo, the triangle-field VAE run/checkpoint is:

```text
$ROOT/outputs/triangle_field_vae_51483691
step0060000
```

### 16.4 Encode Triangle-Field VAE Latents

Once the VAE checkpoint exists, encode canonical triangle-field latents:

```bash
export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
export TRIANGLE_FIELD_VAE_RUN=triangle_field_vae_51483691
export TRIANGLE_FIELD_VAE_CKPT=step0060000
export TRIANGLE_FIELD_LATENT_NAME=${TRIANGLE_FIELD_VAE_RUN}_${TRIANGLE_FIELD_VAE_CKPT}_256

export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

python data_toolkit/encode_triangle_field_latent.py \
  --root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --triangle_field_latent_root "$ROOT" \
  --resolution 256 \
  --model_root "$ROOT/outputs" \
  --enc_model "$TRIANGLE_FIELD_VAE_RUN" \
  --ckpt "$TRIANGLE_FIELD_VAE_CKPT" \
  --loader_workers 4 \
  --saver_workers 4
```

This writes:

```text
$ROOT/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME/
```

Each encoded object has:

- `<sha>.npz`: sparse latent `coords` and `feats`
- `<sha>.cache.pt`: trimmed sparse spatial cache for decoder-backed snapshots when guide subdivisions are not provided

Merge latent metadata:

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --triangle_field_latent_root "$ROOT" \
  --michelangelo_latent_root "$ROOT" \
  --shape_latent_root "$ROOT"
```

### 16.5 Create Triangle-Field Latent Split Views

Create train/test split-local latent roots:

```bash
python data_toolkit/dataset_to_trellis/make_latent_split_views.py \
  --root "$ROOT" \
  --latent-kind triangle_field_latent \
  --latent-name "$TRIANGLE_FIELD_LATENT_NAME" \
  --overwrite-metadata \
  --overwrite-links
```

This creates:

```text
$ROOT/splits/train/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME/
$ROOT/splits/test/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME/
```

Compute normalization from the train split only:

```bash
python data_toolkit/compute_latent_normalization.py \
  --latent-root "$ROOT/splits/train/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME" \
  --latent-kind triangle_field_latent
```

The flow dataset auto-loads this `normalization.json`. Do not compute it from the test split.

### 16.6 Ensure Conditioning Latents Exist

The triangle-field flow path assumes these split-local conditioning roots already exist:

```text
$ROOT/splits/train/michelangelo_latents/shapevae256_pretrained/
$ROOT/splits/test/michelangelo_latents/shapevae256_pretrained/
$ROOT/splits/train/shape_latents/occupancy_shape_vae_step0110000_256/
$ROOT/splits/test/shape_latents/occupancy_shape_vae_step0110000_256/
```

The shape-latent root must also have train normalization:

```text
$ROOT/splits/train/shape_latents/occupancy_shape_vae_step0110000_256/normalization.json
```

If missing, run the Michelangelo and occupancy-shape latent sections above, then create split views and compute shape-latent normalization.

For the current triangle-field flow defaults, the expected shape latent name is:

```bash
export SHAPE_LATENT_NAME=occupancy_shape_vae_step0110000_256
```

If that canonical shape latent already exists but split views or normalization are missing, run:

```bash
python data_toolkit/dataset_to_trellis/make_latent_split_views.py \
  --root "$ROOT" \
  --latent-kind shape_latent \
  --latent-name "$SHAPE_LATENT_NAME" \
  --overwrite-metadata \
  --overwrite-links

python data_toolkit/compute_latent_normalization.py \
  --latent-root "$ROOT/splits/train/shape_latents/$SHAPE_LATENT_NAME" \
  --latent-kind shape_latent
```

### 16.7 Train the Michelangelo + Shape Conditioned Triangle-Field Flow

The training path uses:

- dataset: `MichelangeloShapeConditionedTriangleFieldSLat`
- config: `configs/gen/slat_flow_michelangelo_shape2triangle_field_dit_1_3B_256_bf16.json`
- Slurm script: `train_michelangelo_shape2triangle_field_flow.sh`
- target: `triangle_field_latent`
- cross-attention condition: `michelangelo_latent`
- sparse concat condition: `shape_latent`

The default script values currently target:

```text
TRIANGLE_FIELD_LATENT_NAME=triangle_field_vae_51483691_step0060000_256
MICHELANGELO_NAME=shapevae256_pretrained
SHAPE_LATENT_NAME=occupancy_shape_vae_step0110000_256
```

Start training:

```bash
RUN_NAME=michelangelo_shape2triangle_field_flow_51483691_step0060000 \
sbatch train_michelangelo_shape2triangle_field_flow.sh
```

The denoiser dimensions are:

```text
32 noisy triangle-field latent channels + 32 occupancy shape latent channels -> in_channels=64
32 predicted velocity channels -> out_channels=32
Michelangelo token width -> cond_channels=64
latent token grid resolution -> 16
```

The dataset checks that triangle-field latent coordinates and shape latent coordinates match exactly before concatenating. If they differ, training raises instead of silently corrupting the conditioning.

### 16.8 Evaluate Triangle-Field Flow

Run:

```bash
sbatch eval_michelangelo_shape2triangle_field_flow.sh \
  "$ROOT/outputs/michelangelo_shape2triangle_field_flow_51483691_step0060000"
```

Useful overrides:

```bash
EVAL_SPLIT=test \
EVAL_CKPT=latest \
EVAL_NUM_SAMPLES=64 \
EVAL_SAMPLING_STEPS=12 \
EVAL_GUIDANCE_STRENGTH=1.0 \
sbatch eval_michelangelo_shape2triangle_field_flow.sh \
  "$ROOT/outputs/michelangelo_shape2triangle_field_flow_51483691_step0060000"
```

Evaluation writes:

```text
$ROOT/outputs/<FLOW_RUN>/eval_test_<SLURM_JOB_ID>/metrics.json
$ROOT/outputs/<FLOW_RUN>/eval_test_<SLURM_JOB_ID>/samples/
```

Snapshots decode generated triangle-field latents with:

```text
shape latent -> occupancy shape decoder(return_subs=True) -> guide_subs
generated triangle-field latent -> triangle-field decoder(guide_subs=guide_subs)
```

### 16.9 Switching Triangle-Field VAE Checkpoints

If you train flow against a different triangle-field VAE checkpoint, treat it as a new target distribution:

1. Re-encode triangle-field latents with the new `--ckpt`.
2. Rebuild metadata.
3. Recreate triangle-field latent split views.
4. Recompute train normalization.
5. Update `triangle_field_slat_dec_ckpt` in the flow config to the same checkpoint.
6. Start a new flow run.

Do not resume a flow trained on one VAE checkpoint using latents from another checkpoint.

## 17. Target End State for the Flow Model

The intended final model is:

```text
michelangelo_latent -> gaussian_distance_latent -> gaussian_distance_decoder -> 6-channel distance voxels
```

That is the shape-conditioned analogue of the repo’s stock texturing branch:

```text
shape_latent -> pbr_latent -> pbr_decoder
```

For your branch:

- target latent = Gaussian-distance latent
- conditioning latent = Michelangelo latent
- decoder = your trained Gaussian-distance VAE decoder

For the triangle-field branch:

- target latent = Triangle-field latent
- conditioning latents = Michelangelo latent plus occupancy shape latent
- decoder = your trained triangle-field VAE decoder
- guide subdivisions = occupancy shape decoder `return_subs=True`

## Notes and Compatibility Constraints

- This workflow uses `ObjaverseXL` only as the TRELLIS dataset adapter. It does not download Objaverse assets when `metadata.csv` already exists and `local_path` points at local files.
- OBJ sidecar files such as `.mtl` files and textures should remain next to the OBJ paths referenced in `metadata.csv`.
- The Gaussian-distance voxelizer currently expects the `pbr_dump` representation as its standardized mesh input.
- Michelangelo encoding uses `mesh_dumps/` plus sampled pointclouds, not `dual_grid_256/`.
- The Gaussian-distance latent encoder currently expects either:
  - `--enc_pretrained`, or
  - `--enc_model` together with `--ckpt`
- To keep latent generation compatible with the existing VAE train/test split, reuse:
  - `splits/train/instances.txt`
  - `splits/test/instances.txt`
  rather than creating a new split.
- Latent normalization stats should be computed on the **train split only** and reused for:
  - training
  - evaluation
  - inference
- The stock latent-flow path uses normalization for both the target latent and the conditioning latent.
- For this branch, only the Gaussian-distance target latent is normalized; Michelangelo latents are used directly as the dense `cond` path.
- `build_metadata.py` updates stage-local `metadata.csv` files and `statistics.txt`, but it does not reliably propagate every stage column back into root `metadata.csv`. Validate stages using the stage directory’s own `metadata.csv`.
- The 6-channel Gaussian-distance VAE intentionally uses `lambda_render = 0.0`. The stock render path assumes standard PBR semantics and is not appropriate for arbitrary distance channels.
- For large datasets, use `--rank` and `--world_size` for:
  - `dump_mesh.py`
  - `dump_pbr.py`
  - `dual_grid.py`
  - `voxelize_gaussian_distance.py`
  - `encode_michelangelo_latent.py`
- If you interrupt voxelization and rerun, the Gaussian-distance script resumes by skipping existing compatible `.vxz` files.
