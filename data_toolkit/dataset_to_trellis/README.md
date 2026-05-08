# Local Mesh Directory to TRELLIS Gaussian-Distance Flow Pipeline

This folder contains helpers for converting a local tree of mesh files into TRELLIS-compatible data and training the custom 6-channel Gaussian-distance branch discussed in this repo.

This README is organized around the full end-to-end path:

1. local mesh directory
2. mesh / PBR preprocessing
3. Gaussian-distance voxel generation
4. train/test split creation
5. Gaussian-distance VAE training + evaluation
6. Michelangelo-latent preprocessing for geometry conditioning
7. Gaussian-distance latent preprocessing
8. Michelangelo-conditioned Gaussian-distance flow training

The important boundary is:

- Everything through the Gaussian-distance VAE is implemented in this repo.
- The Michelangelo-conditioned Gaussian-distance flow training path is also implemented, using precomputed Michelangelo latents as conditioning and precomputed Gaussian-distance VAE latents as the flow target.

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
  michelangelo_latents/
    shapevae256_pretrained/
  gaussian_distance_latents/
    gaussian_distance_vae_step0350000_256/
  splits/
    train/
      metadata.csv
      gaussian_distance_voxels_256/
      michelangelo_latents/
        shapevae256_pretrained/
      gaussian_distance_latents/
        gaussian_distance_vae_step0350000_256/
    test/
      metadata.csv
      gaussian_distance_voxels_256/
      michelangelo_latents/
        shapevae256_pretrained/
      gaussian_distance_latents/
        gaussian_distance_vae_step0350000_256/
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

## 10. Encode Michelangelo Latents at 256

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

## 11. Encode Gaussian-Distance Latents

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

## 12. Create Split-Specific Latent Views

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

### Validate

Check for:

```text
$ROOT/splits/train/michelangelo_latents/<name>/metadata.csv
$ROOT/splits/test/michelangelo_latents/<name>/metadata.csv
$ROOT/splits/train/gaussian_distance_latents/<name>/metadata.csv
$ROOT/splits/test/gaussian_distance_latents/<name>/metadata.csv
```

Also check the Gaussian-distance cache sidecars in the split-local view:

```bash
find "$ROOT/splits/train/gaussian_distance_latents/gaussian_distance_vae_step0350000_256" \
  -name '*.cache.pt' | head
```

## 13. Compute Latent Normalization Statistics

The flow dataset normalizes the Gaussian-distance target latents. Compute these stats from the **train split only**.

Gaussian-distance latent stats:

```bash
python data_toolkit/compute_latent_normalization.py \
  --latent-root "$ROOT/splits/train/gaussian_distance_latents/gaussian_distance_vae_step0350000_256" \
  --latent-kind gaussian_distance_latent
```

This command writes:

```text
<latent-root>/normalization.json
```

You do not need to paste the stats into the flow config. `MichelangeloConditionedGaussianDistanceSLat` auto-loads `normalization.json` from each `gaussian_distance_latent` root in `data_dir` and raises an error if it is missing.

### Validate

Inspect the JSON and confirm:

- `files_used > 0`
- `total_tokens > 0`
- `mean` and `std` have the expected latent channel length
- no `std` entries are zero or NaN

## 14. Train the Michelangelo-Conditioned Gaussian-Distance Flow

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

## 15. Target End State for the Flow Model

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
