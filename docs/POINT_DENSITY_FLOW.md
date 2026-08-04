# Point Density Flow

This experiment predicts the existing 512-resolution surface-density target on
arbitrary mesh surface points. It uses Hunyuan3D's point cross-attention encoder
and query decoder with an eight-block, timestep-modulated Hunyuan DiT latent
stack. It uses the TRELLIS trainer, flow schedule, logging, EMA, checkpoints, and
snapshot workflow.

## Dependencies

The model imports architecture code from
`/home/koussa/scratch/Hunyuan3D-2.1/hy3dshape` by default. Override
`HUNYUAN3D_SHAPE_ROOT` if that source tree moves. It also requires `einops` and a
CUDA-enabled `torch_cluster`; a CPU-only build cannot run FPS on training data.
For the current RTX PRO 6000 Blackwell and Ampere `spgpu2` nodes, build it with:

```bash
module load gcc/11.2.0 cuda/12.8.1
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0;8.6;12.0' MAX_JOBS=8 \
  pip install --force-reinstall --no-deps --no-build-isolation \
  --no-binary=torch_cluster --no-cache-dir torch_cluster==1.6.3
```

## Data And Filtering

Meshes are centered at their bounding-box center and uniformly scaled by
`0.99999 / max_bbox_extent`, matching the TRELLIS triangle-field convention.
Training samples 81,920 context points and 16,384 held-out query points online,
with exact face normals. Hunyuan FPS reduces the context to 4,096 latent anchors.

The density target at a sampled point is

```text
-log(barycentric_interpolated_vertex_mean_incident_area / (1 / 512)^2 + 1e-8)
```

One mean and standard deviation computed over the filtered training set are used
for both training and evaluation. Do not compute independent validation stats.

The submission scripts default to:

- Train: `splits/train_triangle_field_512/metadata.csv`
- Validation: intersection of `splits/test_triangle_field_512/metadata.csv` and
  `metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv`

## Prepare Statistics

```bash
sbatch compute_point_density_stats_objxl4k.sh
```

The default output is:

```text
/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/point_density_stats/density512_filtered_train.json
```

This is only a compact normalization file. Surface points and densities remain
online-sampled, so no second voxelized or point-sampled dataset is created.

## Train

After the statistics job finishes:

```bash
sbatch train_point_density_flow_hunyuan_dit8.sh
```

The model/trainer config is
`configs/gen/point_density_flow_hunyuan_dit8_512_bf16_objxl4k.json`. The script
requests two RTX PRO 6000 GPUs, 8 CPUs and 60 GB RAM per GPU. Override `RUN_NAME`,
`POINT_DENSITY_STATS`, or the filter CSV environment variables when needed.

Outputs follow the normal TRELLIS layout:

```text
$ROOT/outputs/$RUN_NAME/ckpts/
$ROOT/outputs/$RUN_NAME/samples/
$ROOT/outputs/$RUN_NAME/log.txt
$ROOT/outputs/$RUN_NAME/tb_logs/
```

The context objective is L2 velocity after converting the model's predicted
clean density to velocity. The held-out query objective is L2 clean-density
prediction with equal weight. Timesteps use `logitNormal(mean=0, std=1)` and
`sigma_min=1e-5`.

## Validation And Inference

Validation snapshots use deterministic surface samples and deterministic FPS
anchors. The anchor identities remain fixed throughout each Euler trajectory.
Snapshots save ground-truth query density, predicted query density, and the
sampled context density.

At inference, sample a fixed context point set and normals from the normalized
mesh, choose fixed FPS anchors once, and denoise context density on those points.
After reaching `t=0`, re-encode once and cache the latent tokens; this final
re-encode is inference-only and allows density queries at arbitrary new surface
points and normals.
