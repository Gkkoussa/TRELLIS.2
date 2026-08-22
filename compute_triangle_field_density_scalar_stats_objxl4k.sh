#!/bin/bash
#SBATCH --job-name=trifield-density-scalar-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=jjparkcv_owned1

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
VOXELS=$ROOT/triangle_field_voxels_density_elongation_field_nested_from_512/triangle_field_voxels_128
LATENTS=$ROOT/triangle_field_latents/triangle_field_vae_allres_avgfrom512_invarea_fullaux_52727251_step0060000_128
OUTPUT=$ROOT/triangle_field_density_stats/density128_avgfrom512_filtered_train_scalar_stats.json
PER_MESH_OUTPUT=$ROOT/triangle_field_density_stats/density128_avgfrom512_filtered_train_per_mesh.csv

python data_toolkit/compute_triangle_field_density_scalar_stats.py \
  --voxel_dir "$VOXELS" \
  --instances "$ROOT/splits/train_triangle_field_512/instances.txt" \
  --metadata_csv "$ROOT/triangle_field_voxels_64_avg_from_512/metadata.csv" \
  --metadata_csv "$ROOT/triangle_field_voxels_128_avg_from_512/metadata.csv" \
  --metadata_csv "$LATENTS/metadata.csv" \
  --metadata_csv "$VOXELS/metadata.csv" \
  --output "$OUTPUT" \
  --per_mesh_output "$PER_MESH_OUTPUT" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
