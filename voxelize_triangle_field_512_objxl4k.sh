#!/bin/bash
#SBATCH --job-name=trifield-512-objxl4k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export INSTANCES="${INSTANCES:-$ROOT/instances_no_triangle_dense.txt}"
export RESOLUTION="${RESOLUTION:-512}"
export MAX_WORKERS="${MAX_WORKERS:-8}"
export ZSTD_LEVEL="${ZSTD_LEVEL:-5}"

python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --resolution "$RESOLUTION" \
  --instances "$INSTANCES" \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level "$ZSTD_LEVEL" \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers "$MAX_WORKERS"
