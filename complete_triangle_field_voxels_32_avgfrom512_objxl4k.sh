#!/bin/bash
#SBATCH --job-name=fill-trifield-avg512-r32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
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

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k

python data_toolkit/downsample_triangle_field_voxels.py \
  --source_root "$ROOT/triangle_field_voxels_512" \
  --output_root "$ROOT" \
  --source_resolution 512 \
  --resolutions 32 \
  --instances "$ROOT/splits/train_triangle_field_512/instances.txt" \
  --feature_dtype float16 \
  --compression zstd \
  --zstd_level 5 \
  --device cuda \
  --max_workers 1 \
  --skip_existing
