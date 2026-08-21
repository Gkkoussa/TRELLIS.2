#!/bin/bash
#SBATCH --job-name=trifield-zarr-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=standard
#SBATCH --account=jjparkcv_owned1

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
OUTPUT="$ROOT/validation/triangle_field_zarr_benchmark_128"

python data_toolkit/benchmark_triangle_field_zarr.py \
  --dataset_root "$ROOT" \
  --instances "$ROOT/splits/train_triangle_field_512/instances.txt" \
  --output_dir "$OUTPUT" \
  --num_instances 128 \
  --resolutions 32,64,128,256,512 \
  --lower_suffix avg_from_512 \
  --chunk_voxels 65536 \
  --compression_level 5 \
  --num_random_reads 512 \
  --num_workers 8 \
  --seed 0
