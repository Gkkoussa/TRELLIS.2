#!/bin/bash
#SBATCH --job-name=trifield-avgdown-512
#SBATCH --partition=standard
#SBATCH --account=jjparkcv_owned1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/trifield-avgdown-512-%j.out
#SBATCH --error=logs/trifield-avgdown-512-%j.err

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export PYTHON="${PYTHON:-/home/koussa/scratch/envs/trellis2/bin/python}"
export SOURCE_ROOT="${SOURCE_ROOT:-$ROOT/triangle_field_voxels_512}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT}"
export RESOLUTIONS="${RESOLUTIONS:-32,64,128,256}"
export INSTANCES="${INSTANCES:-$ROOT/instances_no_triangle_dense.txt}"
export MAX_WORKERS="${MAX_WORKERS:-8}"
export CHUNKSIZE="${CHUNKSIZE:-2}"
export FEATURE_DTYPE="${FEATURE_DTYPE:-float16}"
export ZSTD_LEVEL="${ZSTD_LEVEL:-5}"
export DEVICE="${DEVICE:-cpu}"

"$PYTHON" data_toolkit/downsample_triangle_field_voxels.py \
  --source_root "$SOURCE_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --source_resolution 512 \
  --resolutions "$RESOLUTIONS" \
  --instances "$INSTANCES" \
  --feature_dtype "$FEATURE_DTYPE" \
  --compression zstd \
  --zstd_level "$ZSTD_LEVEL" \
  --device "$DEVICE" \
  --max_workers "$MAX_WORKERS" \
  --chunksize "$CHUNKSIZE" \
  --skip_existing
