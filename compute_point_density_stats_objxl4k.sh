#!/bin/bash
#SBATCH --job-name=point-density-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=jjparkcv_owned1

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

eval "$(conda shell.bash hook)"
conda activate trellis2

ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
FIELD_NAME="${FIELD_NAME:-density}"
TRAIN_FILTER_CSV="${TRAIN_FILTER_CSV:-$ROOT/splits/train_triangle_field_512/metadata.csv}"
OUTPUT="${OUTPUT:-$ROOT/point_density_stats/${FIELD_NAME}512_filtered_train.json}"

python data_toolkit/compute_point_density_stats.py \
  --root "$ROOT" \
  --metadata_filter_csv "$TRAIN_FILTER_CSV" \
  --output "$OUTPUT" \
  --field_name "$FIELD_NAME" \
  --resolution 512 \
  --samples_per_mesh 4096 \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
