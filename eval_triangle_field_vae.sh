#!/bin/bash
#SBATCH --job-name=trellis-trifield-vae-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch eval_triangle_field_vae.sh /path/to/run_dir"
  exit 1
fi

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_DIR="$1"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_${EVAL_SPLIT}_${EVAL_CKPT}_${SLURM_JOB_ID}}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-16}"
export EVAL_SNAPSHOT_BATCH_SIZE="${EVAL_SNAPSHOT_BATCH_SIZE:-1}"
export EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-512}"
export EVAL_WEIGHT_CLAMP_MAX="${EVAL_WEIGHT_CLAMP_MAX:-100.0}"

python /home/koussa/scratch/TRELLIS.2/eval_triangle_field_vae.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --ckpt "$EVAL_CKPT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$EVAL_NUM_WORKERS" \
  --num_samples "$EVAL_NUM_SAMPLES" \
  --snapshot_batch_size "$EVAL_SNAPSHOT_BATCH_SIZE" \
  --render_resolution "$EVAL_RENDER_RESOLUTION" \
  --weight_clamp_max "$EVAL_WEIGHT_CLAMP_MAX"
