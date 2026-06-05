#!/bin/bash
#SBATCH --job-name=trellis-occ-shape-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch eval_occupancy_shape_vae.sh /path/to/run_dir"
  exit 1
fi

cd /home/koussa/scratch/TRELLIS.2

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
export RUN_DIR="$1"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_${EVAL_SPLIT}_${SLURM_JOB_ID}}"

EXTRA_ARGS=()
if [ -n "${EVAL_BATCH_SIZE:-}" ]; then
  EXTRA_ARGS+=(--batch_size "$EVAL_BATCH_SIZE")
fi
if [ -n "${EVAL_NUM_WORKERS:-}" ]; then
  EXTRA_ARGS+=(--num_workers "$EVAL_NUM_WORKERS")
fi
if [ -n "${EVAL_MAX_BATCHES:-}" ]; then
  EXTRA_ARGS+=(--max_batches "$EVAL_MAX_BATCHES")
fi
if [ -n "${EVAL_NUM_SAMPLES:-}" ]; then
  EXTRA_ARGS+=(--num_samples "$EVAL_NUM_SAMPLES")
fi
if [ -n "${EVAL_SNAPSHOT_BATCH_SIZE:-}" ]; then
  EXTRA_ARGS+=(--snapshot_batch_size "$EVAL_SNAPSHOT_BATCH_SIZE")
fi

python /home/koussa/scratch/TRELLIS.2/eval_occupancy_shape_vae.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --ckpt "$EVAL_CKPT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --deterministic_posterior \
  "${EXTRA_ARGS[@]}"
