#!/bin/bash
#SBATCH --job-name=trellis-gdist-flow-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch eval_gaussian_distance_flow.sh /path/to/flow_run_dir"
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
export LATENT_NAME="${LATENT_NAME:-gaussian_distance_vae_step0350000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-64}"
export EVAL_SNAPSHOT_BATCH_SIZE="${EVAL_SNAPSHOT_BATCH_SIZE:-4}"
export EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-512}"
export EVAL_SAMPLING_STEPS="${EVAL_SAMPLING_STEPS:-12}"
export EVAL_GUIDANCE_STRENGTH="${EVAL_GUIDANCE_STRENGTH:-3.0}"

EXTRA_ARGS=()
if [ -n "${EVAL_EMA_RATE:-}" ]; then
  EXTRA_ARGS+=(--ema_rate "$EVAL_EMA_RATE")
fi
if [ -n "${EVAL_BATCH_SIZE:-}" ]; then
  EXTRA_ARGS+=(--batch_size "$EVAL_BATCH_SIZE")
fi
if [ -n "${EVAL_NUM_WORKERS:-}" ]; then
  EXTRA_ARGS+=(--num_workers "$EVAL_NUM_WORKERS")
fi
if [ -n "${EVAL_MAX_BATCHES:-}" ]; then
  EXTRA_ARGS+=(--max_batches "$EVAL_MAX_BATCHES")
fi

python /home/koussa/scratch/TRELLIS.2/eval_gaussian_distance_flow.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --ckpt "$EVAL_CKPT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --gaussian_distance_latent_name "$LATENT_NAME" \
  --michelangelo_latent_name "$MICHELANGELO_NAME" \
  --num_samples "$EVAL_NUM_SAMPLES" \
  --snapshot_batch_size "$EVAL_SNAPSHOT_BATCH_SIZE" \
  --render_resolution "$EVAL_RENDER_RESOLUTION" \
  --sampling_steps "$EVAL_SAMPLING_STEPS" \
  --guidance_strength "$EVAL_GUIDANCE_STRENGTH" \
  "${EXTRA_ARGS[@]}"
