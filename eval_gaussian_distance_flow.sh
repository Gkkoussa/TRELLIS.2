#!/bin/bash
#SBATCH --job-name=trellis-gdist-flow-eval
#SBATCH --output=./job_logs/trellis-gdist-flow-eval_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gres=gpu:1

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

mkdir -p job_logs

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
export LATENT_NAME="${LATENT_NAME:-gaussian_distance_vae_step0230000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export RUN_NAME="${RUN_NAME:-michelangelo2gaussian_distance_flow_50023629}"
export RUN_DIR="${1:-$ROOT/outputs/$RUN_NAME}"

export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_${EVAL_SPLIT}_${SLURM_JOB_ID}}"
export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-64}"
export EVAL_SNAPSHOT_BATCH_SIZE="${EVAL_SNAPSHOT_BATCH_SIZE:-4}"
export EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-1024}"

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

python /home/gpranav/pranav_work/scratch/TRELLIS.2/eval_gaussian_distance_flow.py \
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
  "${EXTRA_ARGS[@]}"
