#!/bin/bash
#SBATCH --job-name=trellis-gdist-flow-export
#SBATCH --output=./job_logs/trellis-gdist-flow-export_%j.log
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

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export LATENT_NAME="${LATENT_NAME:-gaussian_distance_vae_step0230000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export RUN_NAME="${RUN_NAME:-michelangelo2gaussian_distance_flow_50023629}"
export RUN_DIR="${1:-$ROOT/outputs/$RUN_NAME}"

export EXPORT_SPLIT="${EXPORT_SPLIT:-test}"
export EXPORT_CKPT="${EXPORT_CKPT:-latest}"
export EXPORT_RUN_NAME="${EXPORT_RUN_NAME:-sample_debug_${EXPORT_SPLIT}_${SLURM_JOB_ID}}"
export EXPORT_NUM_SAMPLES="${EXPORT_NUM_SAMPLES:-16}"
export EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-4}"
export EXPORT_SAMPLING_STEPS="${EXPORT_SAMPLING_STEPS:-12}"
export EXPORT_GUIDANCE_STRENGTH="${EXPORT_GUIDANCE_STRENGTH:-3.0}"
export EXPORT_RENDER_RESOLUTION="${EXPORT_RENDER_RESOLUTION:-1024}"
export EXPORT_SEED="${EXPORT_SEED:-0}"

EXTRA_ARGS=()
if [ -n "${EXPORT_EMA_RATE:-}" ]; then
  EXTRA_ARGS+=(--ema_rate "$EXPORT_EMA_RATE")
fi
if [ -n "${EXPORT_START_INDEX:-}" ]; then
  EXTRA_ARGS+=(--start_index "$EXPORT_START_INDEX")
fi
if [ -n "${EXPORT_INDICES:-}" ]; then
  EXTRA_ARGS+=(--indices "$EXPORT_INDICES")
fi
if [ -n "${EXPORT_SHA256S:-}" ]; then
  EXTRA_ARGS+=(--sha256s "$EXPORT_SHA256S")
fi
if [ "${EXPORT_RANDOM:-0}" = "1" ]; then
  EXTRA_ARGS+=(--random)
fi
if [ "${EXPORT_SAVE_DECODED_NPZ:-0}" = "1" ]; then
  EXTRA_ARGS+=(--save_decoded_npz)
fi

python /home/gpranav/pranav_work/scratch/TRELLIS.2/export_gaussian_distance_flow_samples.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --split "$EXPORT_SPLIT" \
  --ckpt "$EXPORT_CKPT" \
  --output_dir "$RUN_DIR/$EXPORT_RUN_NAME" \
  --gaussian_distance_latent_name "$LATENT_NAME" \
  --michelangelo_latent_name "$MICHELANGELO_NAME" \
  --num_samples "$EXPORT_NUM_SAMPLES" \
  --batch_size "$EXPORT_BATCH_SIZE" \
  --sampling_steps "$EXPORT_SAMPLING_STEPS" \
  --guidance_strength "$EXPORT_GUIDANCE_STRENGTH" \
  --render_resolution "$EXPORT_RENDER_RESOLUTION" \
  --seed "$EXPORT_SEED" \
  "${EXTRA_ARGS[@]}"
