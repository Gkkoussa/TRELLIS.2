#!/bin/bash
#SBATCH --job-name=trifield-flow-gif
#SBATCH --output=./triangle_job_logs/trifield-flow-gif_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gres=gpu:1

set -euo pipefail

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

mkdir -p triangle_job_logs

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"

if [ -n "${1:-}" ]; then
  export RUN_DIR="$1"
  export RUN_NAME="$(basename "$RUN_DIR")"
elif [ -n "${RUN_NAME:-}" ]; then
  export RUN_DIR="$ROOT/outputs/$RUN_NAME"
else
  mapfile -t RUN_CANDIDATES < <(
    find "$ROOT/outputs" -maxdepth 1 -type d -name 'michelangelo_shape2triangle_field_flow_filtered_*' -printf '%T@ %p\n' \
      | sort -nr \
      | awk '{print $2}'
  )
  if [ "${#RUN_CANDIDATES[@]}" -eq 0 ]; then
    echo "No michelangelo_shape2triangle_field_flow_filtered_* run found under $ROOT/outputs."
    echo "Pass a run directory as the first argument or set RUN_NAME."
    exit 1
  fi
  export RUN_DIR="${RUN_CANDIDATES[0]}"
  export RUN_NAME="$(basename "$RUN_DIR")"
fi

if [ ! -f "$RUN_DIR/config.json" ] && [ -z "${CONFIG:-}" ]; then
  export CONFIG="configs/gen/slat_flow_michelangelo_shape2triangle_field_filtered_dit_1_3B_256_bf16.json"
fi

if [ ! -d "$RUN_DIR/ckpts" ]; then
  echo "Checkpoint directory not found: $RUN_DIR/ckpts"
  echo "Pass the actual training run directory or set RUN_NAME."
  exit 1
fi

echo "Visualizing run: $RUN_DIR"

export GIF_SPLIT="${GIF_SPLIT:-test}"
export GIF_CKPT="${GIF_CKPT:-latest}"
export GIF_SAMPLE_INDEX="${GIF_SAMPLE_INDEX:-0}"
export GIF_STEPS="${GIF_STEPS:-50}"
export GIF_FRAMES="${GIF_FRAMES:-26}"
export GIF_DURATION_MS="${GIF_DURATION_MS:-160}"
export GIF_RENDER_RESOLUTION="${GIF_RENDER_RESOLUTION:-256}"
export GIF_RENDER_SSAA="${GIF_RENDER_SSAA:-2}"
export GIF_GUIDANCE_STRENGTH="${GIF_GUIDANCE_STRENGTH:-1.0}"
export GIF_RUN_NAME="${GIF_RUN_NAME:-sampling_gif_${GIF_SPLIT}_${SLURM_JOB_ID:-manual}}"

export TRIANGLE_FIELD_LATENT_NAME="${TRIANGLE_FIELD_LATENT_NAME:-triangle_field_vae_51685536_step0100000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export SHAPE_LATENT_NAME="${SHAPE_LATENT_NAME:-occupancy_shape_vae_triangle_filtered_51728720_step0100000_256}"

# Default to the no-train duplicate test subset intersected with the triangle-area filter.
export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-/home/gpranav/pranav_work/scratch/TRELLIS.2/metadata_test_no_train_duplicates.csv}"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
export GIF_METADATA_FILTER_CSV="${GIF_METADATA_FILTER_CSV:-$ROOT/splits/${GIF_SPLIT}/metadata.csv,$NO_TRAIN_DUPLICATE_CSV,$TRIANGLE_FILTER_CSV}"

export DATA_DIR="{\"${GIF_SPLIT}\":{\"metadata\":\"$ROOT/splits/${GIF_SPLIT}\",\"triangle_field_latent\":\"$ROOT/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/michelangelo_latents/$MICHELANGELO_NAME\",\"shape_latent\":\"$ROOT/shape_latents/$SHAPE_LATENT_NAME\",\"_metadata_filter_csv\":\"$GIF_METADATA_FILTER_CSV\"}}"

EXTRA_ARGS=()
if [ -n "${CONFIG:-}" ]; then
  EXTRA_ARGS+=(--config "$CONFIG")
fi
if [ -n "${GIF_EMA_RATE:-}" ]; then
  EXTRA_ARGS+=(--ema_rate "$GIF_EMA_RATE")
fi
if [ "${GIF_NO_GT:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no_gt)
fi

python /home/gpranav/pranav_work/scratch/TRELLIS.2/visualize_triangle_field_flow_gif.py \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --split "$GIF_SPLIT" \
  --ckpt "$GIF_CKPT" \
  --output_dir "$RUN_DIR/$GIF_RUN_NAME" \
  --sample_index "$GIF_SAMPLE_INDEX" \
  --steps "$GIF_STEPS" \
  --frames "$GIF_FRAMES" \
  --duration_ms "$GIF_DURATION_MS" \
  --render_resolution "$GIF_RENDER_RESOLUTION" \
  --render_ssaa "$GIF_RENDER_SSAA" \
  --guidance_strength "$GIF_GUIDANCE_STRENGTH" \
  "${EXTRA_ARGS[@]}"
