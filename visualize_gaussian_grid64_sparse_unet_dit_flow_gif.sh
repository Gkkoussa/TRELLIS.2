#!/bin/bash
#SBATCH --job-name=trellis-ggrid64-unetdit-gif
#SBATCH --output=./job_logs/trellis-ggrid64-unetdit-gif_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
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

if [ -n "${1:-}" ]; then
  export RUN_DIR="$1"
  export RUN_NAME="$(basename "$RUN_DIR")"
elif [ -n "${RUN_NAME:-}" ]; then
  export RUN_DIR="$ROOT/outputs/$RUN_NAME"
else
  mapfile -t RUN_CANDIDATES < <(
    find "$ROOT/outputs" -maxdepth 1 -type d -name 'gaussian_grid64_sparse_unet_dit_flow_triangle_area_filtered_*' -printf '%T@ %p\n' \
      | sort -nr \
      | awk '{print $2}'
  )
  if [ "${#RUN_CANDIDATES[@]}" -eq 0 ]; then
    echo "No gaussian_grid64_sparse_unet_dit_flow_triangle_area_filtered_* run found under $ROOT/outputs."
    echo "Pass a run directory as the first argument or set RUN_NAME."
    exit 1
  fi
  export RUN_DIR="${RUN_CANDIDATES[0]}"
  export RUN_NAME="$(basename "$RUN_DIR")"
fi

if [ ! -f "$RUN_DIR/config.json" ] && [ -z "${CONFIG:-}" ]; then
  export CONFIG="configs/gen/gaussian_grid64_sparse_unet_dit_flow_bf16_triangle_area_filtered.json"
fi

if [ ! -d "$RUN_DIR/ckpts" ]; then
  echo "Checkpoint directory not found: $RUN_DIR/ckpts"
  echo "Pass the actual training run directory or set RUN_NAME."
  exit 1
fi

echo "Visualizing run: $RUN_DIR"

export GIF_SPLIT="${GIF_SPLIT:-test}"
export GIF_CKPT="${GIF_CKPT:-latest}"
export GIF_SAMPLE_INDEX="${GIF_SAMPLE_INDEX:-33}"
export GIF_STEPS="${GIF_STEPS:-50}"
export GIF_FRAMES="${GIF_FRAMES:-26}"
export GIF_DURATION_MS="${GIF_DURATION_MS:-160}"
export GIF_RENDER_RESOLUTION="${GIF_RENDER_RESOLUTION:-256}"
export GIF_RENDER_SSAA="${GIF_RENDER_SSAA:-2}"
export GIF_RUN_NAME="${GIF_RUN_NAME:-sampling_gif_${GIF_SPLIT}_${SLURM_JOB_ID:-manual}}"

# Default to the same non-training duplicate test subset intersected with the triangle-area filter.
export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-/home/gpranav/pranav_work/scratch/TRELLIS.2/metadata_test_no_train_duplicates.csv}"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_triangle_area_scores.csv}"
export GIF_METADATA_FILTER_CSV="${GIF_METADATA_FILTER_CSV:-$NO_TRAIN_DUPLICATE_CSV,$TRIANGLE_FILTER_CSV}"

export DATA_DIR="{\"${GIF_SPLIT}\":{\"base\":\"$ROOT/splits/${GIF_SPLIT}\",\"gaussian_distance_voxel\":\"$ROOT/gaussian_distance_voxels_64\",\"_metadata_filter_csv\":\"$GIF_METADATA_FILTER_CSV\"}}"

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

python /home/gpranav/pranav_work/scratch/TRELLIS.2/visualize_gaussian_grid64_sparse_flow_gif.py \
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
  "${EXTRA_ARGS[@]}"
