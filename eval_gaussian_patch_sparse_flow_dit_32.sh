#!/bin/bash
#SBATCH --job-name=trellis-gpatch-sparse32-eval
#SBATCH --output=./job_logs/trellis-gpatch-sparse32-eval_%j.log
#SBATCH --nodes=1
#SBATCH --partition=spgpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --account=jjparkcv_owned1
#SBATCH --gres=gpu:1

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

mkdir -p job_logs

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"

if [ -n "${1:-}" ]; then
  export RUN_DIR="$1"
  export RUN_NAME="$(basename "$RUN_DIR")"
elif [ -n "${RUN_NAME:-}" ]; then
  export RUN_DIR="$ROOT/outputs/$RUN_NAME"
else
  mapfile -t RUN_CANDIDATES < <(
    find "$ROOT/outputs" -maxdepth 1 -type d -name 'gaussian_patch_sparse_flow_dit32_*' -printf '%T@ %p\n' \
      | sort -nr \
      | awk '{print $2}'
  )
  if [ "${#RUN_CANDIDATES[@]}" -eq 0 ]; then
    echo "No gaussian_patch_sparse_flow_dit32_* run found under $ROOT/outputs."
    echo "Pass a run directory as the first argument or set RUN_NAME."
    exit 1
  fi
  export RUN_DIR="${RUN_CANDIDATES[0]}"
  export RUN_NAME="$(basename "$RUN_DIR")"
fi

if [ ! -f "$RUN_DIR/config.json" ] && [ -z "${CONFIG:-}" ]; then
  export CONFIG="configs/gen/gaussian_patch_sparse_flow_dit_32_1_3B_bf16.json"
fi

if [ ! -d "$RUN_DIR/ckpts" ]; then
  echo "Checkpoint directory not found: $RUN_DIR/ckpts"
  echo "Pass the actual training run directory or set RUN_NAME."
  exit 1
fi

echo "Evaluating run: $RUN_DIR"

export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_sparse_${EVAL_SPLIT}_${SLURM_JOB_ID:-manual}}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-8}"
export EVAL_GENERATED_SAMPLES="${EVAL_GENERATED_SAMPLES:-8}"
export EVAL_SAMPLING_STEPS="${EVAL_SAMPLING_STEPS:-50}"
export EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-256}"
export EVAL_RENDER_SSAA="${EVAL_RENDER_SSAA:-2}"
export EVAL_RECONSTRUCTION_TS="${EVAL_RECONSTRUCTION_TS:-0.1,0.3,0.5,0.7,0.9}"
export EVAL_METADATA_FILTER_CSV="${EVAL_METADATA_FILTER_CSV:-/gpfs/accounts/jjparkcv_root/jjparkcv0/gpranav/TRELLIS.2/metadata_test_no_train_duplicates.csv}"

EXTRA_ARGS=()
if [ -n "$EVAL_METADATA_FILTER_CSV" ]; then
  EXTRA_ARGS+=(--metadata_filter_csv "$EVAL_METADATA_FILTER_CSV")
fi
if [ -n "${CONFIG:-}" ]; then
  EXTRA_ARGS+=(--config "$CONFIG")
fi
if [ -n "${EVAL_EMA_RATE:-}" ]; then
  EXTRA_ARGS+=(--ema_rate "$EVAL_EMA_RATE")
fi
if [ -n "${EVAL_MAX_BATCHES:-}" ]; then
  EXTRA_ARGS+=(--max_batches "$EVAL_MAX_BATCHES")
fi

python /home/gpranav/pranav_work/scratch/TRELLIS.2/eval_gaussian_patch_sparse_flow.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --ckpt "$EVAL_CKPT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$EVAL_NUM_WORKERS" \
  --num_samples "$EVAL_NUM_SAMPLES" \
  --generated_samples "$EVAL_GENERATED_SAMPLES" \
  --reconstruction_ts "$EVAL_RECONSTRUCTION_TS" \
  --sampling_steps "$EVAL_SAMPLING_STEPS" \
  --render_resolution "$EVAL_RENDER_RESOLUTION" \
  --render_ssaa "$EVAL_RENDER_SSAA" \
  "${EXTRA_ARGS[@]}"
