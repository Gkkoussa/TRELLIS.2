#!/usr/bin/env bash
#SBATCH --job-name=trifield-qem-dtrirefine-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

module load cuda/12.8
module load gcc/11
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

REPO=/home/gpranav/pranav_work/scratch/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_DIR="${RUN_DIR:-}"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_EMA_RATE="${EVAL_EMA_RATE:-none}"
export EVAL_PRIMARY_THRESHOLD="${EVAL_PRIMARY_THRESHOLD:-0.5}"
export EVAL_NUM_VISUALIZATIONS="${EVAL_NUM_VISUALIZATIONS:-16}"
export EVAL_NUM_STAGE_VISUALIZATIONS="${EVAL_NUM_STAGE_VISUALIZATIONS:-4}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Set RUN_DIR to a completed d_tri edge-refinement training run." >&2
  exit 2
fi

NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-$ROOT/metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv}"
TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
METADATA_FILTERS="$ROOT/splits/$EVAL_SPLIT/metadata.csv,$NO_TRAIN_DUPLICATE_CSV,$TRIANGLE_FILTER_CSV"

COMMAND=(
  python "$REPO/eval_triangle_field_qem_vertex_vae_dtri_refine.py"
  --run_dir "$RUN_DIR"
  --root "$ROOT"
  --resolution 512
  --split "$EVAL_SPLIT"
  --ckpt "$EVAL_CKPT"
  --ema_rate "$EVAL_EMA_RATE"
  --batch_size 1
  --num_workers 2
  --num_visualizations "$EVAL_NUM_VISUALIZATIONS"
  --num_stage_visualizations "$EVAL_NUM_STAGE_VISUALIZATIONS"
  --primary_threshold "$EVAL_PRIMARY_THRESHOLD"
  --image_resolution 512
  --ssaa 4
  --vertex_marker_pixels 6
  --metadata_filter_csv "$METADATA_FILTERS"
)
if [[ -n "${EVAL_MAX_SAMPLES:-}" ]]; then
  COMMAND+=(--max_eval_samples "$EVAL_MAX_SAMPLES")
fi
if [[ -n "${EVAL_INSTANCES:-}" ]]; then
  COMMAND+=(--instances "$EVAL_INSTANCES")
fi
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output_dir "$EVAL_OUTPUT_DIR")
fi

echo "Run directory: $RUN_DIR"
echo "R512 checkpoint: $EVAL_CKPT (EMA selector: $EVAL_EMA_RATE)"
"${COMMAND[@]}"

