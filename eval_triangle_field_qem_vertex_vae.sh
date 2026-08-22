#!/usr/bin/env bash
#SBATCH --job-name=trifield-qem-vsub-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/%x-%A_%a.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2
#SBATCH --array=0-4%5

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
export EVAL_EMA_RATE="${EVAL_EMA_RATE:-0.9999}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export EVAL_NUM_VISUALIZATIONS="${EVAL_NUM_VISUALIZATIONS:-16}"
export EVAL_NUM_STAGE_VISUALIZATIONS="${EVAL_NUM_STAGE_VISUALIZATIONS:-4}"
export EVAL_PRIMARY_THRESHOLD="${EVAL_PRIMARY_THRESHOLD:-0.5}"
export EVAL_IMAGE_RESOLUTION="${EVAL_IMAGE_RESOLUTION:-512}"
export EVAL_SSAA="${EVAL_SSAA:-4}"
export EVAL_VERTEX_MARKER_PIXELS="${EVAL_VERTEX_MARKER_PIXELS:-6}"
export EVAL_FINAL_CHILDREN_PER_PARENT="${EVAL_FINAL_CHILDREN_PER_PARENT:-}"
export EVAL_CHILD_LIMIT_STAGES="${EVAL_CHILD_LIMIT_STAGES:-3}"

RESOLUTIONS=(32 64 128 256 512)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#RESOLUTIONS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 4; got $TASK_INDEX" >&2
  exit 2
fi
RESOLUTION="${RESOLUTIONS[$TASK_INDEX]}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Set RUN_DIR to a completed triangle-field QEM vertex-prediction run." >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/config.json" ]]; then
  echo "Missing run config: $RUN_DIR/config.json" >&2
  exit 1
fi
if [[ ! -f "$ROOT/triangle_field_voxels_${RESOLUTION}/metadata.csv" && \
      ! -f "$ROOT/splits/$EVAL_SPLIT/triangle_field_voxels_${RESOLUTION}/metadata.csv" ]]; then
  echo "Missing triangle-field metadata for resolution $RESOLUTION" >&2
  exit 1
fi
if [[ ! -f "$ROOT/qem_edge_collapsed_meshes_${RESOLUTION}/metadata.csv" ]]; then
  echo "Missing QEM metadata for resolution $RESOLUTION" >&2
  exit 1
fi

# Match the standard VAE evaluation filtering policy. The loader intersects all
# CSVs by sha256; their row ordering is irrelevant.
export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-$ROOT/metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv}"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
for FILTER_PATH in \
  "$ROOT/splits/$EVAL_SPLIT/metadata.csv" \
  "$NO_TRAIN_DUPLICATE_CSV" \
  "$TRIANGLE_FILTER_CSV"; do
  if [[ ! -f "$FILTER_PATH" ]]; then
    echo "Missing evaluation metadata filter: $FILTER_PATH" >&2
    exit 1
  fi
done
export EVAL_METADATA_FILTER_CSV="${EVAL_METADATA_FILTER_CSV:-$ROOT/splits/$EVAL_SPLIT/metadata.csv,$NO_TRAIN_DUPLICATE_CSV,$TRIANGLE_FILTER_CSV}"

export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

COMMAND=(
  python "$REPO/eval_triangle_field_qem_vertex_vae.py"
  --run_dir "$RUN_DIR"
  --root "$ROOT"
  --resolution "$RESOLUTION"
  --split "$EVAL_SPLIT"
  --ckpt "$EVAL_CKPT"
  --ema_rate "$EVAL_EMA_RATE"
  --batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$EVAL_NUM_WORKERS"
  --num_visualizations "$EVAL_NUM_VISUALIZATIONS"
  --num_stage_visualizations "$EVAL_NUM_STAGE_VISUALIZATIONS"
  --primary_threshold "$EVAL_PRIMARY_THRESHOLD"
  --image_resolution "$EVAL_IMAGE_RESOLUTION"
  --ssaa "$EVAL_SSAA"
  --vertex_marker_pixels "$EVAL_VERTEX_MARKER_PIXELS"
  --metadata_filter_csv "$EVAL_METADATA_FILTER_CSV"
)

if [[ -n "${EVAL_INSTANCES:-}" ]]; then
  COMMAND+=(--instances "$EVAL_INSTANCES")
fi
if [[ -n "${EVAL_MAX_SAMPLES:-}" ]]; then
  COMMAND+=(--max_eval_samples "$EVAL_MAX_SAMPLES")
fi
if [[ -n "$EVAL_FINAL_CHILDREN_PER_PARENT" ]]; then
  COMMAND+=(--final_children_per_parent "$EVAL_FINAL_CHILDREN_PER_PARENT")
  NORMALIZED_CHILD_LIMIT_STAGES="${EVAL_CHILD_LIMIT_STAGES//,/ }"
  read -r -a CHILD_LIMIT_STAGE_ARGS <<< "$NORMALIZED_CHILD_LIMIT_STAGES"
  if (( ${#CHILD_LIMIT_STAGE_ARGS[@]} == 0 )); then
    echo "EVAL_CHILD_LIMIT_STAGES must specify at least one stage." >&2
    exit 2
  fi
  COMMAND+=(--child_limit_stages "${CHILD_LIMIT_STAGE_ARGS[@]}")
fi
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output_dir "$EVAL_OUTPUT_DIR/resolution_${RESOLUTION}")
fi
if [[ "${EVAL_SAMPLE_POSTERIOR:-0}" == "1" ]]; then
  COMMAND+=(--sample_posterior)
fi

echo "Run directory: $RUN_DIR"
echo "Split: $EVAL_SPLIT"
echo "Resolution: $RESOLUTION"
echo "Checkpoint: $EVAL_CKPT (EMA $EVAL_EMA_RATE)"
echo "Metadata filters: $EVAL_METADATA_FILTER_CSV"
echo "Primary probability threshold: $EVAL_PRIMARY_THRESHOLD"
echo "Visualizations: $EVAL_NUM_VISUALIZATIONS"
echo "Stage-detail visualizations: $EVAL_NUM_STAGE_VISUALIZATIONS"
echo "Minimum vertex marker width: $EVAL_VERTEX_MARKER_PIXELS pixels"
if [[ -n "$EVAL_FINAL_CHILDREN_PER_PARENT" && "$RESOLUTION" == "512" ]]; then
  echo "R512 child constraint: top-$EVAL_FINAL_CHILDREN_PER_PARENT at stages ${EVAL_CHILD_LIMIT_STAGES//,/ }"
else
  echo "R512 child constraint: unlimited"
fi

"${COMMAND[@]}"
