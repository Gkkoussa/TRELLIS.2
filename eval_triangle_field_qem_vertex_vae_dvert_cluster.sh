#!/usr/bin/env bash
#SBATCH --job-name=trifield-qem-dvert-cluster
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
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export EVAL_NUM_VISUALIZATIONS="${EVAL_NUM_VISUALIZATIONS:-16}"
export EVAL_PRIMARY_THRESHOLD="${EVAL_PRIMARY_THRESHOLD:-0.5}"
export EVAL_ROLLOUT_THRESHOLD="${EVAL_ROLLOUT_THRESHOLD:-0.1}"
export EVAL_DVERT_THRESHOLD="${EVAL_DVERT_THRESHOLD:-0.8}"
export EVAL_DVERT_CONNECTIVITY="${EVAL_DVERT_CONNECTIVITY:-26}"
export EVAL_IMAGE_RESOLUTION="${EVAL_IMAGE_RESOLUTION:-512}"
export EVAL_SSAA="${EVAL_SSAA:-4}"
export EVAL_VERTEX_MARKER_PIXELS="${EVAL_VERTEX_MARKER_PIXELS:-6}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Set RUN_DIR to a triangle-field QEM vertex-prediction training folder." >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/config.json" ]]; then
  echo "Missing run config: $RUN_DIR/config.json" >&2
  exit 1
fi
if [[ ! -f "$ROOT/triangle_field_voxels_512/metadata.csv" && \
      ! -f "$ROOT/splits/$EVAL_SPLIT/triangle_field_voxels_512/metadata.csv" ]]; then
  echo "Missing R512 triangle-field metadata." >&2
  exit 1
fi
if [[ ! -f "$ROOT/qem_edge_collapsed_meshes_512/metadata.csv" ]]; then
  echo "Missing R512 QEM metadata." >&2
  exit 1
fi

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
  python "$REPO/eval_triangle_field_qem_vertex_vae_dvert_cluster.py"
  --run_dir "$RUN_DIR"
  --root "$ROOT"
  --split "$EVAL_SPLIT"
  --ckpt "$EVAL_CKPT"
  --ema_rate "$EVAL_EMA_RATE"
  --batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$EVAL_NUM_WORKERS"
  --num_visualizations "$EVAL_NUM_VISUALIZATIONS"
  --primary_threshold "$EVAL_PRIMARY_THRESHOLD"
  --rollout_threshold "$EVAL_ROLLOUT_THRESHOLD"
  --dvert_threshold "$EVAL_DVERT_THRESHOLD"
  --connectivity "$EVAL_DVERT_CONNECTIVITY"
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
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output_dir "$EVAL_OUTPUT_DIR")
fi
if [[ "${EVAL_SAMPLE_POSTERIOR:-0}" == "1" ]]; then
  COMMAND+=(--sample_posterior)
fi

echo "Training run: $RUN_DIR"
echo "Checkpoint: $EVAL_CKPT (EMA $EVAL_EMA_RATE)"
echo "Evaluation: R512 $EVAL_SPLIT"
echo "Vertex threshold: $EVAL_PRIMARY_THRESHOLD"
echo "R512 child constraint: unlimited (no top-k at any stage)"
echo "d_vert clustering: > $EVAL_DVERT_THRESHOLD with $EVAL_DVERT_CONNECTIVITY-connectivity"
echo "Metadata filters: $EVAL_METADATA_FILTER_CSV"
echo "Default output: an eval_qem_vertex_dvert_cluster_* folder inside RUN_DIR"

"${COMMAND[@]}"
