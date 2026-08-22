#!/usr/bin/env bash
#SBATCH --job-name=trifield-qem-vedge-clique
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
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/triangle_field_vae_allres_qem_vertex_edge_57883041}"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_EMA_RATE="${EVAL_EMA_RATE:-none}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
export EVAL_NUM_VISUALIZATIONS="${EVAL_NUM_VISUALIZATIONS:-16}"
export EVAL_VERTEX_THRESHOLD="${EVAL_VERTEX_THRESHOLD:-0.5}"
export EVAL_ROLLOUT_THRESHOLD="${EVAL_ROLLOUT_THRESHOLD:-0.1}"
export EVAL_EDGE_THRESHOLD="${EVAL_EDGE_THRESHOLD:-0.8}"
export EVAL_DVERT_THRESHOLD="${EVAL_DVERT_THRESHOLD:-0.8}"
export EVAL_CLUSTER_CONNECTIVITY="${EVAL_CLUSTER_CONNECTIVITY:-26}"
export EVAL_EDGE_PAIR_BATCH_SIZE="${EVAL_EDGE_PAIR_BATCH_SIZE:-16384}"
export EVAL_IMAGE_RESOLUTION="${EVAL_IMAGE_RESOLUTION:-512}"
export EVAL_SSAA="${EVAL_SSAA:-4}"
export EVAL_VERTEX_MARKER_PIXELS="${EVAL_VERTEX_MARKER_PIXELS:-6}"
export EVAL_OBJ_UP_AXIS="${EVAL_OBJ_UP_AXIS:-y}"

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
  python "$REPO/eval_triangle_field_qem_vertex_edge_clique_mesh.py"
  --run_dir "$RUN_DIR"
  --root "$ROOT"
  --split "$EVAL_SPLIT"
  --ckpt "$EVAL_CKPT"
  --ema_rate "$EVAL_EMA_RATE"
  --batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$EVAL_NUM_WORKERS"
  --num_visualizations "$EVAL_NUM_VISUALIZATIONS"
  --vertex_threshold "$EVAL_VERTEX_THRESHOLD"
  --rollout_threshold "$EVAL_ROLLOUT_THRESHOLD"
  --edge_threshold "$EVAL_EDGE_THRESHOLD"
  --dvert_threshold "$EVAL_DVERT_THRESHOLD"
  --cluster_connectivity "$EVAL_CLUSTER_CONNECTIVITY"
  --edge_pair_batch_size "$EVAL_EDGE_PAIR_BATCH_SIZE"
  --image_resolution "$EVAL_IMAGE_RESOLUTION"
  --ssaa "$EVAL_SSAA"
  --vertex_marker_pixels "$EVAL_VERTEX_MARKER_PIXELS"
  --obj_up_axis "$EVAL_OBJ_UP_AXIS"
  --metadata_filter_csv "$EVAL_METADATA_FILTER_CSV"
)

if [[ -n "${EVAL_INSTANCES:-}" ]]; then
  COMMAND+=(--instances "$EVAL_INSTANCES")
fi
if [[ -n "$EVAL_MAX_SAMPLES" ]]; then
  COMMAND+=(--max_eval_samples "$EVAL_MAX_SAMPLES")
fi
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  COMMAND+=(--output_dir "$EVAL_OUTPUT_DIR")
fi
if [[ "${EVAL_SAMPLE_POSTERIOR:-0}" == "1" ]]; then
  COMMAND+=(--sample_posterior)
fi

echo "Run directory: $RUN_DIR"
echo "Checkpoint: $EVAL_CKPT (EMA $EVAL_EMA_RATE)"
echo "Evaluation: R512 $EVAL_SPLIT"
echo "Vertex threshold: $EVAL_VERTEX_THRESHOLD (rollout $EVAL_ROLLOUT_THRESHOLD)"
echo "Edge threshold: $EVAL_EDGE_THRESHOLD"
echo "Post-edge d_vert centroid merge: threshold $EVAL_DVERT_THRESHOLD; connectivity $EVAL_CLUSTER_CONNECTIVITY"
echo "Primary OBJ up axis: $EVAL_OBJ_UP_AXIS (canonical Z-up copies are also saved)"
echo "Edge candidates: every unordered pair of predicted final vertices"
echo "Pair-scoring GPU chunk size: $EVAL_EDGE_PAIR_BATCH_SIZE"
echo "Mesh cleanup: remove collapsed self-edges, duplicate edges, and duplicate faces"
echo "Mesh faces: every unique 3-clique in the centroid-remapped graph"
echo "Metadata filters: $EVAL_METADATA_FILTER_CSV"

"${COMMAND[@]}"
