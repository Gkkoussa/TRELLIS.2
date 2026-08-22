#!/usr/bin/env bash
#SBATCH --job-name=trifield-vae-qem-vattn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
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
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-triangle_field_vae_allres_qem_vertex_attention}"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_JOB_ID}}"
export VERTEX_CONFIG="${VERTEX_CONFIG:-$REPO/configs/scvae/triangle_field_vae_allres_qem_vertex_attention.json}"
export TRAIN_INSTANCES="${TRAIN_INSTANCES:-$ROOT/splits/train_triangle_field_512/instances.txt}"
export NUM_GPUS="${NUM_GPUS:-2}"

if [ ! -f "$VERTEX_CONFIG" ]; then
  echo "Missing attention vertex-prediction config: $VERTEX_CONFIG" >&2
  exit 1
fi
if [ ! -f "$TRAIN_INSTANCES" ]; then
  echo "Missing filtered training instances: $TRAIN_INSTANCES" >&2
  exit 1
fi

for RESOLUTION in 32 64 128 256 512; do
  VOXEL_DIR="$ROOT/triangle_field_voxels_${RESOLUTION}"
  QEM_DIR="$ROOT/qem_edge_collapsed_meshes_${RESOLUTION}"
  if [ ! -f "$VOXEL_DIR/metadata.csv" ]; then
    echo "Missing triangle-field metadata: $VOXEL_DIR/metadata.csv" >&2
    exit 1
  fi
  if [ ! -f "$QEM_DIR/metadata.csv" ]; then
    echo "Missing QEM metadata: $QEM_DIR/metadata.csv" >&2
    exit 1
  fi
done

OUTPUT_DIR="$ROOT/outputs/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
JOB_CONFIG="$OUTPUT_DIR/config.json"

python - "$VERTEX_CONFIG" "$JOB_CONFIG" "$TRAIN_INSTANCES" <<'PY'
import json
import sys

source, destination, train_instances = sys.argv[1:4]
with open(source, "r", encoding="utf-8") as handle:
    config = json.load(handle)
config["dataset"]["args"]["instances_path"] = train_instances
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=4)
PY

DATA_DIR="{\"objxl4k_filtered_train\":{\"triangle_field_voxel_32\":\"$ROOT/triangle_field_voxels_32\",\"triangle_field_voxel_64\":\"$ROOT/triangle_field_voxels_64\",\"triangle_field_voxel_128\":\"$ROOT/triangle_field_voxels_128\",\"triangle_field_voxel_256\":\"$ROOT/triangle_field_voxels_256\",\"triangle_field_voxel_512\":\"$ROOT/triangle_field_voxels_512\"}}"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

python "$REPO/train.py" \
  --config "$JOB_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
