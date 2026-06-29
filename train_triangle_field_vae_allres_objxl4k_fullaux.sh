#!/bin/bash
#SBATCH --job-name=trellis-trifield-vae-allres-fullaux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-triangle_field_vae_allres_invarea_fullaux}"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_JOB_ID}}"
export FLOW_CONFIG="${FLOW_CONFIG:-/home/koussa/scratch/TRELLIS.2/configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_allres_objxl4k_invarea_fullaux.json}"
export TRAIN_INSTANCES="${TRAIN_INSTANCES:-$ROOT/splits/train_triangle_field_512/instances.txt}"
export NUM_GPUS="${NUM_GPUS:-2}"

mkdir -p "$ROOT/outputs/$RUN_NAME"
JOB_CONFIG="$ROOT/outputs/$RUN_NAME/config.json"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

if [ ! -f "$TRAIN_INSTANCES" ]; then
  echo "Missing filtered train instances: $TRAIN_INSTANCES" >&2
  exit 1
fi

for RESOLUTION in 32 64 128 256 512; do
  VOXEL_DIR="$ROOT/triangle_field_voxels_${RESOLUTION}"
  if [ ! -f "$VOXEL_DIR/metadata.csv" ]; then
    echo "Missing triangle-field voxel metadata: $VOXEL_DIR/metadata.csv" >&2
    exit 1
  fi
done

DATA_DIR="{\"objxl4k_filtered_train\":{\"triangle_field_voxel_32\":\"$ROOT/triangle_field_voxels_32\",\"triangle_field_voxel_64\":\"$ROOT/triangle_field_voxels_64\",\"triangle_field_voxel_128\":\"$ROOT/triangle_field_voxels_128\",\"triangle_field_voxel_256\":\"$ROOT/triangle_field_voxels_256\",\"triangle_field_voxel_512\":\"$ROOT/triangle_field_voxels_512\"}}"

python - "$FLOW_CONFIG" "$JOB_CONFIG" "$TRAIN_INSTANCES" <<'PY'
import json
import sys

src, dst, train_instances = sys.argv[1:4]
with open(src, 'r') as f:
    cfg = json.load(f)
cfg['dataset']['args']['instances_path'] = train_instances
with open(dst, 'w') as f:
    json.dump(cfg, f, indent=4)
PY

python /home/koussa/scratch/TRELLIS.2/train.py \
  --config "$JOB_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
