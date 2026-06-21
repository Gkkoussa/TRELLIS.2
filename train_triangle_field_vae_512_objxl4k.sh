#!/bin/bash
#SBATCH --job-name=trellis-trifield-vae512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
export RUN_NAME="${RUN_NAME:-triangle_field_vae_512_ft256_${SLURM_JOB_ID}}"
export FLOW_CONFIG="${FLOW_CONFIG:-/home/koussa/scratch/TRELLIS.2/configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_512_ft_objxl4k.json}"
export BASE_SPLIT_DIR="${BASE_SPLIT_DIR:-$ROOT/splits/train_triangle_field_512}"
export TRIANGLE_FIELD_VOXEL_DIR="${TRIANGLE_FIELD_VOXEL_DIR:-$ROOT/splits/train/triangle_field_voxels_512}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

if [ ! -f "$BASE_SPLIT_DIR/metadata.csv" ]; then
  echo "Missing filtered base metadata: $BASE_SPLIT_DIR/metadata.csv" >&2
  echo "Create it from the 512 triangle-field subset before launching training." >&2
  exit 1
fi

if [ ! -f "$TRIANGLE_FIELD_VOXEL_DIR/metadata.csv" ]; then
  echo "Missing triangle-field voxel metadata: $TRIANGLE_FIELD_VOXEL_DIR/metadata.csv" >&2
  exit 1
fi

DATA_DIR="{\"objxl4k_train\":{\"base\":\"$BASE_SPLIT_DIR\",\"triangle_field_voxel\":\"$TRIANGLE_FIELD_VOXEL_DIR\"}}"

python /home/koussa/scratch/TRELLIS.2/train.py \
  --config "$FLOW_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 4 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
