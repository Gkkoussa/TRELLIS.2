#!/bin/bash
#SBATCH --job-name=trellis-point-density-flow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

module load gcc/11.2.0 cuda/12.8.1
eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export HUNYUAN3D_SHAPE_ROOT="${HUNYUAN3D_SHAPE_ROOT:-/home/koussa/scratch/Hunyuan3D-2.1/hy3dshape}"
export FIELD_NAME="${FIELD_NAME:-density}"
export CONFIG="${CONFIG:-$PWD/configs/gen/point_${FIELD_NAME}_flow_hunyuan_dit8_512_bf16_objxl4k.json}"
export RUN_NAME="${RUN_NAME:-point_${FIELD_NAME}_flow_hunyuan_dit8_512_${SLURM_JOB_ID}}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CKPT="${CKPT:-latest}"
export LOAD_DIR="${LOAD_DIR:-}"
export TRAIN_FILTER_CSV="${TRAIN_FILTER_CSV:-$ROOT/splits/train_triangle_field_512/metadata.csv}"
export EXTRA_TRAIN_ROOT="${EXTRA_TRAIN_ROOT:-}"
export EXTRA_TRAIN_FILTER_CSV="${EXTRA_TRAIN_FILTER_CSV:-}"
export TEST_FILTER_CSV="${TEST_FILTER_CSV:-$ROOT/splits/test_triangle_field_512/metadata.csv}"
export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-$ROOT/metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv}"
export POINT_DENSITY_STATS="${POINT_DENSITY_STATS:-$ROOT/point_density_stats/${FIELD_NAME}512_filtered_train.json}"

for path in "$CONFIG" "$TRAIN_FILTER_CSV" "$TEST_FILTER_CSV" "$NO_TRAIN_DUPLICATE_CSV" "$POINT_DENSITY_STATS"; do
  if [ ! -f "$path" ]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if [ -n "$EXTRA_TRAIN_ROOT" ] || [ -n "$EXTRA_TRAIN_FILTER_CSV" ]; then
  if [ -z "$EXTRA_TRAIN_ROOT" ] || [ -z "$EXTRA_TRAIN_FILTER_CSV" ]; then
    echo "EXTRA_TRAIN_ROOT and EXTRA_TRAIN_FILTER_CSV must be set together." >&2
    exit 1
  fi
  for path in "$EXTRA_TRAIN_ROOT/metadata.csv" "$EXTRA_TRAIN_FILTER_CSV"; do
    if [ ! -f "$path" ]; then
      echo "Missing required extra training file: $path" >&2
      exit 1
    fi
  done
fi
python -c "from pathlib import Path; from einops import rearrange; import torch_cluster; from torch_cluster import fps; from trellis2.models.point_density_flow import HunyuanPointDensityFlowModel; assert list(Path(torch_cluster.__file__).parent.glob('_fps_cuda*'))" || {
  echo "The trellis2 environment is missing a Hunyuan dependency or CUDA-enabled torch_cluster." >&2
  exit 1
}

OUTPUT_DIR="$ROOT/outputs/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
LOAD_DIR="${LOAD_DIR:-$OUTPUT_DIR}"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

TRAIN_DATA_DIR="{\"objxl4k_filtered_train\":{\"mesh\":\"$ROOT\",\"_metadata_filter_csv\":\"$TRAIN_FILTER_CSV\"}}"
if [ -n "$EXTRA_TRAIN_ROOT" ]; then
  TRAIN_DATA_DIR="{\"objxl4k_filtered_train\":{\"mesh\":\"$ROOT\",\"_metadata_filter_csv\":\"$TRAIN_FILTER_CSV\"},\"additional_meshes\":{\"mesh\":\"$EXTRA_TRAIN_ROOT\",\"_metadata_filter_csv\":\"$EXTRA_TRAIN_FILTER_CSV\"}}"
fi
VALIDATION_DATA_DIR="{\"objxl4k_no_duplicate_filtered_test\":{\"mesh\":\"$ROOT\",\"_metadata_filter_csv\":\"$TEST_FILTER_CSV,$NO_TRAIN_DUPLICATE_CSV\"}}"

python train.py \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --load_dir "$LOAD_DIR" \
  --ckpt "$CKPT" \
  --data_dir "$TRAIN_DATA_DIR" \
  --validation_data_dir "$VALIDATION_DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
