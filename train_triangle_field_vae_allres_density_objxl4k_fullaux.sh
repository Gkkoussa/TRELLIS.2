#!/bin/bash
#SBATCH --job-name=trellis-trifield-vae-allres-density
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
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-triangle_field_vae_allres_density_fullaux_noareaweight}"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_JOB_ID}}"
export VAE_CONFIG="${VAE_CONFIG:-$PWD/configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_allres_objxl4k_density_fullaux_noareaweight.json}"
export TRAIN_INSTANCES="${TRAIN_INSTANCES:-$ROOT/splits/train_triangle_field_512/instances.txt}"
export VOXEL_TEMPLATE="${VOXEL_TEMPLATE:-$ROOT/triangle_field_voxels_density_elongation_field/triangle_field_voxels_{resolution}}"
export NUM_GPUS="${NUM_GPUS:-2}"

if [ ! -f "$TRAIN_INSTANCES" ]; then
  echo "Missing filtered train instances: $TRAIN_INSTANCES" >&2
  exit 1
fi

for resolution in 32 64 128 256 512; do
  voxel_dir="${VOXEL_TEMPLATE//\{resolution\}/$resolution}"
  if [ ! -f "$voxel_dir/metadata.csv" ]; then
    echo "Missing density triangle-field metadata: $voxel_dir/metadata.csv" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/outputs/$RUN_NAME"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

DATA_DIR="{\"objxl4k_filtered_train\":{\"triangle_field_voxel\":\"$VOXEL_TEMPLATE\"}}"

python train.py \
  --config "$VAE_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
