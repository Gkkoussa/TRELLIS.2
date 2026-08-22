#!/bin/bash
#SBATCH --job-name=trellis-trifield-latent-sr-allres-denelong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=48:00:00
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
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-triangle_field_latent_sr_flow_allres_selfcond_input_fullaux_noiseonly_density_elongation_cfgmix_from64to128}"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_JOB_ID}}"
export FLOW_CONFIG="${FLOW_CONFIG:-$PWD/configs/gen/triangle_field_latent_sr_flow_allres_film_selfcond_input_fullaux_noiseonly_density_elongation_cfgmix_f16c32_fp16_objxl4k.json}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CKPT="${CKPT:-none}"
export LOAD_DIR="${LOAD_DIR:-}"
export TRIANGLE_FIELD_VOXEL_TEMPLATE="${TRIANGLE_FIELD_VOXEL_TEMPLATE:-$ROOT/triangle_field_voxels_{resolution}}"
export TRIANGLE_FIELD_LATENT_TEMPLATE="${TRIANGLE_FIELD_LATENT_TEMPLATE:-$ROOT/triangle_field_latents/triangle_field_vae_allres_invarea_fullaux_52454251_step0320000_{resolution}}"
export DENSITY_TRIANGLE_FIELD_VOXEL_TEMPLATE="${DENSITY_TRIANGLE_FIELD_VOXEL_TEMPLATE:-$ROOT/triangle_field_voxels_density_elongation_field/triangle_field_voxels_{resolution}}"
export INSTANCES_PATH="${INSTANCES_PATH:-$ROOT/splits/train_triangle_field_512/instances.txt}"

for resolution in 32 64 128 256 512; do
  for template in "$TRIANGLE_FIELD_VOXEL_TEMPLATE" "$TRIANGLE_FIELD_LATENT_TEMPLATE" "$DENSITY_TRIANGLE_FIELD_VOXEL_TEMPLATE"; do
    directory="${template//\{resolution\}/$resolution}"
    if [ ! -f "$directory/metadata.csv" ]; then
      echo "Missing metadata: $directory/metadata.csv" >&2
      exit 1
    fi
  done
done
if [ ! -f "$INSTANCES_PATH" ]; then
  echo "Missing filtered train split: $INSTANCES_PATH" >&2
  exit 1
fi

mkdir -p "$ROOT/outputs/$RUN_NAME"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

DATA_DIR="{\"objxl4k_filtered\":{\"triangle_field_voxel\":\"$TRIANGLE_FIELD_VOXEL_TEMPLATE\",\"triangle_field_latent\":\"$TRIANGLE_FIELD_LATENT_TEMPLATE\",\"density_triangle_field_voxel\":\"$DENSITY_TRIANGLE_FIELD_VOXEL_TEMPLATE\"}}"

LOAD_ARGS=()
if [ -n "$LOAD_DIR" ]; then
  LOAD_ARGS+=(--load_dir "$LOAD_DIR")
fi

python train.py \
  --config "$FLOW_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  "${LOAD_ARGS[@]}" \
  --ckpt "$CKPT" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
