#!/bin/bash
#SBATCH --job-name=trellis-trifield-flow-shape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
export TRIANGLE_FIELD_LATENT_NAME="${TRIANGLE_FIELD_LATENT_NAME:-triangle_field_vae_51483691_step0060000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export SHAPE_LATENT_NAME="${SHAPE_LATENT_NAME:-occupancy_shape_vae_step0110000_256}"
export FLOW_CONFIG="${FLOW_CONFIG:-/home/koussa/scratch/TRELLIS.2/configs/gen/slat_flow_michelangelo_shape2triangle_field_dit_1_3B_256_bf16.json}"
export RUN_NAME="${RUN_NAME:-michelangelo_shape2triangle_field_flow_${SLURM_JOB_ID}}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"

DATA_DIR="{\"neuframe_train\":{\"metadata\":\"$ROOT/splits/train\",\"triangle_field_latent\":\"$ROOT/splits/train/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/splits/train/michelangelo_latents/$MICHELANGELO_NAME\",\"shape_latent\":\"$ROOT/splits/train/shape_latents/$SHAPE_LATENT_NAME\"}}"

python /home/koussa/scratch/TRELLIS.2/train.py \
  --config "$FLOW_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 8 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
