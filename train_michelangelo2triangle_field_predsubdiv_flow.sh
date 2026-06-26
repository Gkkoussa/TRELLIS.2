#!/bin/bash
#SBATCH --job-name=trellis-trifield-flow-predsub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
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
export TRIANGLE_FIELD_LATENT_NAME="${TRIANGLE_FIELD_LATENT_NAME:-triangle_field_vae_256_predsubdiv_c64_52039264_step0150000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export FLOW_CONFIG="${FLOW_CONFIG:-/home/koussa/scratch/TRELLIS.2/configs/gen/slat_flow_michelangelo2triangle_field_predsubdiv_c64_dit_small_256_bf16_objxl4k.json}"
export RUN_NAME="${RUN_NAME:-michelangelo2triangle_field_predsubdiv_c64_flow_small_${SLURM_JOB_ID}}"
export NUM_GPUS="${NUM_GPUS:-1}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"

DATA_DIR="{\"objxl4k_train\":{\"metadata\":\"$ROOT/splits/train\",\"triangle_field_latent\":\"$ROOT/splits/train/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/splits/train/michelangelo_latents/$MICHELANGELO_NAME\"}}"

python /home/koussa/scratch/TRELLIS.2/train.py \
  --config "$FLOW_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
