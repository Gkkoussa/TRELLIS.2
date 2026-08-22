#!/bin/bash
#SBATCH --job-name=trifield-flow-filtered
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export TRIANGLE_FIELD_LATENT_NAME="${TRIANGLE_FIELD_LATENT_NAME:-triangle_field_vae_51685536_step0100000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export SHAPE_LATENT_NAME="${SHAPE_LATENT_NAME:-occupancy_shape_vae_triangle_filtered_51728720_step0100000_256}"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
export FLOW_CONFIG="${FLOW_CONFIG:-/home/gpranav/pranav_work/scratch/TRELLIS.2/configs/gen/slat_flow_michelangelo_shape2triangle_field_filtered_dit_1_3B_256_bf16.json}"
export RUN_NAME="${RUN_NAME:-michelangelo_shape2triangle_field_flow_filtered_${SLURM_JOB_ID}}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"

DATA_DIR="{\"train\":{\"metadata\":\"$ROOT/splits/train\",\"triangle_field_latent\":\"$ROOT/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/michelangelo_latents/$MICHELANGELO_NAME\",\"shape_latent\":\"$ROOT/shape_latents/$SHAPE_LATENT_NAME\",\"_metadata_filter_csv\":\"$ROOT/splits/train/metadata.csv,$TRIANGLE_FILTER_CSV\"}}"

python /home/gpranav/pranav_work/scratch/TRELLIS.2/train.py \
  --config "$FLOW_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 8 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
