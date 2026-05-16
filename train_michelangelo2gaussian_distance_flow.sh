#!/bin/bash
#SBATCH --job-name=trellis-gdist-flow
#SBATCH --output=./job_logs/trellis-gdist-flow_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gres=gpu:4

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

mkdir -p job_logs

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
export LATENT_NAME=gaussian_distance_vae_step0230000_256
export MICHELANGELO_NAME=shapevae256_pretrained
export RUN_NAME="${RUN_NAME:-michelangelo2gaussian_distance_flow_${SLURM_JOB_ID}}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((12000 + RANDOM % 20000))
export TRELLIS_DIST_TIMEOUT_MINUTES=${TRELLIS_DIST_TIMEOUT_MINUTES:-60}

export DATA_DIR="{\"train\":{\"metadata\":\"$ROOT/splits/train\",\"gaussian_distance_latent\":\"$ROOT/splits/train/gaussian_distance_latents/$LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/splits/train/michelangelo_latents/$MICHELANGELO_NAME\"}}"

python train.py \
  --config configs/gen/slat_flow_michelangelo2gaussian_distance_dit_1_3B_256_bf16.json \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 4 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
