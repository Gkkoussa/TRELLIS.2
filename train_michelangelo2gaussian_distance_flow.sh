#!/bin/bash
#SBATCH --job-name=trellis-gdist-flow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
export LATENT_NAME=gaussian_distance_vae_step0350000_256
export MICHELANGELO_NAME=shapevae256_pretrained
export RUN_NAME="${RUN_NAME:-michelangelo2gaussian_distance_flow_${SLURM_JOB_ID}}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES=${TRELLIS_DIST_TIMEOUT_MINUTES:-60}

DATA_DIR="{\"neuframe_train\":{\"metadata\":\"$ROOT/splits/train\",\"gaussian_distance_latent\":\"$ROOT/splits/train/gaussian_distance_latents/$LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/splits/train/michelangelo_latents/$MICHELANGELO_NAME\"}}"

python /home/koussa/scratch/TRELLIS.2/train.py \
  --config /home/koussa/scratch/TRELLIS.2/configs/gen/slat_flow_michelangelo2gaussian_distance_dit_1_3B_256_bf16.json \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 4 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
