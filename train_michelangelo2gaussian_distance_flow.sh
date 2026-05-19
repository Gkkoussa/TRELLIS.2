#!/bin/bash
#SBATCH --job-name=trellis-gdist-flow
#SBATCH --output=./job_logs/trellis-gdist-flow_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gres=gpu:6

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

mkdir -p job_logs

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
export LATENT_NAME="${LATENT_NAME:-gaussian_distance_vae_512_step0220000_512}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"

# Resume the in-progress 4-GPU run (step/optimizer/EMA from latest misc_*.pt).
# Override for a fresh run: RUN_NAME=michelangelo2gaussian_distance_flow_${SLURM_JOB_ID}
export RUN_NAME="${RUN_NAME:-michelangelo2gaussian_distance_flow_50391241}"
export RUN_DIR="$ROOT/outputs/$RUN_NAME"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((12000 + RANDOM % 20000))
export TRELLIS_DIST_TIMEOUT_MINUTES=${TRELLIS_DIST_TIMEOUT_MINUTES:-60}

export DATA_DIR="{\"train\":{\"metadata\":\"$ROOT/splits/train\",\"gaussian_distance_latent\":\"$ROOT/splits/train/gaussian_distance_latents/$LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/splits/train/michelangelo_latents/$MICHELANGELO_NAME\"}}"

# batch_size_per_gpu=4 x 8 GPUs = global batch 32 (same as 8 x 4 GPUs before)
export CONFIG="${CONFIG:-configs/gen/slat_flow_michelangelo2gaussian_distance_dit_1_3B_512_bf16_8gpu.json}"
export CKPT="${CKPT:-latest}"

mkdir -p "$RUN_DIR"

python train.py \
  --config "$CONFIG" \
  --output_dir "$RUN_DIR" \
  --load_dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 6 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
