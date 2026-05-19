#!/bin/bash
#SBATCH --job-name=train_gaussian_distance_vae
#SBATCH --output=./job_logs/train_gaussian_distance_vae_%j.log
#SBATCH --nodes=1
#SBATCH --partition=spgpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --account=jjparkcv_owned1
#SBATCH --gres=gpu:4

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

mkdir -p job_logs

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"

export DATA_DIR="{\"train\":{\"base\":\"$ROOT/splits/train\",\"gaussian_distance_voxel\":\"$ROOT/splits/train/gaussian_distance_voxels_256\"}}"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((12000 + RANDOM % 20000))

python train.py \
  --config configs/scvae/gaussian_distance_vae_next_dc_f16c32_fp16.json \
  --output_dir "$ROOT/outputs/gaussian_distance_vae_manual" \
  --data_dir "$DATA_DIR" \
  --num_gpus 4 \
  --auto_retry 0 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT"