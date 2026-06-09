#!/bin/bash
#SBATCH --job-name=trellis-trifield-vae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
export RUN_NAME="${RUN_NAME:-triangle_field_vae_${SLURM_JOB_ID}}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES=${TRELLIS_DIST_TIMEOUT_MINUTES:-60}
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

DATA_DIR="{\"neuframe_train\":{\"base\":\"$ROOT/splits/train\",\"triangle_field_voxel\":\"$ROOT/splits/train/triangle_field_voxels_256\"}}"

python /home/koussa/scratch/TRELLIS.2/train.py \
  --config /home/koussa/scratch/TRELLIS.2/configs/scvae/triangle_field_vae_next_dc_f16c32_fp16.json \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 8 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
