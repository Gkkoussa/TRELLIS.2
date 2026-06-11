#!/bin/bash
#SBATCH --job-name=trellis-ggrid64-unetdit-flow
#SBATCH --output=./job_logs/trellis-ggrid64-unetdit-flow_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gres=gpu:8

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

mkdir -p job_logs

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_triangle_area_scores.csv}"

export RUN_NAME="${RUN_NAME:-gaussian_grid64_sparse_unet_dit_flow_triangle_area_filtered_${SLURM_JOB_ID:-manual}}"
export RUN_DIR="$ROOT/outputs/$RUN_NAME"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"

export DATA_DIR="{\"train\":{\"base\":\"$ROOT/splits/train\",\"gaussian_distance_voxel\":\"$ROOT/gaussian_distance_voxels_64\",\"_metadata_filter_csv\":\"$ROOT/splits/train/metadata.csv,$TRIANGLE_FILTER_CSV\"}}"

export CONFIG="${CONFIG:-configs/gen/gaussian_grid64_sparse_unet_dit_flow_bf16_triangle_area_filtered_less_compress.json}"
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
  --num_gpus "${NUM_GPUS:-8}" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
