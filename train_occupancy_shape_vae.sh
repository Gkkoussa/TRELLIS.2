#!/bin/bash
#SBATCH --job-name=trellis-occ-shape-vae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

module load cuda/12.8
module load gcc/11

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN_NAME="${RUN_NAME:-occupancy_shape_vae_triangle_filtered_${SLURM_JOB_ID}}"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES=${TRELLIS_DIST_TIMEOUT_MINUTES:-60}

DATA_DIR="{\"train\":{\"base\":\"$ROOT/splits/train\",\"gaussian_distance_voxel\":\"$ROOT/gaussian_distance_voxels_256\",\"_metadata_filter_csv\":\"$ROOT/splits/train/metadata.csv,$TRIANGLE_FILTER_CSV\"}}"

python /home/gpranav/pranav_work/scratch/TRELLIS.2/train.py \
  --config /home/gpranav/pranav_work/scratch/TRELLIS.2/configs/scvae/occupancy_shape_vae_next_dc_f16c32_fp16.json \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 4 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
