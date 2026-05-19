#!/bin/bash
#SBATCH --job-name=trellis-gdist-vae-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN_NAME=gaussian_distance_vae_512_smoke

mkdir -p "$ROOT/outputs/$RUN_NAME"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_SKIP_STARTUP_SNAPSHOTS=1
export TRELLIS_DIST_TIMEOUT_MINUTES=${TRELLIS_DIST_TIMEOUT_MINUTES:-60}
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export FLEX_GEMM_AUTOTUNE_CACHE_PATH="${TMPDIR:-/tmp}/flex_gemm_${SLURM_JOB_ID}/autotune_cache.json"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
mkdir -p "$(dirname "$FLEX_GEMM_AUTOTUNE_CACHE_PATH")"
rm -f "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH.lock" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH".tmp*

DATA_DIR="{\"train\":{\"base\":\"$ROOT/splits/train\",\"gaussian_distance_voxel\":\"$ROOT/splits/train/gaussian_distance_voxels_512\"}}"

python train.py \
  --config configs/scvae/gaussian_distance_vae_next_dc_f16c32_fp16_ft_512.json \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 1 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 0
