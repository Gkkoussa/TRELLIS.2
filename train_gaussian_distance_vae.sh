#!/bin/bash
#SBATCH --job-name=trellis-gdist-vae-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

module load cuda/12.8
module load gcc/11

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN_NAME="${RUN_NAME:-gaussian_distance_vae_512}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

# Single-node multi-GPU: bind the rendezvous server to loopback. Using the node's
# primary IP can reverse-DNS to the cluster hostname and trigger c10d errno 97
# ("Address family not supported by protocol") while ranks connect to MASTER_ADDR.
# For multi-node jobs, override: MASTER_ADDR=<head-node-ipv4> sbatch ...
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

# Debug/safety envs
export TRELLIS_DIST_TIMEOUT_MINUTES=${TRELLIS_DIST_TIMEOUT_MINUTES:-90}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Avoid heavy startup paths that can stall NCCL (large all_gather during snapshots).
export TRELLIS_SKIP_STARTUP_SNAPSHOTS="${TRELLIS_SKIP_STARTUP_SNAPSHOTS:-1}"

# flex_gemm: use a per-job cache under Slurm TMPDIR so ranks don't fight one tmp file.
export FLEX_GEMM_AUTOTUNE_CACHE_PATH="${TMPDIR:-/tmp}/flex_gemm_${SLURM_JOB_ID:-local}/autotune_cache.json"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
mkdir -p "$(dirname "$FLEX_GEMM_AUTOTUNE_CACHE_PATH")"
rm -f "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH.lock" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH".tmp*

# Clean corrupted flex_gemm autotune cache from failed multi-GPU runs (default package path)
mkdir -p /home/gpranav/.flex_gemm
rm -f /home/gpranav/.flex_gemm/autotune_cache.json
rm -f /home/gpranav/.flex_gemm/autotune_cache.json.tmp*

DATA_DIR="{\"train\":{\"base\":\"$ROOT/splits/train\",\"gaussian_distance_voxel\":\"$ROOT/splits/train/gaussian_distance_voxels_512\"}}"

python train.py \
  --config configs/scvae/gaussian_distance_vae_next_dc_f16c32_fp16_ft_512.json \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 4 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 0
