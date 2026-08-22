#!/bin/bash
#SBATCH --job-name=trifield-zarr-vae-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate trellis2

export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

python data_toolkit/smoke_test_triangle_field_zarr_vae.py \
  --config configs/scvae/triangle_field_vae_next_dc_f16c32_fp16_allres_full239k_avgfrom512_random_noarea.json \
  --batch_sizes "${BATCH_SIZES:-32:16,64:16,128:8,256:4,512:2}" \
  --warmup_steps "${WARMUP_STEPS:-1}" \
  --timed_steps "${TIMED_STEPS:-2}" \
  --loader_batches "${LOADER_BATCHES:-20}" \
  --load_quantile "${LOAD_QUANTILE:-0.95}"
