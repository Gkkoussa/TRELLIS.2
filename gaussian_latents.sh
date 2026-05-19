#!/usr/bin/env bash
#SBATCH --job-name=trellis2_gaussian_latents_512_missing    
#SBATCH --output=./job_logs/trellis2_gaussian_latents_512_missing_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=spgpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --account=jjparkcv_owned1
#SBATCH --gpus-per-task=1
#SBATCH --array=0-3%4

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p job_logs

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k

# Keep flex_gemm autotune state isolated per Slurm task. Without this, parallel
# array jobs can deadlock on /home/gpranav/.flex_gemm/autotune_cache.json.lock.
export FLEX_GEMM_AUTOTUNE_CACHE_PATH="${TMPDIR:-/tmp}/flex_gemm_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}/autotune_cache.json"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
mkdir -p "$(dirname "$FLEX_GEMM_AUTOTUNE_CACHE_PATH")"

python data_toolkit/encode_gaussian_distance_latent.py \
  --root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --gaussian_distance_latent_root "$ROOT" \
  --resolution 512 \
  --model_root "$ROOT/outputs" \
  --enc_model gaussian_distance_vae_512 \
  --ckpt step0220000 \
  --loader_workers 4 \
  --read_threads 1 \
  --saver_workers 4 \
  --rank "$SLURM_ARRAY_TASK_ID" \
  --world_size 4
