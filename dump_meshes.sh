#!/bin/bash
#SBATCH --job-name=encode_gaussian_distance_latent
#SBATCH --output=./job_logs/encode_gaussian_distance_latent_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gpus-per-task=1
#SBATCH --array=1,4,5,7%4

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Running shard $SLURM_ARRAY_TASK_ID / 16"

python data_toolkit/encode_gaussian_distance_latent.py \
  --root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --gaussian_distance_latent_root "$ROOT" \
  --resolution 256 \
  --model_root "$ROOT/outputs" \
  --enc_model gaussian_distance_vae \
  --ckpt step0230000 \
  --loader_workers 4 \
  --read_threads 1 \
  --saver_workers 4 \
  --rank "$SLURM_ARRAY_TASK_ID" \
  --world_size 16