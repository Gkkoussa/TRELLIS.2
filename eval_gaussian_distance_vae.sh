#!/bin/bash
#SBATCH --job-name=trellis-gdist-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN_NAME=gaussian_distance_vae

export RUN_DIR="${1:-$ROOT/outputs/$RUN_NAME}"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_RUN_NAME="eval_${EVAL_SPLIT}_${EVAL_CKPT}_${SLURM_JOB_ID}"
export EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-512}"

python eval_pbr_vae.py \
  --run_dir "$RUN_DIR" \
  --ckpt "$EVAL_CKPT" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --num_samples 64 \
  --snapshot_batch_size 4 \
  --render_resolution "$EVAL_RENDER_RESOLUTION" \
  --deterministic_posterior
