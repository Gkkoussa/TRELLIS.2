#!/bin/bash
#SBATCH --job-name=trellis-gdist-overlimit-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

module load cuda/12.8
module load gcc/11

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN_NAME=gaussian_distance_vae_512

export RUN_DIR="${1:-$ROOT/outputs/$RUN_NAME}"
export EVAL_SPLIT="${EVAL_SPLIT:-train}"
export EVAL_CKPT="${EVAL_CKPT:-step0220000}"
export MIN_ACTIVE_VOXELS="${MIN_ACTIVE_VOXELS:-1000000}"
export MAX_ACTIVE_VOXELS="${MAX_ACTIVE_VOXELS:-100000000}"
export EVAL_MAX_INSTANCES="${EVAL_MAX_INSTANCES:-}"
export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-16}"
export EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-512}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_${EVAL_SPLIT}_over${MIN_ACTIVE_VOXELS}_${EVAL_CKPT}_${SLURM_JOB_ID}}"

EXTRA_ARGS=()
if [ -n "$EVAL_MAX_INSTANCES" ]; then
  EXTRA_ARGS+=(--max_instances "$EVAL_MAX_INSTANCES")
fi

python eval_gaussian_distance_vae_overlimit.py \
  --run_dir "$RUN_DIR" \
  --ckpt "$EVAL_CKPT" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --min_active_voxels "$MIN_ACTIVE_VOXELS" \
  --max_active_voxels "$MAX_ACTIVE_VOXELS" \
  --batch_size 1 \
  --num_workers 0 \
  --num_samples "$EVAL_NUM_SAMPLES" \
  --snapshot_batch_size 1 \
  --render_resolution "$EVAL_RENDER_RESOLUTION" \
  --deterministic_posterior \
  "${EXTRA_ARGS[@]}"
