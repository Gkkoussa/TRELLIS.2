#!/bin/bash
#SBATCH --job-name=trellis-gdist-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch eval_gaussian_distance_vae.sh /path/to/run_dir"
  exit 1
fi

cd /home/koussa/scratch/TRELLIS.2

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT=/nfs/turbo/coe-jjparkcv-medium/koussa/neuframe
export RUN_DIR="$1"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_RUN_NAME="eval_${EVAL_SPLIT}_${SLURM_JOB_ID}"
export EVAL_RENDER_RESOLUTION="${EVAL_RENDER_RESOLUTION:-512}"

python /home/koussa/scratch/TRELLIS.2/eval_pbr_vae.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --num_samples 64 \
  --snapshot_batch_size 4 \
  --render_resolution "$EVAL_RENDER_RESOLUTION" \
  --deterministic_posterior
