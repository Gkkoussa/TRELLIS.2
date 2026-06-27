#!/bin/bash
#SBATCH --job-name=trellis-occ-shape-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2


cd /home/gpranav/pranav_work/scratch/TRELLIS.2

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

module load cuda/12.8
module load gcc/11

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
export RUN_DIR="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/outputs/occupancy_shape_vae_triangle_filtered_51728720"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-100000}"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-/home/gpranav/pranav_work/scratch/TRELLIS.2/metadata_test_no_train_duplicates.csv}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_${EVAL_SPLIT}_triangle_no_train_${SLURM_JOB_ID}}"
export EVAL_METADATA_FILTER_CSV="${EVAL_METADATA_FILTER_CSV:-$NO_TRAIN_DUPLICATE_CSV,$TRIANGLE_FILTER_CSV}"

EXTRA_ARGS=()
if [ -n "${EVAL_BATCH_SIZE:-}" ]; then
  EXTRA_ARGS+=(--batch_size "$EVAL_BATCH_SIZE")
fi
if [ -n "${EVAL_NUM_WORKERS:-}" ]; then
  EXTRA_ARGS+=(--num_workers "$EVAL_NUM_WORKERS")
fi
if [ -n "${EVAL_MAX_BATCHES:-}" ]; then
  EXTRA_ARGS+=(--max_batches "$EVAL_MAX_BATCHES")
fi
if [ -n "${EVAL_NUM_SAMPLES:-}" ]; then
  EXTRA_ARGS+=(--num_samples "$EVAL_NUM_SAMPLES")
fi
if [ -n "${EVAL_SNAPSHOT_BATCH_SIZE:-}" ]; then
  EXTRA_ARGS+=(--snapshot_batch_size "$EVAL_SNAPSHOT_BATCH_SIZE")
fi

python /home/gpranav/pranav_work/scratch/TRELLIS.2/eval_occupancy_shape_vae.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --split "$EVAL_SPLIT" \
  --ckpt "$EVAL_CKPT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --metadata_filter_csv "$EVAL_METADATA_FILTER_CSV" \
  --deterministic_posterior \
  "${EXTRA_ARGS[@]}"
