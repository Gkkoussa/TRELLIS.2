#!/bin/bash
#SBATCH --job-name=trifield-flow-l1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

module load cuda/12.8
module load gcc/11

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_DIR="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/outputs/michelangelo_shape2triangle_field_flow_filtered_51756371"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-latest}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_${EVAL_SPLIT}_l1_export_${SLURM_JOB_ID}}"

export TRIANGLE_FIELD_LATENT_NAME="${TRIANGLE_FIELD_LATENT_NAME:-triangle_field_vae_51685536_step0100000_256}"
export MICHELANGELO_NAME="${MICHELANGELO_NAME:-shapevae256_pretrained}"
export SHAPE_LATENT_NAME="${SHAPE_LATENT_NAME:-occupancy_shape_vae_triangle_filtered_51728720_step0100000_256}"
export TRIANGLE_FILTER_CSV="${TRIANGLE_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-/home/gpranav/pranav_work/scratch/TRELLIS.2/metadata_test_no_train_duplicates.csv}"
export EVAL_METADATA_FILTER_CSV="${EVAL_METADATA_FILTER_CSV:-$ROOT/splits/${EVAL_SPLIT}/metadata.csv,$NO_TRAIN_DUPLICATE_CSV,$TRIANGLE_FILTER_CSV}"

export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-64}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export EVAL_DECODE_BATCH_SIZE="${EVAL_DECODE_BATCH_SIZE:-4}"
export EVAL_SNAPSHOT_NUM_SAMPLES="${EVAL_SNAPSHOT_NUM_SAMPLES:-$EVAL_NUM_SAMPLES}"
export EVAL_SNAPSHOT_BATCH_SIZE="${EVAL_SNAPSHOT_BATCH_SIZE:-$EVAL_BATCH_SIZE}"
export EVAL_SAMPLING_STEPS="${EVAL_SAMPLING_STEPS:-12}"
export EVAL_GUIDANCE_STRENGTH="${EVAL_GUIDANCE_STRENGTH:-1.0}"

DATA_DIR="{\"${EVAL_SPLIT}\":{\"metadata\":\"$ROOT/splits/${EVAL_SPLIT}\",\"triangle_field_latent\":\"$ROOT/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME\",\"michelangelo_latent\":\"$ROOT/michelangelo_latents/$MICHELANGELO_NAME\",\"shape_latent\":\"$ROOT/shape_latents/$SHAPE_LATENT_NAME\",\"_metadata_filter_csv\":\"$EVAL_METADATA_FILTER_CSV\"}}"

EXTRA_ARGS=()
if [ -n "${EVAL_EMA_RATE:-}" ]; then
  EXTRA_ARGS+=(--ema_rate "$EVAL_EMA_RATE")
fi
if [ "${EVAL_SHUFFLE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--shuffle)
fi
if [ "${EVAL_NO_INDIVIDUAL_VISUALIZATIONS:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no_individual_visualizations)
fi

python /home/gpranav/pranav_work/scratch/TRELLIS.2/eval_triangle_field_flow_l1_export.py \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --split "$EVAL_SPLIT" \
  --ckpt "$EVAL_CKPT" \
  --output_dir "$RUN_DIR/$EVAL_RUN_NAME" \
  --num_samples "$EVAL_NUM_SAMPLES" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --decode_batch_size "$EVAL_DECODE_BATCH_SIZE" \
  --snapshot_num_samples "$EVAL_SNAPSHOT_NUM_SAMPLES" \
  --snapshot_batch_size "$EVAL_SNAPSHOT_BATCH_SIZE" \
  --sampling_steps "$EVAL_SAMPLING_STEPS" \
  --guidance_strength "$EVAL_GUIDANCE_STRENGTH" \
  "${EXTRA_ARGS[@]}"
