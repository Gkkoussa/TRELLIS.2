#!/bin/bash
#SBATCH --job-name=trellis-trifield-latent-sr-uncond256to512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

eval "$(conda shell.bash hook)"
module load cuda/12.8
module load gcc/11

conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

export PYTHONUNBUFFERED=1
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/triangle_field_latent_sr_flow_256to512_film_1m_52473478}"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_CKPT="${EVAL_CKPT:-10000}"
# export EVAL_EMA_RATE="${EVAL_EMA_RATE:-0.9999}"
export LOW_RESOLUTION="${LOW_RESOLUTION:-256}"
export HIGH_RESOLUTION="${HIGH_RESOLUTION:-512}"
export TRIANGLE_FIELD_LATENT_NAME="${TRIANGLE_FIELD_LATENT_NAME:-triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_512}"

export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-$ROOT/metadata_test_no_train_duplicates/metadata_test_no_train_duplicates.csv}"
export TRIANGLE_DENSITY_FILTER_CSV="${TRIANGLE_DENSITY_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
export EVAL_METADATA_FILTER_CSV="${EVAL_METADATA_FILTER_CSV:-$NO_TRAIN_DUPLICATE_CSV,$TRIANGLE_DENSITY_FILTER_CSV}"
export TRELLIS_METADATA_FILTER_CSV="$EVAL_METADATA_FILTER_CSV"

export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-64}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
export EVAL_SAMPLING_STEPS="${EVAL_SAMPLING_STEPS:-12}"
export EVAL_SEED="${EVAL_SEED:-0}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-eval_uncond_${EVAL_SPLIT}_${LOW_RESOLUTION}to${HIGH_RESOLUTION}_latent_flow_step${EVAL_CKPT}_n${EVAL_NUM_SAMPLES}_${SLURM_JOB_ID:-local}}"

if [ ! -d "$RUN_DIR/ckpts" ]; then
  echo "Checkpoint directory not found: $RUN_DIR/ckpts" >&2
  exit 1
fi
if [ ! -f "$ROOT/splits/$EVAL_SPLIT/instances.txt" ]; then
  echo "Split instances file not found: $ROOT/splits/$EVAL_SPLIT/instances.txt" >&2
  exit 1
fi
if [ ! -f "$NO_TRAIN_DUPLICATE_CSV" ]; then
  echo "No-train-duplicate metadata CSV not found: $NO_TRAIN_DUPLICATE_CSV" >&2
  exit 1
fi
if [ ! -f "$TRIANGLE_DENSITY_FILTER_CSV" ]; then
  echo "Triangle-density metadata CSV not found: $TRIANGLE_DENSITY_FILTER_CSV" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "${EVAL_EMA_RATE:-}" ]; then
  EXTRA_ARGS+=(--ema_rate "$EVAL_EMA_RATE")
fi
if [ -n "${EVAL_OUTPUT_DIR:-}" ]; then
  EXTRA_ARGS+=(--output_dir "$EVAL_OUTPUT_DIR")
else
  EXTRA_ARGS+=(--output_dir "$RUN_DIR/$EVAL_RUN_NAME")
fi
EXTRA_ARGS+=(--latent_name "$TRIANGLE_FIELD_LATENT_NAME")
if [ -n "${EVAL_RENDER_RESOLUTION:-}" ]; then
  EXTRA_ARGS+=(--render_resolution "$EVAL_RENDER_RESOLUTION")
fi

echo "Run dir: $RUN_DIR"
echo "Checkpoint: $EVAL_CKPT"
echo "Split: $EVAL_SPLIT"
echo "Metadata filter: $TRELLIS_METADATA_FILTER_CSV"
echo "No-train duplicates filter: $NO_TRAIN_DUPLICATE_CSV"
echo "Triangle density filter: $TRIANGLE_DENSITY_FILTER_CSV"
echo "Latents: $ROOT/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME"
echo "Unconditional mode: z_t=random latent noise, cond=zeros_like(cond)"
echo "Samples: $EVAL_NUM_SAMPLES"
echo "Batch size: $EVAL_BATCH_SIZE"
echo "Output: ${EVAL_OUTPUT_DIR:-$RUN_DIR/$EVAL_RUN_NAME}"

python -u /home/gpranav/pranav_work/scratch/TRELLIS.2/eval_triangle_field_latent_sr_flow_uncond.py \
  --run_dir "$RUN_DIR" \
  --root "$ROOT" \
  --ckpt "$EVAL_CKPT" \
  --split "$EVAL_SPLIT" \
  --low_resolution "$LOW_RESOLUTION" \
  --high_resolution "$HIGH_RESOLUTION" \
  --num_samples "$EVAL_NUM_SAMPLES" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$EVAL_NUM_WORKERS" \
  --steps "$EVAL_SAMPLING_STEPS" \
  --seed "$EVAL_SEED" \
  "${EXTRA_ARGS[@]}"
