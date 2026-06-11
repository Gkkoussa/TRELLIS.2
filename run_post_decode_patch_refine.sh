#!/bin/bash
#SBATCH --job-name=trellis-postdecode-refine
#SBATCH --output=./job_logs/trellis-postdecode-refine_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=jjparkcv_owned2

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate "${CONDA_ENV:-/home/gpranav/pranav_work/scratch/envs/trellis2}"

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/
mkdir -p job_logs

ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
SPLIT="${SPLIT:-test}"

LATENT_RUN_DIR="${LATENT_RUN_DIR:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/outputs/michelangelo2gaussian_distance_flow_50391241}"
LATENT_CONFIG="${LATENT_CONFIG:-configs/gen/slat_flow_michelangelo2gaussian_distance_dit_1_3B_512_bf16_8gpu.json}"
LATENT_CKPT="${LATENT_CKPT:-latest}"
LATENT_EMA_RATE="${LATENT_EMA_RATE:-}"
LATENT_SAMPLING_STEPS="${LATENT_SAMPLING_STEPS:-12}"
LATENT_GUIDANCE_STRENGTH="${LATENT_GUIDANCE_STRENGTH:-3.0}"

PATCH_RUN_DIR="${PATCH_RUN_DIR:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/outputs/gaussian_patch_sparse_flow_dit32_50710472}"
PATCH_CONFIG="${PATCH_CONFIG:-configs/gen/gaussian_patch_sparse_flow_dit_32_1_3B_bf16.json}"
PATCH_CKPT="${PATCH_CKPT:-latest}"
PATCH_EMA_RATE="${PATCH_EMA_RATE:-}"
PATCH_T="${PATCH_T:-1.0}"
PATCH_STEPS="${PATCH_STEPS:-50}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-16}"
PATCH_STRIDE="${PATCH_STRIDE:-16}"
MAX_PATCHES="${MAX_PATCHES:-4096}"

GAUSSIAN_DISTANCE_LATENT_NAME="${GAUSSIAN_DISTANCE_LATENT_NAME:-gaussian_distance_vae_512_step0220000_512}"
MICHELANGELO_LATENT_NAME="${MICHELANGELO_LATENT_NAME:-shapevae256_pretrained}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
START_INDEX="${START_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./post_decode_patch_refine_outputs}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-256}"
RENDER_SSAA="${RENDER_SSAA:-2}"
SEED="${SEED:-0}"

CMD=(
  python post_decode_patch_refine.py
  --latent_run_dir "$LATENT_RUN_DIR"
  --latent_config "$LATENT_CONFIG"
  --latent_ckpt "$LATENT_CKPT"
  --latent_sampling_steps "$LATENT_SAMPLING_STEPS"
  --latent_guidance_strength "$LATENT_GUIDANCE_STRENGTH"
  --patch_run_dir "$PATCH_RUN_DIR"
  --patch_config "$PATCH_CONFIG"
  --patch_ckpt "$PATCH_CKPT"
  --patch_t "$PATCH_T"
  --patch_steps "$PATCH_STEPS"
  --patch_batch_size "$PATCH_BATCH_SIZE"
  --patch_stride "$PATCH_STRIDE"
  --max_patches "$MAX_PATCHES"
  --root "$ROOT"
  --split "$SPLIT"
  --gaussian_distance_latent_name "$GAUSSIAN_DISTANCE_LATENT_NAME"
  --michelangelo_latent_name "$MICHELANGELO_LATENT_NAME"
  --num_samples "$NUM_SAMPLES"
  --start_index "$START_INDEX"
  --output_dir "$OUTPUT_DIR"
  --render_resolution "$RENDER_RESOLUTION"
  --render_ssaa "$RENDER_SSAA"
  --seed "$SEED"
)

if [[ -n "$LATENT_EMA_RATE" ]]; then
  CMD+=(--latent_ema_rate "$LATENT_EMA_RATE")
fi
if [[ -n "$PATCH_EMA_RATE" ]]; then
  CMD+=(--patch_ema_rate "$PATCH_EMA_RATE")
fi
if [[ -n "${INDICES:-}" ]]; then
  CMD+=(--indices "$INDICES")
fi
if [[ -n "${SHA256S:-}" ]]; then
  CMD+=(--sha256s "$SHA256S")
fi
if [[ -n "${METADATA_FILTER_CSV:-}" ]]; then
  CMD+=(--metadata_filter_csv "$METADATA_FILTER_CSV")
fi
if [[ "${RANDOM_SAMPLES:-0}" == "1" ]]; then
  CMD+=(--random)
fi
if [[ "${SAVE_VXZ:-0}" == "1" ]]; then
  CMD+=(--save_vxz)
fi

echo "${CMD[@]}"
"${CMD[@]}"
