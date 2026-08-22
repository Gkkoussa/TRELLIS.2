#!/bin/bash
#SBATCH --job-name=encode-trifield-allres
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-r%a-%j.out
#SBATCH --error=logs/%x-r%a-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

RESOLUTIONS=(32 64 256 512)
RESOLUTION="${RESOLUTION:-${RESOLUTIONS[${SLURM_ARRAY_TASK_ID:?Submit as an array over indices 0-3}]}}"
case "$RESOLUTION" in
  32|64|256|512) ;;
  *) echo "Unsupported resolution: $RESOLUTION" >&2; exit 1 ;;
esac

ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
MODEL_ROOT="$ROOT/outputs"
ENC_MODEL="triangle_field_vae_allres_invarea_fullaux_52454251"
CKPT="step0320000"
INSTANCES="$ROOT/splits/train_triangle_field_512/instances.txt"
CHECKPOINT="$MODEL_ROOT/$ENC_MODEL/ckpts/encoder_${CKPT}.pt"
VOXEL_METADATA="$ROOT/triangle_field_voxels_${RESOLUTION}/metadata.csv"

for path in "$INSTANCES" "$CHECKPOINT" "$VOXEL_METADATA"; do
  if [ ! -f "$path" ]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

echo "Resolution: $RESOLUTION"
echo "Encoder: $CHECKPOINT"
echo "Instances: $INSTANCES"

python data_toolkit/encode_triangle_field_latent.py \
  --root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --triangle_field_latent_root "$ROOT" \
  --resolution "$RESOLUTION" \
  --model_root "$MODEL_ROOT" \
  --enc_model "$ENC_MODEL" \
  --ckpt "$CKPT" \
  --instances "$INSTANCES" \
  --loader_workers 8 \
  --saver_workers 4 \
  --queue_size 16
