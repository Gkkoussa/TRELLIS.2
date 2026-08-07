#!/bin/bash
#SBATCH --job-name=encode-trifield-avg512-s60k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
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

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
MODEL_ROOT="$ROOT/outputs"
ENC_MODEL=triangle_field_vae_allres_avgfrom512_invarea_fullaux_52727251
CKPT=step0060000
INSTANCES="$ROOT/splits/train_triangle_field_512/instances.txt"
VOXEL_VIEW="$ROOT/triangle_field_voxels_avgfrom512_encode_view"

if [ "$RESOLUTION" = 512 ]; then
  VOXEL_SOURCE="$ROOT/triangle_field_voxels_512"
else
  VOXEL_SOURCE="$ROOT/triangle_field_voxels_${RESOLUTION}_avg_from_512"
fi

CHECKPOINT="$MODEL_ROOT/$ENC_MODEL/ckpts/encoder_${CKPT}.pt"
for path in "$INSTANCES" "$CHECKPOINT" "$VOXEL_SOURCE/metadata.csv"; do
  if [ ! -f "$path" ]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

mkdir -p "$VOXEL_VIEW"
ln -sfn "$VOXEL_SOURCE" "$VOXEL_VIEW/triangle_field_voxels_${RESOLUTION}"

export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

echo "Resolution: $RESOLUTION"
echo "Voxel source: $VOXEL_SOURCE"
echo "Encoder: $CHECKPOINT"
echo "Instances: $INSTANCES"

python data_toolkit/encode_triangle_field_latent.py \
  --root "$ROOT" \
  --triangle_field_voxel_root "$VOXEL_VIEW" \
  --triangle_field_latent_root "$ROOT" \
  --resolution "$RESOLUTION" \
  --model_root "$MODEL_ROOT" \
  --enc_model "$ENC_MODEL" \
  --ckpt "$CKPT" \
  --instances "$INSTANCES" \
  --loader_workers 8 \
  --saver_workers 4 \
  --queue_size 16

LATENT_DIR="$ROOT/triangle_field_latents/${ENC_MODEL}_${CKPT}_${RESOLUTION}"
python -c \
  "from data_toolkit.build_metadata import update_metadata; from easydict import EasyDict; update_metadata('$LATENT_DIR', EasyDict(from_merged_records=False))"

echo "Completed latent encoding and metadata merge: $LATENT_DIR"
