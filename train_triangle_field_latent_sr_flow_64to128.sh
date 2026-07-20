#!/bin/bash
#SBATCH --job-name=trellis-trifield-latent-sr64to128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-triangle_field_latent_sr_flow_64to128_film}"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_JOB_ID}}"
export FLOW_CONFIG="${FLOW_CONFIG:-/home/koussa/scratch/TRELLIS.2/configs/gen/triangle_field_latent_sr_flow_64to128_film_f16c32_fp16_objxl4k.json}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CKPT="${CKPT:-none}"
export LOW_TRIANGLE_FIELD_VOXEL_DIR="${LOW_TRIANGLE_FIELD_VOXEL_DIR:-$ROOT/triangle_field_voxels_64}"
export HIGH_TRIANGLE_FIELD_VOXEL_DIR="${HIGH_TRIANGLE_FIELD_VOXEL_DIR:-$ROOT/triangle_field_voxels_128}"
export DENSITY_TRIANGLE_FIELD_VOXEL_DIR="${DENSITY_TRIANGLE_FIELD_VOXEL_DIR:-}"
export TRIANGLE_FIELD_LATENT_DIR="${TRIANGLE_FIELD_LATENT_DIR:-$ROOT/triangle_field_latents/triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_128}"
export SPLIT="${SPLIT:-train}"
export INSTANCES_PATH="${INSTANCES_PATH:-$ROOT/splits/train_triangle_field_512/instances.txt}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

for d in "$LOW_TRIANGLE_FIELD_VOXEL_DIR" "$HIGH_TRIANGLE_FIELD_VOXEL_DIR" "$TRIANGLE_FIELD_LATENT_DIR"; do
  if [ ! -f "$d/metadata.csv" ]; then
    echo "Missing metadata: $d/metadata.csv" >&2
    exit 1
  fi
done
if [ -n "$DENSITY_TRIANGLE_FIELD_VOXEL_DIR" ] && [ ! -f "$DENSITY_TRIANGLE_FIELD_VOXEL_DIR/metadata.csv" ]; then
  echo "Missing density metadata: $DENSITY_TRIANGLE_FIELD_VOXEL_DIR/metadata.csv" >&2
  exit 1
fi
if [ ! -f "$INSTANCES_PATH" ]; then
  echo "Missing split instances: $INSTANCES_PATH" >&2
  exit 1
fi

export FILTERED_FLOW_CONFIG="$ROOT/outputs/$RUN_NAME/config.${SPLIT}.json"
python -c "import json, os; src=os.environ['FLOW_CONFIG']; dst=os.environ['FILTERED_FLOW_CONFIG']; instances=os.environ['INSTANCES_PATH']; cfg=json.load(open(src)); cfg['dataset']['args']['instances_path']=instances; json.dump(cfg, open(dst, 'w'), indent=4)"

if [ -n "$DENSITY_TRIANGLE_FIELD_VOXEL_DIR" ]; then
  DATA_DIR="{\"objxl4k_filtered\":{\"low_triangle_field_voxel\":\"$LOW_TRIANGLE_FIELD_VOXEL_DIR\",\"high_triangle_field_voxel\":\"$HIGH_TRIANGLE_FIELD_VOXEL_DIR\",\"triangle_field_latent\":\"$TRIANGLE_FIELD_LATENT_DIR\",\"density_triangle_field_voxel\":\"$DENSITY_TRIANGLE_FIELD_VOXEL_DIR\"}}"
else
  DATA_DIR="{\"objxl4k_filtered\":{\"low_triangle_field_voxel\":\"$LOW_TRIANGLE_FIELD_VOXEL_DIR\",\"high_triangle_field_voxel\":\"$HIGH_TRIANGLE_FIELD_VOXEL_DIR\",\"triangle_field_latent\":\"$TRIANGLE_FIELD_LATENT_DIR\"}}"
fi

python /home/koussa/scratch/TRELLIS.2/train.py \
  --config "$FILTERED_FLOW_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --ckpt "$CKPT" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
