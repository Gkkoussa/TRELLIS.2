#!/usr/bin/env bash
#SBATCH --job-name=trifield-vae-qem-dtrirefine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

module load cuda/12.8
module load gcc/11

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

REPO=/home/gpranav/pranav_work/scratch/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-triangle_field_vae_allres_qem_vertex_edge_dtri_refine}"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_JOB_ID}}"
export REFINE_CONFIG="${REFINE_CONFIG:-$REPO/configs/scvae/triangle_field_vae_allres_qem_vertex_edge_dtri_refine.json}"
export TRAIN_INSTANCES="${TRAIN_INSTANCES:-$ROOT/splits/train_triangle_field_512/instances.txt}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BASE_RUN="${BASE_RUN:-$ROOT/outputs/triangle_field_vae_allres_invarea_fullaux_52454251}"
export BASE_STEP="${BASE_STEP:-0320000}"
export FINETUNE_ENCODER="${FINETUNE_ENCODER:-$BASE_RUN/ckpts/encoder_step${BASE_STEP}.pt}"
export FINETUNE_DECODER="${FINETUNE_DECODER:-$BASE_RUN/ckpts/decoder_step${BASE_STEP}.pt}"

for REQUIRED_FILE in \
  "$REFINE_CONFIG" \
  "$TRAIN_INSTANCES" \
  "$FINETUNE_ENCODER" \
  "$FINETUNE_DECODER"; do
  if [[ ! -f "$REQUIRED_FILE" ]]; then
    echo "Missing required file: $REQUIRED_FILE" >&2
    exit 1
  fi
done

for RESOLUTION in 32 64 128 256 512; do
  VOXEL_DIR="$ROOT/triangle_field_voxels_${RESOLUTION}"
  QEM_DIR="$ROOT/qem_edge_collapsed_meshes_${RESOLUTION}"
  if [[ ! -f "$VOXEL_DIR/metadata.csv" ]]; then
    echo "Missing triangle-field metadata: $VOXEL_DIR/metadata.csv" >&2
    exit 1
  fi
  if [[ ! -f "$QEM_DIR/metadata.csv" ]]; then
    echo "Missing QEM metadata: $QEM_DIR/metadata.csv" >&2
    exit 1
  fi
done

OUTPUT_DIR="$ROOT/outputs/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
JOB_CONFIG="$OUTPUT_DIR/config.json"

python - "$REFINE_CONFIG" "$JOB_CONFIG" "$TRAIN_INSTANCES" \
  "$FINETUNE_ENCODER" "$FINETUNE_DECODER" <<'PY'
import json
import os
import sys

source, destination, instances, encoder_checkpoint, decoder_checkpoint = sys.argv[1:6]
with open(source, "r", encoding="utf-8") as handle:
    config = json.load(handle)
config["dataset"]["args"]["instances_path"] = instances
for checkpoint in (encoder_checkpoint, decoder_checkpoint):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
config["trainer"]["args"]["finetune_ckpt"] = {
    "encoder": encoder_checkpoint,
    "decoder": decoder_checkpoint,
}
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=4)
PY

DATA_DIR="{\"objxl4k_filtered_train\":{\"triangle_field_voxel_32\":\"$ROOT/triangle_field_voxels_32\",\"triangle_field_voxel_64\":\"$ROOT/triangle_field_voxels_64\",\"triangle_field_voxel_128\":\"$ROOT/triangle_field_voxels_128\",\"triangle_field_voxel_256\":\"$ROOT/triangle_field_voxels_256\",\"triangle_field_voxel_512\":\"$ROOT/triangle_field_voxels_512\"}}"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE="${FLEX_GEMM_USE_AUTOTUNE_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"

echo "Run: $RUN_NAME"
echo "Initialization encoder: $FINETUNE_ENCODER"
echo "Initialization decoder: $FINETUNE_DECODER"
echo "Raw low-d_tri threshold: $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"]["decoder"]["args"]["r512_edge_dtri_threshold"])' "$JOB_CONFIG")"

python "$REPO/train_triangle_field_vae_dtri_edge_refine.py" \
  --config "$JOB_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
