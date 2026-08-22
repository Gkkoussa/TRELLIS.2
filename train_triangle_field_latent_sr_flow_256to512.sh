#!/bin/bash
#SBATCH --job-name=trellis-trifield-latent-sr256to512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export RUN_NAME="${RUN_NAME:-triangle_field_latent_sr_flow_256to512_film_1m_${SLURM_JOB_ID}}"
export FLOW_CONFIG="${FLOW_CONFIG:-/home/gpranav/pranav_work/scratch/TRELLIS.2/configs/gen/triangle_field_latent_sr_flow_256to512_film_f16c32_fp16_objxl4k_1m.json}"
export LOW_TRIANGLE_FIELD_VOXEL_DIR="${LOW_TRIANGLE_FIELD_VOXEL_DIR:-$ROOT/triangle_field_voxels_256}"
export HIGH_TRIANGLE_FIELD_VOXEL_DIR="${HIGH_TRIANGLE_FIELD_VOXEL_DIR:-$ROOT/triangle_field_voxels_512}"
export TRIANGLE_FIELD_LATENT_DIR="${TRIANGLE_FIELD_LATENT_DIR:-$ROOT/triangle_field_latents/triangle_field_vae_512_invarea_auxdrop_52039231_step0180000_512}"
export SPLIT="${SPLIT:-train}"
export INSTANCES_PATH="${INSTANCES_PATH:-$ROOT/splits/$SPLIT/instances.txt}"

mkdir -p "$ROOT/outputs/$RUN_NAME"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TRELLIS_DIST_TIMEOUT_MINUTES="${TRELLIS_DIST_TIMEOUT_MINUTES:-60}"
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0

for d in "$LOW_TRIANGLE_FIELD_VOXEL_DIR" "$HIGH_TRIANGLE_FIELD_VOXEL_DIR" "$TRIANGLE_FIELD_LATENT_DIR"; do
  if [ ! -f "$d/metadata.csv" ]; then
    echo "Missing metadata: $d/metadata.csv" >&2
    exit 1
  fi
done
if [ ! -f "$INSTANCES_PATH" ]; then
  echo "Missing split instances: $INSTANCES_PATH" >&2
  exit 1
fi

export FILTERED_FLOW_CONFIG="$ROOT/outputs/$RUN_NAME/config.${SPLIT}.json"
python -c "import json, os; src=os.environ['FLOW_CONFIG']; dst=os.environ['FILTERED_FLOW_CONFIG']; instances=os.environ['INSTANCES_PATH']; cfg=json.load(open(src)); cfg['dataset']['args']['instances_path']=instances; json.dump(cfg, open(dst, 'w'), indent=4)"

DATA_DIR="{\"objxl4k_filtered\":{\"low_triangle_field_voxel\":\"$LOW_TRIANGLE_FIELD_VOXEL_DIR\",\"high_triangle_field_voxel\":\"$HIGH_TRIANGLE_FIELD_VOXEL_DIR\",\"triangle_field_latent\":\"$TRIANGLE_FIELD_LATENT_DIR\"}}"

python /home/gpranav/pranav_work/scratch/TRELLIS.2/train.py \
  --config "$FILTERED_FLOW_CONFIG" \
  --output_dir "$ROOT/outputs/$RUN_NAME" \
  --ckpt none \
  --data_dir "$DATA_DIR" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 6 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --auto_retry 3
