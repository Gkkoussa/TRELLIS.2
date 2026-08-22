#!/bin/bash
#SBATCH --job-name=occ-shape-encode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
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

export SHAPE_VAE_RUN="${1:-${SHAPE_VAE_RUN:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/outputs/occupancy_shape_vae_triangle_filtered_51728720}}"
export SHAPE_VAE_CKPT="${2:-${SHAPE_VAE_CKPT:-step0100000}}"

if [ -z "$SHAPE_VAE_RUN" ] || [ -z "$SHAPE_VAE_CKPT" ]; then
  echo "Usage: sbatch encode_occupancy_shape_latent_filtered.sh <shape_vae_run_name> <checkpoint_step>"
  echo "Example: sbatch encode_occupancy_shape_latent_filtered.sh occupancy_shape_vae_triangle_filtered_12345678 step0100000"
  exit 1
fi

export SHAPE_LATENT_NAME="${SHAPE_VAE_RUN}_${SHAPE_VAE_CKPT}_256"
export SHAPE_METADATA_FILTER_CSV="${SHAPE_METADATA_FILTER_CSV:-$ROOT/metadata_no_triangle_dense.csv}"
export SHAPE_RESOLUTION="${SHAPE_RESOLUTION:-256}"
export SHAPE_LOADER_WORKERS="${SHAPE_LOADER_WORKERS:-4}"
export SHAPE_SAVER_WORKERS="${SHAPE_SAVER_WORKERS:-4}"
export SHAPE_READ_THREADS="${SHAPE_READ_THREADS:-1}"

echo "ROOT=$ROOT"
echo "SHAPE_VAE_RUN=$SHAPE_VAE_RUN"
echo "SHAPE_VAE_CKPT=$SHAPE_VAE_CKPT"
echo "SHAPE_LATENT_NAME=$SHAPE_LATENT_NAME"
echo "SHAPE_METADATA_FILTER_CSV=$SHAPE_METADATA_FILTER_CSV"

python data_toolkit/encode_occupancy_shape_latent.py \
  --root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --shape_latent_root "$ROOT" \
  --resolution "$SHAPE_RESOLUTION" \
  --model_root "$ROOT/outputs" \
  --enc_model "$SHAPE_VAE_RUN" \
  --ckpt "$SHAPE_VAE_CKPT" \
  --metadata_filter_csv "$SHAPE_METADATA_FILTER_CSV" \
  --loader_workers "$SHAPE_LOADER_WORKERS" \
  --saver_workers "$SHAPE_SAVER_WORKERS" \
  --read_threads "$SHAPE_READ_THREADS"
