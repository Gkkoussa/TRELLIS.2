#!/usr/bin/env bash
#SBATCH --job-name=voxelize_triangle_field_256
#SBATCH --output=./triangle_job_logs/voxelize_triangle_field_256_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gpus-per-task=1
#SBATCH --array=0-31%8

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export TRIANGLE_FIELD_RESOLUTION="${TRIANGLE_FIELD_RESOLUTION:-256}"
export TRIANGLE_FIELD_WORLD_SIZE="${TRIANGLE_FIELD_WORLD_SIZE:-32}"
export TRIANGLE_FIELD_CANDIDATE_SOURCE="${TRIANGLE_FIELD_CANDIDATE_SOURCE:-native}"

echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-local}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "ROOT: $ROOT"
echo "Resolution: $TRIANGLE_FIELD_RESOLUTION"
echo "Candidate source: $TRIANGLE_FIELD_CANDIDATE_SOURCE"
echo "Running shard ${SLURM_ARRAY_TASK_ID:-0} / $TRIANGLE_FIELD_WORLD_SIZE"

python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --resolution "$TRIANGLE_FIELD_RESOLUTION" \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 3 \
  --candidate_source "$TRIANGLE_FIELD_CANDIDATE_SOURCE" \
  --projection_mode inside_barycentric \
  --max_workers 1 \
  --rank "${SLURM_ARRAY_TASK_ID:-0}" \
  --world_size "$TRIANGLE_FIELD_WORLD_SIZE"
