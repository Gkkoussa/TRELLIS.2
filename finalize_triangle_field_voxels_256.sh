#!/usr/bin/env bash
#SBATCH --job-name=finalize_triangle_field_256
#SBATCH --output=./triangle_job_logs/finalize_triangle_field_256_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1
#SBATCH --account=jjparkcv_owned2


source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export TRIANGLE_FIELD_RESOLUTION="${TRIANGLE_FIELD_RESOLUTION:-256}"
export TRIANGLE_FIELD_CHECK_WORKERS="${TRIANGLE_FIELD_CHECK_WORKERS:-8}"

echo "ROOT: $ROOT"
echo "Resolution: $TRIANGLE_FIELD_RESOLUTION"

python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$ROOT"

python data_toolkit/check_triangle_field_dataset.py \
  --voxel_root "$ROOT/triangle_field_voxels_$TRIANGLE_FIELD_RESOLUTION" \
  --num_workers "$TRIANGLE_FIELD_CHECK_WORKERS" \
  --output_json "$ROOT/triangle_field_voxels_$TRIANGLE_FIELD_RESOLUTION/check_stats.json"
