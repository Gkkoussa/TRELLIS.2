#!/usr/bin/env bash
#SBATCH --job-name=finalize_qem
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/finalize_qem_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

module load cuda/12.8
module load gcc/11

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2
PYTHON=/home/gpranav/pranav_work/scratch/envs/trellis2/bin/python

REPO=/home/gpranav/pranav_work/scratch/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
export QEM_RESOLUTION="${QEM_RESOLUTION:-128}"
export QEM_CHECK_WORKERS="${QEM_CHECK_WORKERS:-8}"
QEM_DIR="$ROOT/qem_edge_collapsed_meshes_$QEM_RESOLUTION"

"$PYTHON" data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --qem_edge_collapsed_root "$ROOT"

"$PYTHON" data_toolkit/check_qem_edge_collapsed_dataset.py \
  --qem_root "$QEM_DIR" \
  --triangle_field_voxel_root "$ROOT/triangle_field_voxels_$QEM_RESOLUTION" \
  --resolution "$QEM_RESOLUTION" \
  --num_workers "$QEM_CHECK_WORKERS" \
  --output_json "$QEM_DIR/validation_stats.json" \
  --invalid_instances "$QEM_DIR/invalid_instances.txt" \
  --failures_csv "$QEM_DIR/validation_failures.csv"
