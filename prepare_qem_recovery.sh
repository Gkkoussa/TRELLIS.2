#!/usr/bin/env bash
#SBATCH --job-name=prepare_qem_recovery
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/prepare_qem_recovery_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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
export QEM_RESOLUTIONS="${QEM_RESOLUTIONS:-32,64,128,256}"
export QEM_CHECK_WORKERS="${QEM_CHECK_WORKERS:-8}"
export QEM_REQUIRE_CLEAN="${QEM_REQUIRE_CLEAN:-0}"

IFS=',' read -r -a RESOLUTIONS <<< "$QEM_RESOLUTIONS"

for resolution in "${RESOLUTIONS[@]}"; do
  qem_dir="$ROOT/qem_edge_collapsed_meshes_$resolution"
  voxel_dir="$ROOT/triangle_field_voxels_$resolution"
  echo "Preparing QEM recovery at resolution $resolution"

  validation_rc=0
  "$PYTHON" data_toolkit/check_qem_edge_collapsed_dataset.py \
    --qem_root "$qem_dir" \
    --triangle_field_voxel_root "$voxel_dir" \
    --resolution "$resolution" \
    --num_workers "$QEM_CHECK_WORKERS" \
    --output_json "$qem_dir/validation_stats.json" \
    --invalid_instances "$qem_dir/invalid_instances.txt" \
    --failures_csv "$qem_dir/validation_failures.csv" || validation_rc=$?

  if [ "$validation_rc" -ne 0 ]; then
    echo "Validation found recoverable bad/missing files at resolution $resolution."
  fi

  recovery_args=(
    --root "$ROOT"
    --resolution "$resolution"
    --qem_root "$qem_dir"
    --triangle_field_voxel_root "$voxel_dir"
    --invalid_instances "$qem_dir/invalid_instances.txt"
    --output "$qem_dir/recovery_instances.txt"
    --summary_json "$qem_dir/recovery_summary.json"
  )
  if [ "$QEM_REQUIRE_CLEAN" = "1" ]; then
    recovery_args+=(--require_empty)
  fi
  "$PYTHON" data_toolkit/prepare_qem_recovery_instances.py "${recovery_args[@]}"
done
