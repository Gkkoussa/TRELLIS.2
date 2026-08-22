#!/usr/bin/env bash
#SBATCH --job-name=qem_edge_collapse
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/qem_edge_collapse_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --account=jjparkcv_owned2
#SBATCH --array=0-31%8

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
export QEM_WORLD_SIZE="${QEM_WORLD_SIZE:-32}"
export QEM_INSTANCES="${QEM_INSTANCES:-}"
export QEM_OVERWRITE="${QEM_OVERWRITE:-0}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qem_matplotlib_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

if [ "$QEM_WORLD_SIZE" -ne 32 ]; then
  echo "QEM_WORLD_SIZE must match the Slurm array size (32)." >&2
  exit 2
fi

echo "ROOT: $ROOT"
echo "Resolution: $QEM_RESOLUTION"
echo "Shard: ${SLURM_ARRAY_TASK_ID}/${QEM_WORLD_SIZE}"
echo "Support assignment: nearest active voxel (no maximum distance)"
echo "Instances: ${QEM_INSTANCES:-all eligible instances}"
echo "Overwrite selected instances: $QEM_OVERWRITE"

qem_args=(
  ObjaverseXL
  --root "$ROOT"
  --pbr_dump_root "$ROOT"
  --triangle_field_voxel_root "$ROOT"
  --qem_edge_collapsed_root "$ROOT"
  --resolution "$QEM_RESOLUTION"
  --rank "$SLURM_ARRAY_TASK_ID"
  --world_size "$QEM_WORLD_SIZE"
  --boundary_weight 1.0
  --zstd_level 3
)

if [ -n "$QEM_INSTANCES" ]; then
  if [ ! -f "$QEM_INSTANCES" ]; then
    echo "QEM instances file does not exist: $QEM_INSTANCES" >&2
    exit 2
  fi
  qem_args+=(--instances "$QEM_INSTANCES")
fi

if [ "$QEM_OVERWRITE" = "1" ]; then
  if [ -z "$QEM_INSTANCES" ]; then
    echo "Refusing QEM_OVERWRITE=1 without a targeted QEM_INSTANCES file." >&2
    exit 2
  fi
  qem_args+=(--overwrite)
elif [ "$QEM_OVERWRITE" != "0" ]; then
  echo "QEM_OVERWRITE must be 0 or 1." >&2
  exit 2
fi

"$PYTHON" data_toolkit/qem_edge_collapse.py "${qem_args[@]}"
