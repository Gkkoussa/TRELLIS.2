#!/usr/bin/env bash
#SBATCH --job-name=hier-vertex-targets
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/hier_vertex_targets_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --account=jjparkcv_owned1
#SBATCH --array=0-31%8

set -euo pipefail

PYTHON=/home/gpranav/pranav_work/scratch/envs/trellis2/bin/python
REPO=/home/gpranav/pranav_work/scratch/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
HIERARCHICAL_VERTEX_TARGET_ROOT="${HIERARCHICAL_VERTEX_TARGET_ROOT:-$ROOT}"
HIERARCHICAL_VERTEX_WORLD_SIZE="${HIERARCHICAL_VERTEX_WORLD_SIZE:-32}"
HIERARCHICAL_VERTEX_RESOLUTIONS="${HIERARCHICAL_VERTEX_RESOLUTIONS:-32 64 128 256 512}"
HIERARCHICAL_VERTEX_INSTANCES="${HIERARCHICAL_VERTEX_INSTANCES:-}"
HIERARCHICAL_VERTEX_OVERWRITE="${HIERARCHICAL_VERTEX_OVERWRITE:-0}"
read -r -a resolution_args <<< "$HIERARCHICAL_VERTEX_RESOLUTIONS"

if [ "$HIERARCHICAL_VERTEX_WORLD_SIZE" -ne 32 ]; then
  echo "HIERARCHICAL_VERTEX_WORLD_SIZE must match the Slurm array size (32)." >&2
  exit 2
fi
if [ "$HIERARCHICAL_VERTEX_OVERWRITE" != "0" ] && [ "$HIERARCHICAL_VERTEX_OVERWRITE" != "1" ]; then
  echo "HIERARCHICAL_VERTEX_OVERWRITE must be 0 or 1." >&2
  exit 2
fi
if [ "$HIERARCHICAL_VERTEX_OVERWRITE" = "1" ] && [ -z "$HIERARCHICAL_VERTEX_INSTANCES" ]; then
  echo "Refusing overwrite without HIERARCHICAL_VERTEX_INSTANCES." >&2
  exit 2
fi

echo "ROOT: $ROOT"
echo "Target root: $HIERARCHICAL_VERTEX_TARGET_ROOT"
echo "Resolutions: ${resolution_args[*]}"
echo "Shard: ${SLURM_ARRAY_TASK_ID}/${HIERARCHICAL_VERTEX_WORLD_SIZE}"
echo "Instances: ${HIERARCHICAL_VERTEX_INSTANCES:-all common triangle-field meshes}"
echo "Overwrite targeted instances: $HIERARCHICAL_VERTEX_OVERWRITE"

generation_args=(
  ObjaverseXL
  --root "$ROOT"
  --pbr_dump_root "$ROOT"
  --triangle_field_voxel_root "$ROOT"
  --hierarchical_vertex_target_root "$HIERARCHICAL_VERTEX_TARGET_ROOT"
  --resolutions "${resolution_args[@]}"
  --rank "$SLURM_ARRAY_TASK_ID"
  --world_size "$HIERARCHICAL_VERTEX_WORLD_SIZE"
  --zstd_level 3
)

if [ -n "$HIERARCHICAL_VERTEX_INSTANCES" ]; then
  if [ ! -f "$HIERARCHICAL_VERTEX_INSTANCES" ]; then
    echo "Instances file not found: $HIERARCHICAL_VERTEX_INSTANCES" >&2
    exit 2
  fi
  generation_args+=(--instances "$HIERARCHICAL_VERTEX_INSTANCES")
fi
if [ "$HIERARCHICAL_VERTEX_OVERWRITE" = "1" ]; then
  generation_args+=(--overwrite)
fi

"$PYTHON" data_toolkit/generate_hierarchical_vertex_targets.py "${generation_args[@]}"
