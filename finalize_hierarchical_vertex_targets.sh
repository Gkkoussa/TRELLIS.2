#!/usr/bin/env bash
#SBATCH --job-name=finalize-hier-vertex
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/finalize_hier_vertex_%j.log
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --account=jjparkcv_owned1

set -euo pipefail

PYTHON=/home/gpranav/pranav_work/scratch/envs/trellis2/bin/python
REPO=/home/gpranav/pranav_work/scratch/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
HIERARCHICAL_VERTEX_TARGET_ROOT="${HIERARCHICAL_VERTEX_TARGET_ROOT:-$ROOT}"
HIERARCHICAL_VERTEX_RESOLUTIONS="${HIERARCHICAL_VERTEX_RESOLUTIONS:-32 64 128 256 512}"
HIERARCHICAL_VERTEX_CHECK_WORKERS="${HIERARCHICAL_VERTEX_CHECK_WORKERS:-8}"
HIERARCHICAL_VERTEX_INSTANCES="${HIERARCHICAL_VERTEX_INSTANCES:-}"
HIERARCHICAL_VERTEX_SKIP_SUPPORT_CHECK="${HIERARCHICAL_VERTEX_SKIP_SUPPORT_CHECK:-0}"
read -r -a resolution_args <<< "$HIERARCHICAL_VERTEX_RESOLUTIONS"

echo "ROOT: $ROOT"
echo "Target root: $HIERARCHICAL_VERTEX_TARGET_ROOT"
echo "Resolutions: ${resolution_args[*]}"

"$PYTHON" data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --triangle_field_voxel_root "$ROOT" \
  --qem_edge_collapsed_root "$ROOT" \
  --hierarchical_vertex_target_root "$HIERARCHICAL_VERTEX_TARGET_ROOT"

validation_args=(
  --root "$ROOT"
  --hierarchical_vertex_target_root "$HIERARCHICAL_VERTEX_TARGET_ROOT"
  --triangle_field_voxel_root "$ROOT"
  --resolutions "${resolution_args[@]}"
  --num_workers "$HIERARCHICAL_VERTEX_CHECK_WORKERS"
  --chunksize 2
)

if [ -n "$HIERARCHICAL_VERTEX_INSTANCES" ]; then
  if [ ! -f "$HIERARCHICAL_VERTEX_INSTANCES" ]; then
    echo "Instances file not found: $HIERARCHICAL_VERTEX_INSTANCES" >&2
    exit 2
  fi
  validation_args+=(--instances "$HIERARCHICAL_VERTEX_INSTANCES")
fi
if [ "$HIERARCHICAL_VERTEX_SKIP_SUPPORT_CHECK" = "1" ]; then
  validation_args+=(--skip_support_check)
elif [ "$HIERARCHICAL_VERTEX_SKIP_SUPPORT_CHECK" != "0" ]; then
  echo "HIERARCHICAL_VERTEX_SKIP_SUPPORT_CHECK must be 0 or 1." >&2
  exit 2
fi

"$PYTHON" data_toolkit/check_hierarchical_vertex_targets.py "${validation_args[@]}"
