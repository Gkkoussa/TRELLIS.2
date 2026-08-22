#!/usr/bin/env bash
#SBATCH --job-name=trifield-vertex-support
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/trifield_vertex_support_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --account=jjparkcv_owned1
#SBATCH --array=0-31%8

set -euo pipefail

PYTHON=/home/gpranav/pranav_work/scratch/envs/trellis2/bin/python
REPO=/home/gpranav/pranav_work/scratch/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
VERTEX_SUPPORT_WORLD_SIZE="${VERTEX_SUPPORT_WORLD_SIZE:-32}"
VERTEX_SUPPORT_RESOLUTIONS="${VERTEX_SUPPORT_RESOLUTIONS:-32 64 128 256 512}"
VERTEX_SUPPORT_WORKERS="${VERTEX_SUPPORT_WORKERS:-4}"
VERTEX_SUPPORT_EXAMPLES_PER_MESH="${VERTEX_SUPPORT_EXAMPLES_PER_MESH:-3}"
VERTEX_SUPPORT_OUTPUT="${VERTEX_SUPPORT_OUTPUT:-$ROOT/outputs/triangle_field_vertex_support_audit}"
VERTEX_SUPPORT_INSTANCES="${VERTEX_SUPPORT_INSTANCES:-}"
read -r -a resolution_args <<< "$VERTEX_SUPPORT_RESOLUTIONS"

if [ "$VERTEX_SUPPORT_WORLD_SIZE" -ne 32 ]; then
  echo "VERTEX_SUPPORT_WORLD_SIZE must match the Slurm array size (32)." >&2
  exit 2
fi

echo "ROOT: $ROOT"
echo "Resolutions: ${resolution_args[*]}"
echo "Shard: ${SLURM_ARRAY_TASK_ID}/${VERTEX_SUPPORT_WORLD_SIZE}"
echo "Workers: $VERTEX_SUPPORT_WORKERS"
echo "Output: $VERTEX_SUPPORT_OUTPUT"
echo "Instances: ${VERTEX_SUPPORT_INSTANCES:-all PBR-dumped metadata instances}"

audit_args=(
  audit
  --root "$ROOT"
  --resolutions "${resolution_args[@]}"
  --rank "$SLURM_ARRAY_TASK_ID"
  --world_size "$VERTEX_SUPPORT_WORLD_SIZE"
  --num_workers "$VERTEX_SUPPORT_WORKERS"
  --chunksize 2
  --examples_per_mesh "$VERTEX_SUPPORT_EXAMPLES_PER_MESH"
  --output_dir "$VERTEX_SUPPORT_OUTPUT"
)

if [ -n "$VERTEX_SUPPORT_INSTANCES" ]; then
  if [ ! -f "$VERTEX_SUPPORT_INSTANCES" ]; then
    echo "Instances file not found: $VERTEX_SUPPORT_INSTANCES" >&2
    exit 2
  fi
  audit_args+=(--instances "$VERTEX_SUPPORT_INSTANCES")
fi

"$PYTHON" data_toolkit/check_triangle_field_vertex_support.py "${audit_args[@]}"
