#!/usr/bin/env bash
#SBATCH --job-name=finalize-vertex-support
#SBATCH --output=/home/gpranav/pranav_work/scratch/TRELLIS.2/triangle_job_logs/finalize_vertex_support_%j.log
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --account=jjparkcv_owned1

set -euo pipefail

PYTHON=/home/gpranav/pranav_work/scratch/envs/trellis2/bin/python
REPO=/home/gpranav/pranav_work/scratch/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
VERTEX_SUPPORT_WORLD_SIZE="${VERTEX_SUPPORT_WORLD_SIZE:-32}"
VERTEX_SUPPORT_OUTPUT="${VERTEX_SUPPORT_OUTPUT:-$ROOT/outputs/triangle_field_vertex_support_audit}"

echo "Merging $VERTEX_SUPPORT_WORLD_SIZE shards from: $VERTEX_SUPPORT_OUTPUT"

"$PYTHON" data_toolkit/check_triangle_field_vertex_support.py finalize \
  --output_dir "$VERTEX_SUPPORT_OUTPUT" \
  --world_size "$VERTEX_SUPPORT_WORLD_SIZE"
