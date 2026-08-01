#!/bin/bash
#SBATCH --job-name=trifield-nested512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"
MODE="${MODE:?Set MODE=base or MODE=density_elongation}"
INSTANCES="${INSTANCES:-$ROOT/instances_no_triangle_dense.txt}"
RESOLUTIONS="${RESOLUTIONS:-16,32,64,128,256}"
MAX_WORKERS="${MAX_WORKERS:-8}"

EXTRA_ARGS=()
case "$MODE" in
  base)
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/triangle_field_voxels_nested_from_512}"
    SOURCE_512="$ROOT/triangle_field_voxels_512"
    ;;
  density_elongation)
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/triangle_field_voxels_density_elongation_field_nested_from_512}"
    SOURCE_512="$ROOT/triangle_field_voxels_density_elongation_field/triangle_field_voxels_512"
    EXTRA_ARGS+=(--include_density_field --include_elongation_field)
    ;;
  *)
    echo "Unsupported MODE=$MODE; expected base or density_elongation" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT"
if [[ ! -e "$OUTPUT_ROOT/triangle_field_voxels_512" ]]; then
  ln -s "$SOURCE_512" "$OUTPUT_ROOT/triangle_field_voxels_512"
fi

python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$OUTPUT_ROOT" \
  --support_source_root "$ROOT" \
  --support_source_resolution 512 \
  --resolution "$RESOLUTIONS" \
  --instances "$INSTANCES" \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 5 \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers "$MAX_WORKERS" \
  "${EXTRA_ARGS[@]}"

python data_toolkit/build_metadata.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$OUTPUT_ROOT" \
  --field triangle_field_voxel

python data_toolkit/validate_nested_triangle_field_supports.py \
  --nested_root "$OUTPUT_ROOT" \
  --source_root "$ROOT" \
  --instances "$INSTANCES" \
  --resolutions 16,32,64,128,256,512 \
  --source_resolution 512 \
  --num_workers "$MAX_WORKERS" \
  --output_json "$OUTPUT_ROOT/nested_support_validation.json"
