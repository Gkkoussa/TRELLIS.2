#!/usr/bin/env bash
set -euo pipefail

FIELD="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst"
GT_OBJ="Mesh extraction/input.obj"
OUT_CSV="Mesh extraction/results/centroid_threshold_sweep.csv"

# Edit these lists. The script prints/sorts the closest centroid counts to the GT face count.
DTRI_THRESHOLDS="0.56,0.58,0.60,0.62,0.64"
MAX_DVERTS="0.52,0.55,0.58,0.61"
MIN_COMPONENT_SIZES="1,2,3,4"
CONNECTIVITY="6"
TOP_K="25"

cd "$(dirname "$0")/.."

python "Mesh extraction/sweep_centroid_thresholds.py" \
  "$FIELD" \
  --gt_obj "$GT_OBJ" \
  --dtri_thresholds "$DTRI_THRESHOLDS" \
  --max_dverts "$MAX_DVERTS" \
  --min_component_sizes "$MIN_COMPONENT_SIZES" \
  --connectivity "$CONNECTIVITY" \
  --out_csv "$OUT_CSV" \
  --top_k "$TOP_K"
