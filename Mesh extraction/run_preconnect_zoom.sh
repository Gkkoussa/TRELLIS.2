#!/usr/bin/env bash
set -euo pipefail

# Edit exactly one target:
#   VERTEX_IDS="12,34,56"        # extracted vertex ids around a broken face
#   CENTER_WORLD="0.01,-0.2,0.3" # world-space center from viewer
#   CENTER_VOXEL="250,240,260"   # voxel-grid center

INPUT="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst"
RESULTS_DIR="Mesh extraction/results"

VERTEX_IDS=""
CENTER_WORLD=""
CENTER_VOXEL=""

RADIUS_VOXELS="28"
MARGIN_VOXELS="14"
INCLUDE_VERTEX_BLOB="1"

VERTEX_DVERT_THRESHOLD="0.84"
VERTEX_CLEARANCE="1"
VERTEX_POSITION_MODE="weighted"
VERTEX_CORE_MODE="closest4"

EDGE_DTRI_THRESHOLD="0.25"
EDGE_MIN_DVERT="0.10"
EDGE_MAX_DVERT="0.84"

mkdir -p "$RESULTS_DIR"

cd "$(dirname "$0")/.."

target_count=0
target_args=()
if [[ -n "$VERTEX_IDS" ]]; then
  target_args+=(--vertex_ids "$VERTEX_IDS")
  target_count=$((target_count + 1))
fi
if [[ -n "$CENTER_WORLD" ]]; then
  target_args+=(--center_world "$CENTER_WORLD")
  target_count=$((target_count + 1))
fi
if [[ -n "$CENTER_VOXEL" ]]; then
  target_args+=(--center_voxel "$CENTER_VOXEL")
  target_count=$((target_count + 1))
fi

if [[ "$target_count" != "1" ]]; then
  echo "Set exactly one of VERTEX_IDS, CENTER_WORLD, or CENTER_VOXEL in this script."
  exit 1
fi

zoom_cmd=(
  python "Mesh extraction/visualize_preconnect_zoom.py"
  "$INPUT"
  "${target_args[@]}"
  --radius_voxels "$RADIUS_VOXELS"
  --margin_voxels "$MARGIN_VOXELS"
  --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
  --vertex_clearance "$VERTEX_CLEARANCE"
  --vertex_position_mode "$VERTEX_POSITION_MODE"
  --vertex_core_mode "$VERTEX_CORE_MODE"
  --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
  --edge_min_dvert "$EDGE_MIN_DVERT"
)

if [[ -n "$EDGE_MAX_DVERT" ]]; then
  zoom_cmd+=(--edge_max_dvert "$EDGE_MAX_DVERT")
fi

if [[ "$INCLUDE_VERTEX_BLOB" == "1" ]]; then
  zoom_cmd+=(--include_vertex_blob)
fi

"${zoom_cmd[@]}"
