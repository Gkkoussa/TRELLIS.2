#!/usr/bin/env bash
set -euo pipefail

GT_OBJ="Mesh extraction/input.obj"
FIELD="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst"
RECON_NPZ="Mesh extraction/results/heart_centroid_nearest.npz"
OUT_PREFIX="Mesh extraction/results/heart_centroid_nearest_gt_diagnosis"

VERTEX_MATCH_RADIUS_VOXELS="4.0"
CENTROID_MATCH_RADIUS_VOXELS="6.0"
FIELD_PROBE_RADIUS_VOXELS="3"

cd "$(dirname "$0")/.."

python "Mesh extraction/diagnose_centroid_nearest_vs_gt.py" \
  --gt_obj "$GT_OBJ" \
  --field "$FIELD" \
  --recon_npz "$RECON_NPZ" \
  --out_prefix "$OUT_PREFIX" \
  --vertex_match_radius_voxels "$VERTEX_MATCH_RADIUS_VOXELS" \
  --centroid_match_radius_voxels "$CENTROID_MATCH_RADIUS_VOXELS" \
  --field_probe_radius_voxels "$FIELD_PROBE_RADIUS_VOXELS"
