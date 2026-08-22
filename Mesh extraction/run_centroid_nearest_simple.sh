#!/usr/bin/env bash
set -euo pipefail

# Centroid-nearest mesh:
#   vertices = high d_vert blobs
#   centroids = high d_tri / low d_vert blobs
#   faces = each centroid connected to its 3 nearest vertices

INPUT="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst"
RESULTS_DIR="Mesh extraction/results"
OUT_PREFIX="heart_centroid_nearest"

VERTEX_DVERT_THRESHOLD="0.84"
VERTEX_POSITION_MODE="weighted"
VERTEX_CORE_MODE="closest4"

CENTROID_DTRI_THRESHOLD="0.56"
CENTROID_MAX_DVERT="0.55"
CENTROID_CONNECTIVITY="6"
CENTROID_MIN_COMPONENT_SIZE="3"
CENTROID_TARGET_COUNT="3080"
CENTROID_FILL_SMALL_COMPONENTS_BY="peak_dtri"
CENTROID_POSITION_MODE="weighted"
CENTROID_CORE_MODE="closest4"

# 0 means do not reject by centroid-to-vertex distance.
FACE_MAX_VERTEX_DISTANCE_VOXELS="0.0"

mkdir -p "$RESULTS_DIR"

cd "$(dirname "$0")/.."

python "Mesh extraction/build_centroid_nearest_faces_mesh.py" \
  "$INPUT" \
  --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD" \
  --vertex_position_mode "$VERTEX_POSITION_MODE" \
  --vertex_core_mode "$VERTEX_CORE_MODE" \
  --centroid_dtri_threshold "$CENTROID_DTRI_THRESHOLD" \
  --centroid_max_dvert "$CENTROID_MAX_DVERT" \
  --centroid_connectivity "$CENTROID_CONNECTIVITY" \
  --centroid_min_component_size "$CENTROID_MIN_COMPONENT_SIZE" \
  --centroid_target_count "$CENTROID_TARGET_COUNT" \
  --centroid_fill_small_components_by "$CENTROID_FILL_SMALL_COMPONENTS_BY" \
  --centroid_position_mode "$CENTROID_POSITION_MODE" \
  --centroid_core_mode "$CENTROID_CORE_MODE" \
  --face_max_vertex_distance_voxels "$FACE_MAX_VERTEX_DISTANCE_VOXELS" \
  --out_obj "$RESULTS_DIR/${OUT_PREFIX}.obj" \
  --out_ply "$RESULTS_DIR/${OUT_PREFIX}.ply" \
  --out_npz "$RESULTS_DIR/${OUT_PREFIX}.npz" \
  --out_csv "$RESULTS_DIR/${OUT_PREFIX}_faces.csv" \
  --out_core_ply "$RESULTS_DIR/${OUT_PREFIX}_cores_colored.ply"
