#!/usr/bin/env bash
set -euo pipefail

# Face-support mesh: vertices from high d_vert, faces from high d_tri blobs.

INPUT="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst"
RESULTS_DIR="Mesh extraction/results"
OUT_PREFIX="heart_face_support"

VERTEX_DVERT_THRESHOLD="0.84"
VERTEX_POSITION_MODE="weighted"
VERTEX_CORE_MODE="closest4"

# Face interiors: seed is the strong face-center threshold, grow includes nearby face interior.
FACE_DTRI_SEED_THRESHOLD="0.60"
FACE_DTRI_GROW_THRESHOLD="0.35"
FACE_MAX_DVERT="0.55"
FACE_COMPONENT_MASK="seed"
FACE_MIN_COMPONENT_SIZE="4"
FACE_MAX_COMPONENT_SIZE="0"

# Face-to-vertex assignment.
FACE_CANDIDATE_VERTICES="16"
FACE_CANDIDATE_SEARCH_RADIUS_VOXELS="40.0"
FACE_BARYCENTRIC_MARGIN="0.08"
FACE_MIN_INSIDE_FRACTION="0.60"
FACE_MAX_MEAN_PLANE_DISTANCE_VOXELS="3.0"
FACE_MAX_OUTSIDE_DEFICIT="0.18"
FACE_MAX_EDGE_LENGTH_VOXELS="48.0"
FACE_MAX_VERTEX_DISTANCE_VOXELS="32.0"
FACE_MAX_POINTS_PER_COMPONENT="256"

# Edge support gate: every accepted face must have low-d_tri edge voxels
# supporting all three proposed edges.
EDGE_SUPPORT_DTRI_THRESHOLD="0.25"
EDGE_SUPPORT_MIN_DVERT="0.10"
EDGE_SUPPORT_MAX_DVERT="0.84"
EDGE_SUPPORT_SEARCH_RADIUS_VOXELS="40.0"
EDGE_SUPPORT_MAX_DISTANCE_VOXELS="3.0"
EDGE_SUPPORT_PROJECTION_MARGIN_VOXELS="3.0"
EDGE_SUPPORT_MIN_VOXELS_PER_EDGE="3"

mkdir -p "$RESULTS_DIR"

cd "$(dirname "$0")/.."

python "Mesh extraction/build_face_support_mesh.py" \
  "$INPUT" \
  --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD" \
  --vertex_position_mode "$VERTEX_POSITION_MODE" \
  --vertex_core_mode "$VERTEX_CORE_MODE" \
  --face_dtri_seed_threshold "$FACE_DTRI_SEED_THRESHOLD" \
  --face_dtri_grow_threshold "$FACE_DTRI_GROW_THRESHOLD" \
  --face_max_dvert "$FACE_MAX_DVERT" \
  --face_component_mask "$FACE_COMPONENT_MASK" \
  --face_min_component_size "$FACE_MIN_COMPONENT_SIZE" \
  --face_max_component_size "$FACE_MAX_COMPONENT_SIZE" \
  --face_candidate_vertices "$FACE_CANDIDATE_VERTICES" \
  --face_candidate_search_radius_voxels "$FACE_CANDIDATE_SEARCH_RADIUS_VOXELS" \
  --face_barycentric_margin "$FACE_BARYCENTRIC_MARGIN" \
  --face_min_inside_fraction "$FACE_MIN_INSIDE_FRACTION" \
  --face_max_mean_plane_distance_voxels "$FACE_MAX_MEAN_PLANE_DISTANCE_VOXELS" \
  --face_max_outside_deficit "$FACE_MAX_OUTSIDE_DEFICIT" \
  --face_max_edge_length_voxels "$FACE_MAX_EDGE_LENGTH_VOXELS" \
  --face_max_vertex_distance_voxels "$FACE_MAX_VERTEX_DISTANCE_VOXELS" \
  --face_max_points_per_component "$FACE_MAX_POINTS_PER_COMPONENT" \
  --edge_support_dtri_threshold "$EDGE_SUPPORT_DTRI_THRESHOLD" \
  --edge_support_min_dvert "$EDGE_SUPPORT_MIN_DVERT" \
  --edge_support_max_dvert "$EDGE_SUPPORT_MAX_DVERT" \
  --edge_support_search_radius_voxels "$EDGE_SUPPORT_SEARCH_RADIUS_VOXELS" \
  --edge_support_max_distance_voxels "$EDGE_SUPPORT_MAX_DISTANCE_VOXELS" \
  --edge_support_projection_margin_voxels "$EDGE_SUPPORT_PROJECTION_MARGIN_VOXELS" \
  --edge_support_min_voxels_per_edge "$EDGE_SUPPORT_MIN_VOXELS_PER_EDGE" \
  --out_obj "$RESULTS_DIR/${OUT_PREFIX}.obj" \
  --out_ply "$RESULTS_DIR/${OUT_PREFIX}.ply" \
  --out_npz "$RESULTS_DIR/${OUT_PREFIX}.npz" \
  --out_csv "$RESULTS_DIR/${OUT_PREFIX}_faces.csv"
