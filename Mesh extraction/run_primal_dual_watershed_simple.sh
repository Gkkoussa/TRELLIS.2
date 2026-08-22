#!/usr/bin/env bash
set -euo pipefail

# Primal/dual watershed mesh:
#   F = d_tri grows face basins
#   D = d_vert grows vertex basins
#   each F basin becomes a face from its three dominant D labels

INPUT="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst"
RESULTS_DIR="Mesh extraction/results"
OUT_PREFIX="heart_primal_dual_watershed"

# voxel = raw field axes. input = rotate output coordinates to input.obj axes: (x, z, -y).
OUTPUT_AXES="input"

# Primal face seeds from F=d_tri.
FACE_SEED_THRESHOLD="0.5" #this was 0.56 before
FACE_SEED_CONNECTIVITY="18"
FACE_SEED_MIN_COMPONENT_SIZE="1"
FACE_SEED_TARGET_COUNT="0"
FACE_SEED_FILL_SMALL_BY="none"
FACE_SEED_POSITION_MODE="weighted"
FACE_SEED_CORE_MODE="none"

# Dual vertex seeds from D=d_vert.
VERTEX_SEED_THRESHOLD="0.84"
VERTEX_SEED_CONNECTIVITY="18"
VERTEX_SEED_MIN_COMPONENT_SIZE="1"
VERTEX_SEED_TARGET_COUNT="0"
VERTEX_SEED_FILL_SMALL_BY="none"
VERTEX_SEED_POSITION_MODE="weighted"
VERTEX_SEED_CORE_MODE="none"

# -1 disables fake-boundary merging. Try 0.20 later if F oversegments faces.
MERGE_FAKE_BOUNDARIES_F_THRESHOLD="-1.0"

# top3 means every face basin uses its three dominant dual vertex labels.
# top4_split makes two adjacent triangles when the 4th label is close to the 3rd.
# exact3 rejects basins that touch more or fewer than exactly three dual labels.
FACE_CORNER_POLICY="top4_split"
MIN_CORNER_VOXELS="1"
TOP4_SPLIT_RATIO="0.5"

mkdir -p "$RESULTS_DIR"

cd "$(dirname "$0")/.."

python "Mesh extraction/build_primal_dual_watershed_mesh.py" \
  "$INPUT" \
  --face_seed_threshold "$FACE_SEED_THRESHOLD" \
  --face_seed_connectivity "$FACE_SEED_CONNECTIVITY" \
  --face_seed_min_component_size "$FACE_SEED_MIN_COMPONENT_SIZE" \
  --face_seed_target_count "$FACE_SEED_TARGET_COUNT" \
  --face_seed_fill_small_by "$FACE_SEED_FILL_SMALL_BY" \
  --face_seed_position_mode "$FACE_SEED_POSITION_MODE" \
  --face_seed_core_mode "$FACE_SEED_CORE_MODE" \
  --vertex_seed_threshold "$VERTEX_SEED_THRESHOLD" \
  --vertex_seed_connectivity "$VERTEX_SEED_CONNECTIVITY" \
  --vertex_seed_min_component_size "$VERTEX_SEED_MIN_COMPONENT_SIZE" \
  --vertex_seed_target_count "$VERTEX_SEED_TARGET_COUNT" \
  --vertex_seed_fill_small_by "$VERTEX_SEED_FILL_SMALL_BY" \
  --vertex_seed_position_mode "$VERTEX_SEED_POSITION_MODE" \
  --vertex_seed_core_mode "$VERTEX_SEED_CORE_MODE" \
  --merge_fake_boundaries_f_threshold "$MERGE_FAKE_BOUNDARIES_F_THRESHOLD" \
  --face_corner_policy "$FACE_CORNER_POLICY" \
  --min_corner_voxels "$MIN_CORNER_VOXELS" \
  --top4_split_ratio "$TOP4_SPLIT_RATIO" \
  --output_axes "$OUTPUT_AXES" \
  --out_obj "$RESULTS_DIR/${OUT_PREFIX}.obj" \
  --out_ply "$RESULTS_DIR/${OUT_PREFIX}.ply" \
  --out_npz "$RESULTS_DIR/${OUT_PREFIX}.npz" \
  --out_csv "$RESULTS_DIR/${OUT_PREFIX}_faces.csv" \
  --out_debug_ply "$RESULTS_DIR/${OUT_PREFIX}_debug_voxels.ply"
