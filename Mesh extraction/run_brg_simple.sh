#!/usr/bin/env bash
set -euo pipefail

# Edit these main thresholds/parameters.

INPUT="Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst"
RESULTS_DIR="Mesh extraction/results"
OUT_PREFIX="heart"

VERTEX_DVERT_THRESHOLD="0.84"
# Radius around the collapsed vertex core. Use 1 for small clearing, 0 for no clearing.
VERTEX_CLEARANCE="1"
VERTEX_POSITION_MODE="weighted"
# closest4 = detect each full vertex blob, keep only the 4 voxels closest to the vertex position.
VERTEX_CORE_MODE="closest4"

EDGE_DTRI_THRESHOLD="0.25"
EDGE_MIN_DVERT="0.10"
EDGE_MAX_DVERT="0.84"
EDGE_ATTACH_MODE="evidence"
ATTACH_RADIUS="3"

# Evidence connector: fit cyan edge blobs to candidate vertex-pair segments.
EDGE_EVIDENCE_CANDIDATE_VERTICES="8"
EDGE_EVIDENCE_MAX_SEGMENT_DISTANCE="3.0"
EDGE_EVIDENCE_MAX_PAIR_DISTANCE="64.0"
EDGE_EVIDENCE_PROJECTION_MARGIN="4.0"
# Join broken cyan edge components only when a small midpoint blob touches exactly two components.
EDGE_BRIDGE_MAX_DISTANCE="4.0"
EDGE_BRIDGE_MIDPOINT_MAX_COMPONENT_SIZE="128"

MIDPOINT_DTRI_THRESHOLD="0.275"
MIDPOINT_TOLERANCE="0.05"

# Set to 1 if you also want the colored voxel PLY.
MAKE_VOXEL_VIS="1"
# Set to 1 to save edge-looking connected components before contact filtering.
MAKE_STAGE3_EDGE_COMPONENTS="1"
# Set to 1 to include red vertex cores and orange clearance in the stage-3 PLY.
STAGE3_INCLUDE_VERTEX_CONTEXT="1"

mkdir -p "$RESULTS_DIR"

cd "$(dirname "$0")/.."

mesh_cmd=(
  python "Mesh extraction/build_brg_triangle_mesh.py"
  "$INPUT"
  --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
  --vertex_clearance "$VERTEX_CLEARANCE"
  --vertex_position_mode "$VERTEX_POSITION_MODE"
  --vertex_core_mode "$VERTEX_CORE_MODE"
  --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
  --edge_min_dvert "$EDGE_MIN_DVERT"
  --edge_attach_mode "$EDGE_ATTACH_MODE"
  --edge_evidence_candidate_vertices "$EDGE_EVIDENCE_CANDIDATE_VERTICES"
  --edge_evidence_max_segment_distance "$EDGE_EVIDENCE_MAX_SEGMENT_DISTANCE"
  --edge_evidence_max_pair_distance "$EDGE_EVIDENCE_MAX_PAIR_DISTANCE"
  --edge_evidence_projection_margin "$EDGE_EVIDENCE_PROJECTION_MARGIN"
  --edge_bridge_max_distance "$EDGE_BRIDGE_MAX_DISTANCE"
  --edge_bridge_midpoint_dtri_threshold "$MIDPOINT_DTRI_THRESHOLD"
  --edge_bridge_midpoint_tolerance "$MIDPOINT_TOLERANCE"
  --edge_bridge_midpoint_max_component_size "$EDGE_BRIDGE_MIDPOINT_MAX_COMPONENT_SIZE"
  --attach_radius "$ATTACH_RADIUS"
  --midpoint_dtri_threshold "$MIDPOINT_DTRI_THRESHOLD"
  --midpoint_tolerance "$MIDPOINT_TOLERANCE"
  --out_obj "$RESULTS_DIR/${OUT_PREFIX}_mesh.obj"
  --out_ply "$RESULTS_DIR/${OUT_PREFIX}_mesh.ply"
  --out_npz "$RESULTS_DIR/${OUT_PREFIX}_mesh.npz"
)

if [[ -n "$EDGE_MAX_DVERT" ]]; then
  mesh_cmd+=(--edge_max_dvert "$EDGE_MAX_DVERT")
fi

"${mesh_cmd[@]}"

if [[ "$MAKE_VOXEL_VIS" == "1" ]]; then
  voxel_cmd=(
    python "Mesh extraction/visualize_brg_voxels.py"
    "$INPUT"
    --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
    --vertex_clearance "$VERTEX_CLEARANCE"
    --vertex_position_mode "$VERTEX_POSITION_MODE"
    --vertex_core_mode "$VERTEX_CORE_MODE"
    --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
    --edge_min_dvert "$EDGE_MIN_DVERT"
    --edge_attach_mode "$EDGE_ATTACH_MODE"
    --attach_radius "$ATTACH_RADIUS"
    --midpoint_dtri_threshold "$MIDPOINT_DTRI_THRESHOLD"
    --midpoint_tolerance "$MIDPOINT_TOLERANCE"
    --out_ply "$RESULTS_DIR/${OUT_PREFIX}_voxels_colored.ply"
    --legend_csv "$RESULTS_DIR/${OUT_PREFIX}_voxels_legend.csv"
  )

  if [[ -n "$EDGE_MAX_DVERT" ]]; then
    voxel_cmd+=(--edge_max_dvert "$EDGE_MAX_DVERT")
  fi

  "${voxel_cmd[@]}"
fi

if [[ "$MAKE_STAGE3_EDGE_COMPONENTS" == "1" ]]; then
  stage3_cmd=(
    python "Mesh extraction/visualize_stage3_edge_components.py"
    "$INPUT"
    --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
    --vertex_clearance "$VERTEX_CLEARANCE"
    --vertex_position_mode "$VERTEX_POSITION_MODE"
    --vertex_core_mode "$VERTEX_CORE_MODE"
    --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
    --edge_min_dvert "$EDGE_MIN_DVERT"
    --out_ply "$RESULTS_DIR/${OUT_PREFIX}_stage3_edge_components.ply"
    --out_csv "$RESULTS_DIR/${OUT_PREFIX}_stage3_edge_components.csv"
  )

  if [[ "$STAGE3_INCLUDE_VERTEX_CONTEXT" == "1" ]]; then
    stage3_cmd+=(--include_vertex_context)
  fi

  if [[ -n "$EDGE_MAX_DVERT" ]]; then
    stage3_cmd+=(--edge_max_dvert "$EDGE_MAX_DVERT")
  fi

  "${stage3_cmd[@]}"
fi

echo "Done. Outputs:"
echo "  $RESULTS_DIR/${OUT_PREFIX}_mesh.obj"
echo "  $RESULTS_DIR/${OUT_PREFIX}_mesh.ply"
echo "  $RESULTS_DIR/${OUT_PREFIX}_mesh.npz"
if [[ "$MAKE_VOXEL_VIS" == "1" ]]; then
  echo "  $RESULTS_DIR/${OUT_PREFIX}_voxels_colored.ply"
  echo "  $RESULTS_DIR/${OUT_PREFIX}_voxels_legend.csv"
fi
if [[ "$MAKE_STAGE3_EDGE_COMPONENTS" == "1" ]]; then
  echo "  $RESULTS_DIR/${OUT_PREFIX}_stage3_edge_components.ply"
  echo "  $RESULTS_DIR/${OUT_PREFIX}_stage3_edge_components.csv"
fi
