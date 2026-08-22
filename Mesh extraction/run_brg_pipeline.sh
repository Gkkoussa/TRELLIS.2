#!/usr/bin/env bash
set -euo pipefail

# Barycentric Ridge Graph pipeline runner.
#
# Edit values below, or override them from the command line, e.g.
#
#   VERTEX_DVERT_THRESHOLD=0.88 ATTACH_RADIUS=4 ./Mesh\ extraction/run_brg_pipeline.sh
#
# Outputs go to Mesh extraction/results by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ---------------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------------

TRIANGLE_FIELD="${TRIANGLE_FIELD:-Mesh extraction/00146621_uploads_files_992018_LowPoly_Heart_obj_gt_triangle_field_voxels_512.npz.zst}"
RESULTS_DIR="${RESULTS_DIR:-Mesh extraction/results}"
PREFIX="${PREFIX:-heart}"
RESOLUTION="${RESOLUTION:-}"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------

RUN_MESH="${RUN_MESH:-1}"
RUN_VOXELS="${RUN_VOXELS:-1}"
RUN_ZOOM="${RUN_ZOOM:-0}"
RUN_GRAPH="${RUN_GRAPH:-0}"
RUN_VERTEX_SWEEP="${RUN_VERTEX_SWEEP:-0}"
RUN_EDGE_SWEEP="${RUN_EDGE_SWEEP:-0}"

# ---------------------------------------------------------------------------
# Vertex extraction parameters
# ---------------------------------------------------------------------------

VERTEX_DVERT_THRESHOLD="${VERTEX_DVERT_THRESHOLD:-0.84}"
VERTEX_CONNECTIVITY="${VERTEX_CONNECTIVITY:-18}"
VERTEX_MIN_COMPONENT_SIZE="${VERTEX_MIN_COMPONENT_SIZE:-1}"
VERTEX_POSITION_MODE="${VERTEX_POSITION_MODE:-weighted}"   # weighted, peak, mean

# ---------------------------------------------------------------------------
# Conservative edge-ridge pass parameters
# ---------------------------------------------------------------------------

EDGE_DTRI_THRESHOLD="${EDGE_DTRI_THRESHOLD:-0.175}"
EDGE_MIN_DVERT="${EDGE_MIN_DVERT:-0.25}"
EDGE_MAX_DVERT="${EDGE_MAX_DVERT:-}"                       # empty means vertex threshold
EDGE_CONNECTIVITY="${EDGE_CONNECTIVITY:-18}"
EDGE_MIN_COMPONENT_SIZE="${EDGE_MIN_COMPONENT_SIZE:-2}"
VERTEX_CLEARANCE="${VERTEX_CLEARANCE:-1}"
ATTACH_RADIUS="${ATTACH_RADIUS:-3}"
RIDGE_MIDPOINT_DVERT="${RIDGE_MIDPOINT_DVERT:-0.25}"
RIDGE_MIDPOINT_TOLERANCE="${RIDGE_MIDPOINT_TOLERANCE:-0.075}"
NO_RIDGE_MIDPOINT_CHECK="${NO_RIDGE_MIDPOINT_CHECK:-0}"

# ---------------------------------------------------------------------------
# Midpoint completion parameters
# ---------------------------------------------------------------------------

NO_MIDPOINT_COMPLETION="${NO_MIDPOINT_COMPLETION:-0}"
MIDPOINT_DTRI_THRESHOLD="${MIDPOINT_DTRI_THRESHOLD:-0.25}"
MIDPOINT_DVERT="${MIDPOINT_DVERT:-0.25}"
MIDPOINT_TOLERANCE="${MIDPOINT_TOLERANCE:-0.05}"
MIDPOINT_CONNECTIVITY="${MIDPOINT_CONNECTIVITY:-18}"
MIDPOINT_MIN_COMPONENT_SIZE="${MIDPOINT_MIN_COMPONENT_SIZE:-1}"
MAX_ANCHOR_DISTANCE="${MAX_ANCHOR_DISTANCE:-}"             # empty means no max-distance filter

# ---------------------------------------------------------------------------
# Face / mesh parameters
# ---------------------------------------------------------------------------

MIN_FACE_AREA="${MIN_FACE_AREA:-1e-12}"
NO_ORIENT_OUTWARD="${NO_ORIENT_OUTWARD:-0}"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

VERTEX_SWEEP_THRESHOLDS="${VERTEX_SWEEP_THRESHOLDS:-0.90,0.88,0.86,0.84,0.82,0.80}"
EDGE_SWEEP_DTRI_THRESHOLDS="${EDGE_SWEEP_DTRI_THRESHOLDS:-0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25}"

# ---------------------------------------------------------------------------
# Zoom raster parameters
# ---------------------------------------------------------------------------

FACE_INDEX="${FACE_INDEX:-}"                               # empty chooses by FACE_MODE
FACE_MODE="${FACE_MODE:-p75}"                              # largest, median, p75
MARGIN_VOXELS="${MARGIN_VOXELS:-14}"
DPI="${DPI:-220}"

add_optional_arg() {
  local -n arr_ref="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    arr_ref+=("$flag" "$value")
  fi
}

add_bool_arg() {
  local -n arr_ref="$1"
  local flag="$2"
  local value="$3"
  if [[ "$value" == "1" || "$value" == "true" || "$value" == "TRUE" ]]; then
    arr_ref+=("$flag")
  fi
}

common_args=(
  "$TRIANGLE_FIELD"
)
add_optional_arg common_args "--resolution" "$RESOLUTION"

mesh_args=(
  "${common_args[@]}"
  --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
  --vertex_connectivity "$VERTEX_CONNECTIVITY"
  --vertex_min_component_size "$VERTEX_MIN_COMPONENT_SIZE"
  --vertex_position_mode "$VERTEX_POSITION_MODE"
  --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
  --edge_min_dvert "$EDGE_MIN_DVERT"
  --edge_connectivity "$EDGE_CONNECTIVITY"
  --edge_min_component_size "$EDGE_MIN_COMPONENT_SIZE"
  --vertex_clearance "$VERTEX_CLEARANCE"
  --attach_radius "$ATTACH_RADIUS"
  --ridge_midpoint_dvert "$RIDGE_MIDPOINT_DVERT"
  --ridge_midpoint_tolerance "$RIDGE_MIDPOINT_TOLERANCE"
  --midpoint_dtri_threshold "$MIDPOINT_DTRI_THRESHOLD"
  --midpoint_dvert "$MIDPOINT_DVERT"
  --midpoint_tolerance "$MIDPOINT_TOLERANCE"
  --midpoint_connectivity "$MIDPOINT_CONNECTIVITY"
  --midpoint_min_component_size "$MIDPOINT_MIN_COMPONENT_SIZE"
  --min_face_area "$MIN_FACE_AREA"
)
add_optional_arg mesh_args "--edge_max_dvert" "$EDGE_MAX_DVERT"
add_optional_arg mesh_args "--max_anchor_distance" "$MAX_ANCHOR_DISTANCE"
add_bool_arg mesh_args "--no_ridge_midpoint_check" "$NO_RIDGE_MIDPOINT_CHECK"
add_bool_arg mesh_args "--no_midpoint_completion" "$NO_MIDPOINT_COMPLETION"
add_bool_arg mesh_args "--no_orient_outward" "$NO_ORIENT_OUTWARD"

voxel_args=(
  "${common_args[@]}"
  --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
  --vertex_connectivity "$VERTEX_CONNECTIVITY"
  --vertex_min_component_size "$VERTEX_MIN_COMPONENT_SIZE"
  --vertex_position_mode "$VERTEX_POSITION_MODE"
  --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
  --edge_min_dvert "$EDGE_MIN_DVERT"
  --edge_connectivity "$EDGE_CONNECTIVITY"
  --edge_min_component_size "$EDGE_MIN_COMPONENT_SIZE"
  --vertex_clearance "$VERTEX_CLEARANCE"
  --attach_radius "$ATTACH_RADIUS"
  --ridge_midpoint_dvert "$RIDGE_MIDPOINT_DVERT"
  --ridge_midpoint_tolerance "$RIDGE_MIDPOINT_TOLERANCE"
  --midpoint_dtri_threshold "$MIDPOINT_DTRI_THRESHOLD"
  --midpoint_dvert "$MIDPOINT_DVERT"
  --midpoint_tolerance "$MIDPOINT_TOLERANCE"
)
add_optional_arg voxel_args "--edge_max_dvert" "$EDGE_MAX_DVERT"
add_bool_arg voxel_args "--no_ridge_midpoint_check" "$NO_RIDGE_MIDPOINT_CHECK"

graph_args=(
  "${common_args[@]}"
  --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
  --vertex_connectivity "$VERTEX_CONNECTIVITY"
  --vertex_min_component_size "$VERTEX_MIN_COMPONENT_SIZE"
  --vertex_position_mode "$VERTEX_POSITION_MODE"
  --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
  --edge_min_dvert "$EDGE_MIN_DVERT"
  --edge_connectivity "$EDGE_CONNECTIVITY"
  --edge_min_component_size "$EDGE_MIN_COMPONENT_SIZE"
  --vertex_clearance "$VERTEX_CLEARANCE"
  --attach_radius "$ATTACH_RADIUS"
  --midpoint_dvert "$RIDGE_MIDPOINT_DVERT"
  --midpoint_tolerance "$RIDGE_MIDPOINT_TOLERANCE"
)
add_optional_arg graph_args "--edge_max_dvert" "$EDGE_MAX_DVERT"
add_bool_arg graph_args "--no_midpoint_check" "$NO_RIDGE_MIDPOINT_CHECK"

if [[ "$RUN_VERTEX_SWEEP" == "1" ]]; then
  python 'Mesh extraction/extract_vertices_threshold.py' \
    "$TRIANGLE_FIELD" \
    --thresholds "$VERTEX_SWEEP_THRESHOLDS" \
    --connectivity "$VERTEX_CONNECTIVITY" \
    --position_mode "$VERTEX_POSITION_MODE" \
    --min_component_size "$VERTEX_MIN_COMPONENT_SIZE" \
    --sweep_out_csv "$RESULTS_DIR/${PREFIX}_vertices_threshold_sweep.csv"
fi

if [[ "$RUN_EDGE_SWEEP" == "1" ]]; then
  python 'Mesh extraction/extract_barycentric_ridge_graph.py' \
    "${graph_args[@]}" \
    --sweep_edge_dtri "$EDGE_SWEEP_DTRI_THRESHOLDS" \
    --sweep_out_csv "$RESULTS_DIR/${PREFIX}_brg_edge_dtri_sweep.csv"
fi

if [[ "$RUN_GRAPH" == "1" ]]; then
  python 'Mesh extraction/extract_barycentric_ridge_graph.py' \
    "${graph_args[@]}" \
    --out_npz "$RESULTS_DIR/${PREFIX}_brg_graph.npz" \
    --out_ply "$RESULTS_DIR/${PREFIX}_brg_graph.ply"
fi

if [[ "$RUN_MESH" == "1" ]]; then
  python 'Mesh extraction/build_brg_triangle_mesh.py' \
    "${mesh_args[@]}" \
    --out_obj "$RESULTS_DIR/${PREFIX}_brg_triangle_mesh.obj" \
    --out_ply "$RESULTS_DIR/${PREFIX}_brg_triangle_mesh.ply" \
    --out_npz "$RESULTS_DIR/${PREFIX}_brg_triangle_mesh.npz"
fi

if [[ "$RUN_VOXELS" == "1" ]]; then
  python 'Mesh extraction/visualize_brg_voxels.py' \
    "${voxel_args[@]}" \
    --out_ply "$RESULTS_DIR/${PREFIX}_brg_voxels_colored.ply" \
    --legend_csv "$RESULTS_DIR/${PREFIX}_brg_voxels_legend.csv"
fi

if [[ "$RUN_ZOOM" == "1" ]]; then
  zoom_args=(
    "${common_args[@]}"
    --mesh_npz "$RESULTS_DIR/${PREFIX}_brg_triangle_mesh.npz"
    --face_mode "$FACE_MODE"
    --margin_voxels "$MARGIN_VOXELS"
    --dpi "$DPI"
    --vertex_dvert_threshold "$VERTEX_DVERT_THRESHOLD"
    --vertex_connectivity "$VERTEX_CONNECTIVITY"
    --edge_dtri_threshold "$EDGE_DTRI_THRESHOLD"
    --edge_min_dvert "$EDGE_MIN_DVERT"
    --edge_connectivity "$EDGE_CONNECTIVITY"
    --vertex_clearance "$VERTEX_CLEARANCE"
    --attach_radius "$ATTACH_RADIUS"
    --ridge_midpoint_dvert "$RIDGE_MIDPOINT_DVERT"
    --ridge_midpoint_tolerance "$RIDGE_MIDPOINT_TOLERANCE"
    --midpoint_dtri_threshold "$MIDPOINT_DTRI_THRESHOLD"
    --midpoint_dvert "$MIDPOINT_DVERT"
    --midpoint_tolerance "$MIDPOINT_TOLERANCE"
    --out_png "$RESULTS_DIR/${PREFIX}_brg_zoom.png"
  )
  add_optional_arg zoom_args "--edge_max_dvert" "$EDGE_MAX_DVERT"
  add_optional_arg zoom_args "--face_index" "$FACE_INDEX"

  python 'Mesh extraction/rasterize_brg_zoom.py' "${zoom_args[@]}"
fi

echo "Done. Results are in: $RESULTS_DIR"
