#!/bin/bash
#SBATCH --job-name=trifield239k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
NEW_ROOT="$ROOT/new_meshes"
DATASET_ROOT="$ROOT/triangle_field_full239k_avgfrom512_v2"
MANIFESTS="$DATASET_ROOT/manifests"
EXCLUDED_INSTANCES="${EXCLUDED_INSTANCES:-$PWD/data_toolkit/triangle_field_full239k_excluded_instances.txt}"
SHARDS_PER_TASK="${SHARDS_PER_TASK:-4}"
TOTAL_SHARDS=$(($(wc -l < "$MANIFESTS/shards/index.csv") - 1))
FIRST_SHARD=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
if [ "$FIRST_SHARD" -ge "$TOTAL_SHARDS" ]; then
  echo "Task $SLURM_ARRAY_TASK_ID starts beyond $TOTAL_SHARDS shards"
  exit 0
fi

PENDING_SHARD_IDS=()
for ((INDEX = FIRST_SHARD; INDEX < FIRST_SHARD + SHARDS_PER_TASK && INDEX < TOTAL_SHARDS; INDEX++)); do
  SHARD_ID=$(printf 'shard_%05d' "$INDEX")
  if [ ! -f "$DATASET_ROOT/shards/$SHARD_ID/COMPLETE" ]; then
    PENDING_SHARD_IDS+=("$SHARD_ID")
  fi
done
if [ "${#PENDING_SHARD_IDS[@]}" -eq 0 ]; then
  echo "All shards for task $SLURM_ARRAY_TASK_ID are already complete"
  exit 0
fi

LOCAL_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp/$USER/slurm-$SLURM_JOB_ID}}"
WORK_ROOT="$LOCAL_TMP_ROOT/task_$(printf '%05d' "$SLURM_ARRAY_TASK_ID")"
STAGING="$WORK_ROOT/staging"
PBR_STAGING="$WORK_ROOT/pbr"
TASK_MANIFEST="$WORK_ROOT/manifest"
mkdir -p "$STAGING" "$PBR_STAGING" "$TASK_MANIFEST"
trap 'rm -rf "$WORK_ROOT"' EXIT

if [ ! -f "$EXCLUDED_INSTANCES" ]; then
  echo "Missing exclusion list: $EXCLUDED_INSTANCES" >&2
  exit 1
fi

filter_instances() {
  grep -F -x -v -f "$EXCLUDED_INSTANCES" "$1" > "$2" || true
}

filter_metadata() {
  awk -F, 'NR==FNR { excluded[$1] = 1; next } FNR == 1 || !($1 in excluded)' \
    "$EXCLUDED_INSTANCES" "$1" > "$2"
}

for NAME in instances old_instances new_instances; do
  : > "$TASK_MANIFEST/$NAME.txt"
done
FILTERED_SHARD_MANIFESTS="$WORK_ROOT/shard_manifests"
mkdir -p "$FILTERED_SHARD_MANIFESTS"
FIRST_MANIFEST="$MANIFESTS/shards/${PENDING_SHARD_IDS[0]}/metadata.csv"
head -n 1 "$FIRST_MANIFEST" > "$TASK_MANIFEST/metadata.csv"
for SHARD_ID in "${PENDING_SHARD_IDS[@]}"; do
  SOURCE_SHARD_MANIFEST="$MANIFESTS/shards/$SHARD_ID"
  SHARD_MANIFEST="$FILTERED_SHARD_MANIFESTS/$SHARD_ID"
  mkdir -p "$SHARD_MANIFEST"
  filter_instances "$SOURCE_SHARD_MANIFEST/instances.txt" "$SHARD_MANIFEST/instances.txt"
  filter_instances "$SOURCE_SHARD_MANIFEST/old_instances.txt" "$SHARD_MANIFEST/old_instances.txt"
  filter_instances "$SOURCE_SHARD_MANIFEST/new_instances.txt" "$SHARD_MANIFEST/new_instances.txt"
  filter_metadata "$SOURCE_SHARD_MANIFEST/metadata.csv" "$SHARD_MANIFEST/metadata.csv"
  cat "$SHARD_MANIFEST/instances.txt" >> "$TASK_MANIFEST/instances.txt"
  cat "$SHARD_MANIFEST/old_instances.txt" >> "$TASK_MANIFEST/old_instances.txt"
  cat "$SHARD_MANIFEST/new_instances.txt" >> "$TASK_MANIFEST/new_instances.txt"
  tail -n +2 "$SHARD_MANIFEST/metadata.csv" >> "$TASK_MANIFEST/metadata.csv"
done

if [ -s "$TASK_MANIFEST/new_instances.txt" ]; then
  PERSISTENT_BLENDER_ARCHIVE=/home/koussa/scratch/tools/blender-4.5.1-linux-x64.tar.xz
  export BLENDER_INSTALLATION_PATH="$WORK_ROOT/blender"
  mkdir -p "$BLENDER_INSTALLATION_PATH"
  tar -xf "$PERSISTENT_BLENDER_ARCHIVE" -C "$BLENDER_INSTALLATION_PATH"
  export BLENDER_PATH="$BLENDER_INSTALLATION_PATH/blender-4.5.1-linux-x64/blender"
  python data_toolkit/dump_pbr.py ObjaverseXL \
    --root "$NEW_ROOT" \
    --pbr_dump_root "$PBR_STAGING" \
    --instances "$TASK_MANIFEST/new_instances.txt" \
    --max_workers 8
fi

if [ -s "$TASK_MANIFEST/old_instances.txt" ]; then
  python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
    --root "$ROOT" \
    --pbr_dump_root "$ROOT" \
    --triangle_field_voxel_root "$STAGING" \
    --resolution 512 \
    --instances "$TASK_MANIFEST/old_instances.txt" \
    --feature_dtype float16 \
    --npz_compression zstd \
    --zstd_level 5 \
    --candidate_source native \
    --projection_mode inside_barycentric \
    --max_workers 8
fi

if [ -s "$TASK_MANIFEST/new_instances.txt" ]; then
  python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
    --root "$NEW_ROOT" \
    --pbr_dump_root "$PBR_STAGING" \
    --triangle_field_voxel_root "$STAGING" \
    --resolution 512 \
    --instances "$TASK_MANIFEST/new_instances.txt" \
    --feature_dtype float16 \
    --npz_compression zstd \
    --zstd_level 5 \
    --candidate_source native \
    --projection_mode inside_barycentric \
    --max_workers 8
fi

for COLLECTION in old new; do
  INSTANCE_FILE="$TASK_MANIFEST/${COLLECTION}_instances.txt"
  if [ ! -s "$INSTANCE_FILE" ]; then
    continue
  fi
  SOURCE_ROOT="$ROOT"
  PBR_ROOT="$ROOT"
  if [ "$COLLECTION" = new ]; then
    SOURCE_ROOT="$NEW_ROOT"
    PBR_ROOT="$PBR_STAGING"
  fi
  python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
    --root "$SOURCE_ROOT" \
    --pbr_dump_root "$PBR_ROOT" \
    --triangle_field_voxel_root "$STAGING" \
    --support_source_root "$STAGING" \
    --support_source_resolution 512 \
    --resolution 32,64,128,256 \
    --instances "$INSTANCE_FILE" \
    --feature_dtype float16 \
    --npz_compression zstd \
    --zstd_level 5 \
    --candidate_source native \
    --projection_mode inside_barycentric \
    --max_workers 8
done

python data_toolkit/downsample_triangle_field_voxels.py \
  --source_root "$STAGING/triangle_field_voxels_512" \
  --source_metadata "$TASK_MANIFEST/metadata.csv" \
  --output_root "$STAGING" \
  --resolutions 32,64,128,256 \
  --instances "$TASK_MANIFEST/instances.txt" \
  --output_suffix hybrid_avg_from_512 \
  --feature_mode average_targets_recompute_aux \
  --recomputed_root "$STAGING" \
  --feature_dtype float16 \
  --compression zstd \
  --zstd_level 5 \
  --device cpu \
  --max_workers 8

for SHARD_ID in "${PENDING_SHARD_IDS[@]}"; do
  SHARD_MANIFEST="$FILTERED_SHARD_MANIFESTS/$SHARD_ID"
  OUTPUT_DIR="$DATASET_ROOT/shards/$SHARD_ID"
  mkdir -p "$OUTPUT_DIR"
  python data_toolkit/pack_triangle_field_zarr_shard.py \
    --staging_root "$STAGING" \
    --instances "$SHARD_MANIFEST/instances.txt" \
    --output "$OUTPUT_DIR/triangle_field_${SHARD_ID}.zarr.zip" \
    --resolutions 32,64,128,256,512 \
    --lower_suffix hybrid_avg_from_512 \
    --chunk_voxels 65536 \
    --compression_level 5 \
    --provenance "$MANIFESTS/manifest_summary.json" \
    --verify_existing \
    --no-staging-retained
  touch "$OUTPUT_DIR/COMPLETE"
  echo "Completed $SHARD_ID"
done
