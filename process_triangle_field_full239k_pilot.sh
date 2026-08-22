#!/bin/bash
#SBATCH --job-name=trifield239k-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

PERSISTENT_BLENDER_ARCHIVE=/home/koussa/scratch/tools/blender-4.5.1-linux-x64.tar.xz
export BLENDER_INSTALLATION_PATH="${SLURM_TMPDIR:-/tmp}"
tar -xf "$PERSISTENT_BLENDER_ARCHIVE" -C "$BLENDER_INSTALLATION_PATH"
export BLENDER_PATH="$BLENDER_INSTALLATION_PATH/blender-4.5.1-linux-x64/blender"

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
NEW_ROOT="$ROOT/new_meshes"
DATASET_ROOT="$ROOT/triangle_field_full239k_avgfrom512_v2"
MANIFESTS="$DATASET_ROOT/manifests"
PILOT="$DATASET_ROOT/pilot/shard_00000"
STAGING="$PILOT/staging"

python data_toolkit/build_combined_triangle_field_manifests.py \
  --old_root "$ROOT" \
  --old_instances "$ROOT/splits/train_all_no_test/metadata.csv" \
  --new_root "$NEW_ROOT" \
  --output_dir "$MANIFESTS" \
  --holdout_size 1000 \
  --pilot_old 64 \
  --pilot_new 64

mkdir -p "$PILOT" "$STAGING"

# New meshes need the same normalized Blender dump consumed by the established voxelizer.
python data_toolkit/dump_pbr.py ObjaverseXL \
  --root "$NEW_ROOT" \
  --pbr_dump_root "$NEW_ROOT" \
  --instances "$MANIFESTS/pilot/new_instances.txt" \
  --max_workers 8

# Regenerate corrected 512 fields for both collections into one private staging tree.
python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$STAGING" \
  --resolution 512 \
  --instances "$MANIFESTS/pilot/old_instances.txt" \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 5 \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers 8

python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$NEW_ROOT" \
  --pbr_dump_root "$NEW_ROOT" \
  --triangle_field_voxel_root "$STAGING" \
  --resolution 512 \
  --instances "$MANIFESTS/pilot/new_instances.txt" \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 5 \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers 8

# Recompute coherent auxiliary channels on support derived exactly from corrected 512 support.
for SOURCE_ROOT in "$ROOT" "$NEW_ROOT"; do
  INSTANCE_FILE="$MANIFESTS/pilot/old_instances.txt"
  if [ "$SOURCE_ROOT" = "$NEW_ROOT" ]; then
    INSTANCE_FILE="$MANIFESTS/pilot/new_instances.txt"
  fi
  python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
    --root "$SOURCE_ROOT" \
    --pbr_dump_root "$SOURCE_ROOT" \
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
  --source_metadata "$MANIFESTS/pilot/metadata.csv" \
  --output_root "$STAGING" \
  --resolutions 32,64,128,256 \
  --instances "$MANIFESTS/pilot/instances.txt" \
  --output_suffix hybrid_avg_from_512 \
  --feature_mode average_targets_recompute_aux \
  --recomputed_root "$STAGING" \
  --feature_dtype float16 \
  --compression zstd \
  --zstd_level 5 \
  --device cpu \
  --max_workers 8 \
  --skip_existing

python data_toolkit/validate_hybrid_triangle_field_downsampling.py \
  --root "$STAGING" \
  --instances "$MANIFESTS/pilot/instances.txt" \
  --resolutions 32,64,128,256 \
  --hybrid_suffix hybrid_avg_from_512 \
  --output_json "$PILOT/hybrid_validation.json"

python data_toolkit/pack_triangle_field_zarr_shard.py \
  --staging_root "$STAGING" \
  --instances "$MANIFESTS/pilot/instances.txt" \
  --output "$PILOT/triangle_field_shard_00000.zarr.zip" \
  --resolutions 32,64,128,256,512 \
  --lower_suffix hybrid_avg_from_512 \
  --chunk_voxels 65536 \
  --compression_level 5 \
  --provenance "$MANIFESTS/manifest_summary.json" \
  --verify_existing

touch "$PILOT/COMPLETE"
echo "Pilot complete: $PILOT"
echo "Staging retained: $STAGING"
