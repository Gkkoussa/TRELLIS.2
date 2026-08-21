#!/bin/bash
#SBATCH --job-name=trifield239k-pack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=standard
#SBATCH --account=jjparkcv_owned1

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
DATASET_ROOT="$ROOT/triangle_field_full239k_avgfrom512_v2"
MANIFESTS="$DATASET_ROOT/manifests"
PILOT="$DATASET_ROOT/pilot/shard_00000"
STAGING="$PILOT/staging"

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
