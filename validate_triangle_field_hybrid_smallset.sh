#!/bin/bash
#SBATCH --job-name=trifield-hybrid-qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2

ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k
WORK="$ROOT/validation/triangle_field_hybrid_smallset_20260818"
INSTANCES="$WORK/instances.txt"
mkdir -p "$WORK"

printf '%s\n' \
  001ac932aa199c005de2d2f0cbf39853727a793e5359aa0a37a20312334e9016 \
  00386b0fc13c7c6914ff2ce42e7a8133cbda6de1ff1114c67fe576d4de80168a \
  003b5ddc8fb5b2dc7b6408ff00b73507d09138375d1d9adea18e977efeb210aa \
  004a5078be6c6e53ff192512d0ad02fb06cad89e79ca3e053567163fbdaa7f83 \
  > "$INSTANCES"

python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$WORK" \
  --resolution 512 \
  --instances "$INSTANCES" \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 5 \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers 8

python data_toolkit/voxelize_triangle_field.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --triangle_field_voxel_root "$WORK" \
  --support_source_root "$WORK" \
  --support_source_resolution 512 \
  --resolution 32,64,128,256 \
  --instances "$INSTANCES" \
  --feature_dtype float16 \
  --npz_compression zstd \
  --zstd_level 5 \
  --candidate_source native \
  --projection_mode inside_barycentric \
  --max_workers 8

python data_toolkit/downsample_triangle_field_voxels.py \
  --source_root "$WORK/triangle_field_voxels_512" \
  --source_metadata "$ROOT/metadata.csv" \
  --output_root "$WORK" \
  --resolutions 32,64,128,256 \
  --instances "$INSTANCES" \
  --output_suffix hybrid_avg_from_512 \
  --feature_mode average_targets_recompute_aux \
  --recomputed_root "$WORK" \
  --feature_dtype float16 \
  --compression zstd \
  --zstd_level 5 \
  --device cpu \
  --max_workers 8

python data_toolkit/validate_hybrid_triangle_field_downsampling.py \
  --root "$WORK" \
  --instances "$INSTANCES" \
  --resolutions 32,64,128,256 \
  --hybrid_suffix hybrid_avg_from_512 \
  --output_json "$WORK/validation.json"

python visualize_triangle_field_avg_payloads.py \
  --root "$WORK" \
  --instances "$INSTANCES" \
  --num_samples 4 \
  --resolutions 32,64,128,256 \
  --avg_suffix hybrid_avg_from_512 \
  --output_dir "$WORK/visualizations" \
  --render_resolution 320 \
  --ssaa 4
