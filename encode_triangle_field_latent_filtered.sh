#!/bin/bash
#SBATCH --job-name=trifield-encode-georgeframework
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p triangle_job_logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

module load cuda/12.8
module load gcc/11

export ROOT="${ROOT:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k}"

export TRIANGLE_FIELD_VAE_RUN="${1:-${TRIANGLE_FIELD_VAE_RUN:-triangle_field_vae_512_invarea_auxdrop_52039231}}"
export TRIANGLE_FIELD_VAE_CKPT="${2:-${TRIANGLE_FIELD_VAE_CKPT:-step0180000}}"

if [ -z "$TRIANGLE_FIELD_VAE_RUN" ] || [ -z "$TRIANGLE_FIELD_VAE_CKPT" ]; then
  echo "Usage: sbatch encode_triangle_field_latent_filtered.sh <triangle_field_vae_run_name> <checkpoint_step>"
  echo "Example: sbatch encode_triangle_field_latent_filtered.sh triangle_field_vae_512_invarea_auxdrop_52039231 step0180000"
  exit 1
fi

export SPLIT="${SPLIT:-train}"
export TRIANGLE_FIELD_RESOLUTION="${TRIANGLE_FIELD_RESOLUTION:-512}"
export TRIANGLE_FIELD_MAX_ACTIVE_VOXELS="${TRIANGLE_FIELD_MAX_ACTIVE_VOXELS:-1000000}"
export TRIANGLE_FIELD_LATENT_NAME="${TRIANGLE_FIELD_VAE_RUN}_${TRIANGLE_FIELD_VAE_CKPT}_${TRIANGLE_FIELD_RESOLUTION}"
export NO_TRAIN_DUPLICATE_CSV="${NO_TRAIN_DUPLICATE_CSV:-/home/gpranav/pranav_work/scratch/TRELLIS.2/metadata_test_no_train_duplicates.csv}"
export APPLY_NO_TRAIN_DUPLICATE_CSV="${APPLY_NO_TRAIN_DUPLICATE_CSV:-auto}"
export TRIANGLE_METADATA_FILTER_CSV="${TRIANGLE_METADATA_FILTER_CSV:-}"
export TRIANGLE_LOADER_WORKERS="${TRIANGLE_LOADER_WORKERS:-4}"
export TRIANGLE_SAVER_WORKERS="${TRIANGLE_SAVER_WORKERS:-4}"
export TRIANGLE_FIELD_VOXEL_METADATA="$ROOT/triangle_field_voxels_${TRIANGLE_FIELD_RESOLUTION}/metadata.csv"
export SPLIT_INSTANCES_PATH="$ROOT/splits/$SPLIT/instances.txt"
export GENERATED_INSTANCES_DIR="$ROOT/triangle_field_latents/encode_instance_lists"
export GENERATED_INSTANCES_PATH="$GENERATED_INSTANCES_DIR/${SPLIT}_triangle_field_${TRIANGLE_FIELD_RESOLUTION}_le${TRIANGLE_FIELD_MAX_ACTIVE_VOXELS}.txt"

echo "ROOT=$ROOT"
echo "SPLIT=$SPLIT"
echo "TRIANGLE_FIELD_VAE_RUN=$TRIANGLE_FIELD_VAE_RUN"
echo "TRIANGLE_FIELD_VAE_CKPT=$TRIANGLE_FIELD_VAE_CKPT"
echo "TRIANGLE_FIELD_RESOLUTION=$TRIANGLE_FIELD_RESOLUTION"
echo "TRIANGLE_FIELD_MAX_ACTIVE_VOXELS=$TRIANGLE_FIELD_MAX_ACTIVE_VOXELS"
echo "TRIANGLE_FIELD_LATENT_NAME=$TRIANGLE_FIELD_LATENT_NAME"
echo "SPLIT_INSTANCES_PATH=$SPLIT_INSTANCES_PATH"
echo "TRIANGLE_FIELD_VOXEL_METADATA=$TRIANGLE_FIELD_VOXEL_METADATA"

if [ ! -f "$SPLIT_INSTANCES_PATH" ]; then
  echo "Missing split instances: $SPLIT_INSTANCES_PATH" >&2
  exit 1
fi

if [ ! -f "$TRIANGLE_FIELD_VOXEL_METADATA" ]; then
  echo "Missing triangle-field voxel metadata: $TRIANGLE_FIELD_VOXEL_METADATA" >&2
  exit 1
fi

mkdir -p "$GENERATED_INSTANCES_DIR"
python - <<'PY'
import csv
import os
from pathlib import Path

split_instances_path = Path(os.environ["SPLIT_INSTANCES_PATH"])
voxel_metadata_path = Path(os.environ["TRIANGLE_FIELD_VOXEL_METADATA"])
out_path = Path(os.environ["GENERATED_INSTANCES_PATH"])
cap = int(os.environ["TRIANGLE_FIELD_MAX_ACTIVE_VOXELS"])

split_instances = [
    line.strip()
    for line in split_instances_path.read_text().splitlines()
    if line.strip()
]
valid = set()
with voxel_metadata_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            voxels = int(float(row.get("num_triangle_field_voxels") or 0))
        except ValueError:
            voxels = 0
        voxelized = str(row.get("triangle_field_voxelized")).strip().lower() in {
            "1",
            "true",
            "t",
            "yes",
            "y",
        }
        if voxelized and 0 < voxels <= cap:
            valid.add(row["sha256"])

kept = [sha256 for sha256 in split_instances if sha256 in valid]
out_path.write_text("\n".join(kept) + ("\n" if kept else ""))
print(
    f"Generated capped instances: split={len(split_instances)} "
    f"valid_payload_le_cap={len(valid)} kept={len(kept)} path={out_path}",
    flush=True,
)
PY

case "$APPLY_NO_TRAIN_DUPLICATE_CSV" in
  auto)
    if [ "$SPLIT" = "test" ]; then
      TRIANGLE_METADATA_FILTER_CSV="${TRIANGLE_METADATA_FILTER_CSV:-$NO_TRAIN_DUPLICATE_CSV}"
    fi
    ;;
  1|true|TRUE|yes|YES)
    TRIANGLE_METADATA_FILTER_CSV="${TRIANGLE_METADATA_FILTER_CSV:-$NO_TRAIN_DUPLICATE_CSV}"
    ;;
  0|false|FALSE|no|NO)
    TRIANGLE_METADATA_FILTER_CSV=""
    ;;
  *)
    echo "Invalid APPLY_NO_TRAIN_DUPLICATE_CSV=$APPLY_NO_TRAIN_DUPLICATE_CSV" >&2
    exit 1
    ;;
esac

if [ -n "$TRIANGLE_METADATA_FILTER_CSV" ]; then
  echo "TRIANGLE_METADATA_FILTER_CSV=$TRIANGLE_METADATA_FILTER_CSV"
else
  echo "TRIANGLE_METADATA_FILTER_CSV=<none>"
fi

CMD=(
  python data_toolkit/encode_triangle_field_latent.py
  --root "$ROOT"
  --triangle_field_voxel_root "$ROOT"
  --triangle_field_latent_root "$ROOT"
  --resolution "$TRIANGLE_FIELD_RESOLUTION"
  --model_root "$ROOT/outputs"
  --enc_model "$TRIANGLE_FIELD_VAE_RUN"
  --ckpt "$TRIANGLE_FIELD_VAE_CKPT"
  --instances "$GENERATED_INSTANCES_PATH"
  --loader_workers "$TRIANGLE_LOADER_WORKERS"
  --saver_workers "$TRIANGLE_SAVER_WORKERS"
)

if [ -n "$TRIANGLE_METADATA_FILTER_CSV" ]; then
  CMD+=(--metadata_filter_csv "$TRIANGLE_METADATA_FILTER_CSV")
fi

"${CMD[@]}"

export TRIANGLE_FIELD_LATENT_ROOT="$ROOT/triangle_field_latents/$TRIANGLE_FIELD_LATENT_NAME"
python - <<'PY'
import glob
import os
from pathlib import Path

import pandas as pd

latent_root = Path(os.environ["TRIANGLE_FIELD_LATENT_ROOT"])
metadata_path = latent_root / "metadata.csv"
frames = []

if metadata_path.exists():
    frames.append(pd.read_csv(metadata_path))

for path in sorted(glob.glob(str(latent_root / "new_records" / "part_*.csv"))):
    if os.path.getsize(path) > 0:
        frames.append(pd.read_csv(path))

if not frames:
    raise SystemExit(f"No latent metadata records found under {latent_root}")

metadata = pd.concat(frames, ignore_index=True)
metadata = metadata.drop_duplicates("sha256", keep="last").sort_values("sha256")
metadata.to_csv(metadata_path, index=False)
print(f"Wrote latent metadata: {metadata_path} rows={len(metadata)}", flush=True)
PY
