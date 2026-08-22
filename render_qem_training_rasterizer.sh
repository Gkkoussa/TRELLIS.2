#!/bin/bash
#SBATCH --job-name=qem-train-raster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=triangle_job_logs/%x-%j.log
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2

set -euo pipefail

REPO=/gpfs/accounts/jjparkcv_root/jjparkcv0/gpranav/TRELLIS.2
cd "$REPO"
mkdir -p triangle_job_logs

eval "$(conda shell.bash hook)"
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

module load cuda/12.8
module load gcc/11

QEM_INPUT_DIR="${QEM_INPUT_DIR:-/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k/outputs/qem_edge_collapse_selected_405b4580844ee249c7dbe0507e5bd83c7cc516b3a9ae210f3b64e4ccfa7b09d6}"

python data_toolkit/render_qem_training_rasterizer.py \
  --input-dir "$QEM_INPUT_DIR" \
  --resolutions 32 64 128 256 512 \
  --image-resolution 512 \
  --ssaa 4
