#!/bin/bash
#SBATCH --job-name=eval-point-density-voxels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=jjparkcv_owned2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs
module load gcc/11.2.0 cuda/12.8.1
eval "$(conda shell.bash hook)"
conda activate trellis2

python eval_point_density_voxel_queries.py "$@"
