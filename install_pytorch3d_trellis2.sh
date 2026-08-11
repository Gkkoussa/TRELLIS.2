#!/bin/bash
#SBATCH --job-name=install-pytorch3d-trellis2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1

set -euo pipefail

cd /home/koussa/scratch/TRELLIS.2
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate trellis2
module load gcc/11.2.0 cuda/12.8.1

export FORCE_CUDA=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export TORCH_CUDA_ARCH_LIST="8.6+PTX"

pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
python -c "from pytorch3d.ops import sample_farthest_points; print('PyTorch3D FPS ready')"
