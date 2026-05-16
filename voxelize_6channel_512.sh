#!/usr/bin/env bash
#SBATCH --job-name=voxelize_6channel_512
#SBATCH --output=./job_logs/voxelize_gdist_512_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=spgpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --account=jjparkcv_owned1
#SBATCH --gpus-per-task=1
#SBATCH --array=0-31%16

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2
mkdir -p job_logs

export ROOT=/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k

python data_toolkit/voxelize_gaussian_distance.py ObjaverseXL \
  --root "$ROOT" \
  --pbr_dump_root "$ROOT" \
  --gaussian_distance_voxel_root "$ROOT" \
  --resolution 512 \
  --sigma_multipliers 0.5,3.0,10.0 \
  --max_workers 1 \
  --rank "$SLURM_ARRAY_TASK_ID" \
  --world_size 32