#!/bin/bash
#SBATCH --job-name=visualize-sparse-gaussian-patch-dataset
#SBATCH --output=./job_logs/visualize-sparse-gaussian-patch-dataset_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu-rtx6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --account=jjparkcv_owned2
#SBATCH --gres=gpu:1

source ~/.bashrc
module load cuda/12.8
module load gcc/11
conda activate /home/gpranav/pranav_work/scratch/envs/trellis2

cd /home/gpranav/pranav_work/scratch/TRELLIS.2/

export ROOT="/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
export DATA_DIR="{\"train\":{\"base\":\"$ROOT/splits/train\",\"gaussian_distance_voxel\":\"$ROOT/gaussian_distance_voxels_64\",\"_metadata_filter_csv\":\"$ROOT/splits/train/metadata.csv,$ROOT/metadata_no_local_dense.csv\"}}"

python visualize_sparse_gaussian_patch_dataset.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$ROOT/outputs/gt_edge_vertex_samples_64" \
  --resolution 64 \
  --patch_size 64 \
  --num_samples 16 \
  --seed 0 \
  --zero_cond \
  --foreground_patch_prob 1.0 \
  --max_resample_attempts 1 \
  --render_resolution 512 \
  --ssaa 2