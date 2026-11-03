#!/bin/bash
#SBATCH --job-name=download_dinov3
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/download_dinov3_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/download_dinov3_%j.err
set -euxo pipefail
ROOT=/scratch/users/anirban/tamaghnam/HOI
ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim

source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date)"
echo "=== Downloading DINOv3 checkpoint (resumable via hf download --local-dir) ==="
hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir "$ROOT/assets/dinov3"

echo "=== Downloaded files ==="
ls -la "$ROOT/assets/dinov3"
du -sh "$ROOT/assets/dinov3"

echo "Done $(date)"
