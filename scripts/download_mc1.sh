#!/bin/bash
#SBATCH --job-name=mc1_download
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/mc1_download_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/mc1_download_%j.err
set -euo pipefail

ROOT=/scratch/users/anirban/tamaghnam/HOI
REPO=$ROOT/repo
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date)"

source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh

if [ ! -d /scratch/users/anirban/tamaghnam/envs/agile-download ]; then
  conda create -y -p /scratch/users/anirban/tamaghnam/envs/agile-download --override-channels -c https://conda.anaconda.org/conda-forge python=3.10 pip
fi
conda activate /scratch/users/anirban/tamaghnam/envs/agile-download
python -m pip install --no-user huggingface-hub==0.36.2

mkdir -p "$ROOT/data"
cd "$REPO"
echo "Starting download: MC1, kind=preprocessed"
python tools/download_ho3d.py --scenes MC1 --kind preprocessed --output "$ROOT/data"

echo "Done $(date)"
du -sh "$ROOT/data"
find "$ROOT/data" -maxdepth 4
