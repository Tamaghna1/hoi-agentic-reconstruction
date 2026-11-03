#!/bin/bash
#SBATCH --job-name=check_cuda
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/check_cuda_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/check_cuda_%j.err
set -x
echo "Host: $(hostname)"
nvidia-smi
echo "--- Searching for any CUDA toolkit installs ---"
ls -d /usr/local/cuda* 2>&1
which nvcc 2>&1
nvcc --version 2>&1
echo "--- Searching broader common install paths ---"
find /usr/local -maxdepth 1 -iname "cuda*" 2>/dev/null
find /opt -maxdepth 2 -iname "*cuda*" 2>/dev/null
echo "--- GCC / build tools ---"
gcc --version 2>&1 | head -1
g++ --version 2>&1 | head -1
