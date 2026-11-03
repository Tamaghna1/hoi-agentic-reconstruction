#!/bin/bash
#SBATCH --job-name=verify_cumesh2sdf
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/verify_cumesh2sdf_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/verify_cumesh2sdf_%j.err
set -uxo pipefail
ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim
source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

echo "=== Test 1: import torch first, then torchcumesh2sdf ==="
python -c "
import torch
print('torch imported OK')
import torchcumesh2sdf
print('torchcumesh2sdf imported OK, file:', torchcumesh2sdf.__file__)
print([a for a in dir(torchcumesh2sdf) if not a.startswith('_')])
"

echo "=== Test 2: find libc10.so actual location, and check extension's own needed libs via ldd ==="
find "$ENVDIR/lib/python3.10/site-packages/torch" -iname "libc10.so" 2>&1
EXT_SO=$(find "$ENVDIR/lib/python3.10/site-packages" -iname "*cumesh2sdf*.so" 2>/dev/null | head -1)
echo "extension .so: $EXT_SO"
ldd "$EXT_SO" 2>&1 | grep -i "not found\|libc10\|libtorch"

echo "Done $(date)"
