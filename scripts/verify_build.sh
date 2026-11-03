#!/bin/bash
#SBATCH --job-name=verify_agile_optim
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/verify_build_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/verify_build_%j.err
set -uxo pipefail

ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date)"
nvidia-smi

source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export CUDA_HOME="$ENVDIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.6

echo "=== nvcc path and version (from agile-optim env, not system) ==="
which nvcc
nvcc --version

echo "=== CUDA_HOME headers/libs ==="
ls "$CUDA_HOME/include/cuda_runtime.h" 2>&1
ls "$CUDA_HOME/lib64/libcudart.so"* 2>&1

echo "=== torch ==="
python -c "
import torch
print('torch.__version__ =', torch.__version__)
print('torch.version.cuda =', torch.version.cuda)
print('torch.cuda.is_available() =', torch.cuda.is_available())
print('torch.cuda.get_device_name(0) =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('CUDA matmul OK, result sum =', y.sum().item())
"

echo "=== pytorch3d (import + a real CUDA op: point cloud / mesh rasterization building block) ==="
python -c "
import pytorch3d
print('pytorch3d.__version__ =', pytorch3d.__version__)
from pytorch3d.ops import knn_points
import torch
pts = torch.randn(1, 100, 3, device='cuda')
out = knn_points(pts, pts, K=5)
torch.cuda.synchronize()
print('pytorch3d CUDA knn_points OK, dists shape =', out.dists.shape)
"

echo "=== torchcumesh2sdf (import test) ==="
python -c "
import torchcumesh2sdf
print('torchcumesh2sdf imported OK, module file:', torchcumesh2sdf.__file__)
print(dir(torchcumesh2sdf))
"

echo "=== manopth (import + construct a MANO layer against the transferred MANO_RIGHT.pkl) ==="
python -c "
from manopth.manolayer import ManoLayer
import torch, os
mano_dir = '/scratch/users/anirban/tamaghnam/HOI/assets/mano'
print('MANO_RIGHT.pkl exists:', os.path.isfile(os.path.join(mano_dir, 'MANO_RIGHT.pkl')))
try:
    layer = ManoLayer(mano_root=mano_dir, use_pca=False, flat_hand_mean=True, side='right')
    print('manopth ManoLayer constructed OK from the transferred MANO_RIGHT.pkl')
    pose = torch.zeros(1, 48)
    shape = torch.zeros(1, 10)
    verts, joints = layer(pose, shape)
    print('manopth forward pass OK, verts shape:', verts.shape, 'joints shape:', joints.shape)
except Exception as e:
    print('manopth ManoLayer construction/forward FAILED:', repr(e))
"

echo "=== pip check (re-confirm, now with everything installed) ==="
python -m pip check

echo "Verification finished $(date)"
