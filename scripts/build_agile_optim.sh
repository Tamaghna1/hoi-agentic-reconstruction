#!/bin/bash
#SBATCH --job-name=build_agile_optim
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/build_agile_optim_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/build_agile_optim_%j.err
set -euxo pipefail

ROOT=/scratch/users/anirban/tamaghnam/HOI
REPO=$ROOT/repo
ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date)"

source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh

if [ ! -d "$ENVDIR" ]; then
  conda create -y -p "$ENVDIR" --override-channels -c https://conda.anaconda.org/conda-forge python=3.10 pip
fi
conda activate "$ENVDIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

echo "=== Stage 1: torch (cu118) ==="
python -m pip install --no-user torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

echo "=== Stage 2: pinned build tools ==="
python -m pip install --no-user pip==25.2 setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 numpy==1.23.5 scipy==1.15.3 six==1.17.0

echo "=== Stage 3: chumpy (no build isolation) ==="
python -m pip install --no-user --no-build-isolation chumpy==0.70

echo "=== Stage 4: requirements.txt ==="
cd "$REPO"
python -m pip install --no-user -r requirements.txt

echo "=== Stage 5: CUDA 11.8 toolkit via conda (system has 12.4, not 11.8) ==="
conda install -y -p "$ENVDIR" -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 --override-channels

export CUDA_HOME="$ENVDIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUB_HOME="$CUDA_HOME/include"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
nvcc --version

echo "=== Stage 6: external CUDA extensions (built without a live GPU, FORCE_CUDA=1) ==="
mkdir -p external
if [ ! -d external/pytorch3d ]; then
  git clone https://github.com/facebookresearch/pytorch3d.git external/pytorch3d
  git -C external/pytorch3d checkout 5043d15361d16a7093b4b60572c5f730c6c83308
fi
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=1 python -m pip install --no-user --no-build-isolation ./external/pytorch3d

if [ ! -d external/cumesh2sdf ]; then
  git clone https://github.com/eliphatfs/cumesh2sdf.git external/cumesh2sdf
  git -C external/cumesh2sdf checkout 7789ade81bcd5fa77112d2cb5d825ed0e634e76b
fi
CUDA_VISIBLE_DEVICES="" python -m pip install --no-user --no-build-isolation ./external/cumesh2sdf

if [ ! -d external/manopth ]; then
  git clone https://github.com/hassony2/manopth.git external/manopth
  git -C external/manopth checkout 4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b
fi
python -m pip install --no-user --no-build-isolation ./external/manopth

echo "=== Stage 7: pip check ==="
python -m pip check

echo "=== Stage 8: static release check (no GPU/model deps) ==="
python tools/check_release.py || echo "check_release.py reported issues (see above) — non-fatal for env build"

echo "Build finished $(date)"
