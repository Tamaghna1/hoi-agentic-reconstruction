#!/bin/bash
#SBATCH --job-name=check_hf_auth
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/check_hf_auth_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/check_hf_auth_%j.err
set -uxo pipefail
ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim
source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

echo "=== Checking HF authentication (never prints the token itself) ==="
python -c "
from huggingface_hub import whoami, HfApi
try:
    info = whoami()
    print('AUTHENTICATED as:', info.get('name', '<unknown>'))
except Exception as e:
    print('NOT AUTHENTICATED:', repr(e))
    raise SystemExit(0)

api = HfApi()
try:
    model_info = api.model_info('facebook/dinov3-vitl16-pretrain-lvd1689m')
    print('GATED REPO ACCESS CONFIRMED - model_info retrieved OK')
    print('siblings (files):', [s.rfilename for s in model_info.siblings])
except Exception as e:
    print('GATED REPO ACCESS FAILED:', repr(e))
"
echo "Done $(date)"
