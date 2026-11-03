#!/bin/bash
#SBATCH --job-name=prepare_mano
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/prepare_mano_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/prepare_mano_%j.err
set -euxo pipefail
ROOT=/scratch/users/anirban/tamaghnam/HOI
REPO=$ROOT/repo
ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim

source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

cd "$REPO"
echo "=== Running prepare_mano_assets.py ==="
python tools/prepare_mano_assets.py --model-dir "$ROOT/assets/mano" --finger-mask configs/finger_vertices.json

echo "=== Resulting assets/mano contents ==="
ls -la "$ROOT/assets/mano"

echo "=== Test AGILE's own MANO class (the exact path Runner.__init__ uses, not manopth directly) ==="
python -c "
import yaml, torch
from pathlib import Path
import sys
sys.path.insert(0, '$REPO')
from optimization.models.mano_wrapper import MANO

mano_model_path = '$ROOT/assets/mano'
mano_config_path = Path(mano_model_path) / 'mano_config.yaml'
assert mano_config_path.exists(), f'MANO config not found at {mano_config_path}'
with open(mano_config_path, 'r') as f:
    mano_cfg = yaml.safe_load(f)
mano_cfg = {k.lower(): v for k, v in mano_cfg['MANO'].items()}
model = MANO(**mano_cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
print('AGILE MANO class initialized OK from generated mano_config.yaml')
print('MANO model type:', type(model))
"
echo "Done $(date)"
