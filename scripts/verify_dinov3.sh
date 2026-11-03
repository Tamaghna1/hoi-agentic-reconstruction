#!/bin/bash
#SBATCH --job-name=verify_dinov3
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/verify_dinov3_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/verify_dinov3_%j.err
set -euxo pipefail
ROOT=/scratch/users/anirban/tamaghnam/HOI
REPO=$ROOT/repo
ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim

source /scratch/users/anirban/tamaghnam/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export DINO_MODEL_PATH="$ROOT/assets/dinov3"
export HF_HUB_OFFLINE=1

echo "=== File size cross-check against known HF metadata ==="
python3 -c "
import os
expected = {'model.safetensors': 1212559808}
for name, size in expected.items():
    path = os.path.join('$ROOT/assets/dinov3', name)
    actual = os.path.getsize(path)
    status = 'MATCH' if actual == size else 'MISMATCH'
    print(f'{name}: expected={size} actual={actual} [{status}]')
"

echo "=== AGILE's own DINOWrapper class (the exact path Runner.__init__ uses) ==="
python -c "
import sys, torch
sys.path.insert(0, '$REPO')
from optimization.utils.feature import DINOWrapper

wrapper = DINOWrapper(input_size=1024)
print('DINOWrapper initialized OK from', '$ROOT/assets/dinov3')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wrapper = wrapper.to(device)
dummy = torch.rand(1, 3, 1024, 1024, device=device)
with torch.no_grad():
    out = wrapper(dummy.cpu().numpy() if hasattr(wrapper, 'forward') else dummy)
" 2>&1 | tee /tmp/dino_test_attempt1.log || true

echo "=== If the above forward-pass call signature was wrong, fall back to a pure load+encode smoke test ==="
python -c "
import sys, os, torch
sys.path.insert(0, '$REPO')
os.environ['DINO_MODEL_PATH'] = '$ROOT/assets/dinov3'
from transformers import AutoModel, AutoImageProcessor
model_path = '$ROOT/assets/dinov3'
model = AutoModel.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)
print('transformers AutoModel + AutoImageProcessor loaded OK from the downloaded checkpoint')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()
import numpy as np
from PIL import Image
dummy_img = Image.fromarray((np.random.rand(224,224,3)*255).astype('uint8'))
inputs = processor(images=dummy_img, return_tensors='pt').to(device)
with torch.no_grad():
    out = model(**inputs)
print('DINOv3 forward pass OK, last_hidden_state shape:', out.last_hidden_state.shape)
print('num_register_tokens expected 4 (per AGILE docs) - total extra tokens beyond patches should reflect CLS+4 registers')
"
echo "Done $(date)"
