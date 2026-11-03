#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/smoke_test_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/smoke_test_%j.err
set -uo pipefail
ROOT=/scratch/users/anirban/tamaghnam/HOI
REPO=$ROOT/repo
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
export DINO_MODEL_PATH="$ROOT/assets/dinov3"
export HF_HUB_OFFLINE=1
export WANDB_DIR="$ROOT/logs/wandb"
mkdir -p "$WANDB_DIR"

echo "=== W&B logging (nonfatal) ==="
python -c "
import os
try:
    import wandb
    run = wandb.init(project='hoi-agile-mc1', name='smoke_test_${SLURM_JOB_ID:-unknown}',
                      job_type='smoke_test', dir=os.environ.get('WANDB_DIR'),
                      config={'scene': 'MC1', 'start_frame': 10, 'end_frame': 15,
                              'purpose': 'integrated pipeline smoke test, not A/B/C'})
    wandb.log({'smoke_test_started': 1})
    wandb.finish()
    print('W&B logging OK')
except Exception as e:
    print('W&B logging failed (nonfatal, continuing):', repr(e))
"

START_TIME=$(date +%s)
echo "=== Running integrated smoke test ==="
python "$ROOT/scripts/run_smoke_test.py"
STATUS=$?
END_TIME=$(date +%s)
echo "=== Smoke test exit status: $STATUS, elapsed: $((END_TIME - START_TIME))s ==="

echo "=== pip check (final sanity) ==="
python -m pip check

echo "Done $(date)"
exit $STATUS
