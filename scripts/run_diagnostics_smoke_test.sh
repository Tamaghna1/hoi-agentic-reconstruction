#!/bin/bash
#SBATCH --job-name=diag_smoke_test
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/diag_smoke_test_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/diag_smoke_test_%j.err
set -uo pipefail
ROOT=/scratch/users/anirban/tamaghnam/HOI
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

START_TIME=$(date +%s)
echo "=== Running diagnostics-logging smoke test (isolated from A/B/C/D and from the prior pipeline smoke test) ==="
python "$ROOT/scripts/run_diagnostics_smoke_test.py"
STATUS=$?
END_TIME=$(date +%s)
echo "=== Diagnostics smoke test exit status: $STATUS, elapsed: $((END_TIME - START_TIME))s ==="
echo "Done $(date)"
exit $STATUS
