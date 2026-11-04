#!/bin/bash
#SBATCH --job-name=exp_D
#SBATCH --partition=short,long,a100,medium,ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/exp_D_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/exp_D_%j.err
set -uo pipefail
ROOT=/scratch/users/anirban/tamaghnam/HOI
ENVDIR=/scratch/users/anirban/tamaghnam/envs/agile-optim

echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date) - condition D"
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
mkdir -p "$ROOT/logs/experiments/D_control_iters1600"
export AGILE_DIAGNOSTIC_LOG="$ROOT/logs/experiments/D_control_iters1600/diagnostics.jsonl"

echo "=== W&B logging (nonfatal) ==="
python -c "
try:
    import wandb
    wandb.init(project='hoi-agile-mc1', name='condition_D_${SLURM_JOB_ID:-unknown}',
               job_type='experiment', dir='$WANDB_DIR',
               config={'condition': 'D', 'scene': 'MC1', 'start_frame': 10, 'end_frame': 200,
                       'object_max_iter_per_frame': 1600})
    wandb.log({'experiment_started': 1})
    wandb.finish()
    print('W&B logging OK')
except Exception as e:
    print('W&B logging failed (nonfatal, continuing):', repr(e))
"

START_TIME=$(date +%s)
echo "=== Running condition D (released config + object_max_iter_per_frame=1600, approved window 10-200) ==="
echo "=== AGILE_DIAGNOSTIC_LOG=$AGILE_DIAGNOSTIC_LOG ==="
python "$ROOT/scripts/run_experiment.py" --condition D
STATUS=$?
END_TIME=$(date +%s)
echo "=== Condition D exit status: $STATUS, elapsed: $((END_TIME - START_TIME))s ==="
echo "Done $(date)"
exit $STATUS
