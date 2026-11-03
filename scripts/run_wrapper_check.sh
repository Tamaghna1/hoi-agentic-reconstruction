#!/bin/bash
#SBATCH --job-name=wrapper_check_only
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/wrapper_check_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/wrapper_check_%j.err
set -uo pipefail
export DINO_MODEL_PATH=/scratch/users/anirban/tamaghnam/HOI/assets/dinov3
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date)"
echo "Purpose: exercise run_experiment.py --check-only for A/B/C (config+prerequisite validation only, no AGILE/training imports reached)"
for cond in A B C; do
  echo "--- condition $cond ---"
  /usr/bin/python3 /scratch/users/anirban/tamaghnam/HOI/scripts/run_experiment.py --condition "$cond" --check-only
  echo "exit code: $?"
done
echo "Done $(date)"
