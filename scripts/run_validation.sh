#!/bin/bash
#SBATCH --job-name=validate_mc1_config
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/validate_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/validate_%j.err
set -euo pipefail
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date)"
echo "Purpose: MC1 dataset integrity + A/B/C config-diff validation (stdlib-only, no conda env, independent of job 73170)"
/usr/bin/python3 --version
/usr/bin/python3 /scratch/users/anirban/tamaghnam/HOI/scripts/validate_data_and_config.py
status=$?
echo "Done $(date), exit status $status"
exit $status
