#!/bin/bash
#SBATCH --job-name=analysis_selftest
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00
#SBATCH --output=/scratch/users/anirban/tamaghnam/HOI/logs/analysis_selftest_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/HOI/logs/analysis_selftest_%j.err
set -euo pipefail
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) start $(date)"
echo "Purpose: synthetic self-tests for scripts/analyze_temporal_consistency.py (stdlib-only, no AGILE/torch imports, independent of job 73170)"
/usr/bin/python3 /scratch/users/anirban/tamaghnam/HOI/scripts/analyze_temporal_consistency.py --self-test
echo "exit code: $?"
echo "Done $(date)"
