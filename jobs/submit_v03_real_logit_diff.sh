#!/bin/bash
#SBATCH --job-name=v03_real_logit
#SBATCH --output=log/v03_real_logit_diff.log
#SBATCH --error=log/v03_real_logit_diff.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting Real Logit-Diff (score_mode=logit_diff, alpha=5.0) v0.3 SLURM Job..."
date

python -u scratch/run_v03_real_logit_diff.py

echo "Real Logit-Diff v0.3 Job finished successfully!"
date
