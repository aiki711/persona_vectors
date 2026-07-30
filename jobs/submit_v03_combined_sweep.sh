#!/bin/bash
#SBATCH --job-name=v03_comb
#SBATCH --output=log/v03_combined_sweep.log
#SBATCH --error=log/v03_combined_sweep.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting Job D: Combined Rise-Fall Dynamic Gating Job on Mistral-7B-Instruct-v0.3..."
date

python -u scratch/run_v03_combined_sweep.py

echo "Job D: Combined Rise-Fall Dynamic Gating Job finished successfully!"
date
