#!/bin/bash
#SBATCH --job-name=high_pass_gating
#SBATCH --output=log/high_pass_gating.log
#SBATCH --error=log/high_pass_gating.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting High-Pass Gating SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Execute the experiment script
python scratch/run_high_pass_gating.py

echo "Job finished successfully!"
date
