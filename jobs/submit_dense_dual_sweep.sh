#!/bin/bash
#SBATCH --job-name=dense_dual_sweep
#SBATCH --output=log/dense_dual_sweep.log
#SBATCH --error=log/dense_dual_sweep.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Dense Dual Gating Sweep SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Execute the sweep script
python scratch/run_dense_dual_sweep.py

echo "Job finished successfully!"
date
