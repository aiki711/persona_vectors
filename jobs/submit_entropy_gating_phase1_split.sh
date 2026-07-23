#!/bin/bash
#SBATCH --job-name=entropy_gating_phase1_split
#SBATCH --output=log/entropy_gating_phase1_split.log
#SBATCH --error=log/entropy_gating_phase1_split.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Split Rise-Stage Entropy Gating Sweep SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Execute the split sweep script
python scratch/run_entropy_gating_phase1_split.py

echo "Job finished successfully!"
date
