#!/bin/bash
#SBATCH --job-name=entropy_gating_phase1_rem
#SBATCH --output=log/entropy_gating_phase1_rem.log
#SBATCH --error=log/entropy_gating_phase1_rem.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Remaining Rise-Stage Entropy Gating Sweep SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Execute the sweep script
python scratch/run_entropy_gating_phase1_rem.py

echo "Job finished successfully!"
date
