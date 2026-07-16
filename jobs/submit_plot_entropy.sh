#!/bin/bash
#SBATCH --job-name=plot_entropy
#SBATCH --output=log/plot_entropy.log
#SBATCH --error=log/plot_entropy.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

echo "Starting Entropy Plotting Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Execute the plotting script
python scratch/plot_entropy_and_gating.py

echo "Job finished successfully!"
date
