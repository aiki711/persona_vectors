#!/bin/bash
#SBATCH --job-name=anticipatory_comp
#SBATCH --output=log/resampling_vs_delayed.log
#SBATCH --error=log/resampling_vs_delayed.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=spcc-a40g04

set -e

mkdir -p log

source persona_steering/bin/activate

echo "Starting 1-Token Delayed vs. Anticipatory Re-sampling Comparison SLURM Job..."
date

python -u scratch/run_anticipatory_gating_comparison.py

echo "Generating comparison plots and report..."
python -u scratch/plot_anticipatory_comparison.py

echo "Anticipatory Comparison Job finished successfully!"
date
