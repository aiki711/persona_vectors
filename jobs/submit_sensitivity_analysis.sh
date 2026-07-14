#!/bin/bash
#SBATCH --job-name=gating_sensitivity
#SBATCH --output=log/gating_sensitivity.log
#SBATCH --error=log/gating_sensitivity.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Gating Sensitivity Analysis SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Step 1: Run dual-forward pass token sensitivity data collection
python scripts/04_dyn_layer/02_token_intensity/run_sensitivity_analysis.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --num_prompts 10 \
    --alpha_max 5.0

# Step 2: Fit trends and plot optimal gating efficiency curves
python scripts/04_dyn_layer/02_token_intensity/plot_sensitivity_analysis.py

echo "Job finished successfully!"
date
