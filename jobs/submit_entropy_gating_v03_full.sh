#!/bin/bash
#SBATCH --job-name=entropy_full
#SBATCH --output=log/entropy_gating_v03_full.log
#SBATCH --error=log/entropy_gating_v03_full.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=spcc-a40g04

set -e

mkdir -p log

source persona_steering/bin/activate

echo "Starting Mistral-7B-Instruct-v0.3 Full Dual Sweep SLURM Job..."
date

python -u scratch/run_entropy_gating_v03_full_sweep.py

echo "Mistral-7B-v0.3 Full Dual Sweep Job finished successfully!"
date
