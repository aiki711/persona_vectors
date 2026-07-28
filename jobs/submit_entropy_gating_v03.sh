#!/bin/bash
#SBATCH --job-name=entropy_v03
#SBATCH --output=log/entropy_gating_v03.log
#SBATCH --error=log/entropy_gating_v03.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=spcc-a40g04

set -e

mkdir -p log

source persona_steering/bin/activate

echo "Starting Mistral-7B-Instruct-v0.3 Entropy Gating Sweep SLURM Job..."
date

python -u scratch/run_entropy_gating_v03_sweep.py

echo "Mistral-7B-v0.3 Entropy Gating Sweep Job finished successfully!"
date
