#!/bin/bash
#SBATCH --job-name=v03_base
#SBATCH --output=log/v03_baselines.log
#SBATCH --error=log/v03_baselines.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=spcc-a40g04

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting Job A: Baselines SLURM Job on Mistral-7B-Instruct-v0.3..."
date

python -u scratch/run_v03_baselines.py

echo "Job A: Baselines finished successfully!"
date
