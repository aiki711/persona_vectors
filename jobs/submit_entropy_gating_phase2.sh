#!/bin/bash
#SBATCH --job-name=entropy_phase2
#SBATCH --output=log/entropy_gating_phase2.log
#SBATCH --error=log/entropy_gating_phase2.err
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

echo "Starting Fall-Stage Entropy Gating Sweep (Phase 2) SLURM Job..."
date

python -u scratch/run_entropy_gating_phase2.py

echo "Phase 2 SLURM Job finished successfully!"
date
