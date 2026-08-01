#!/bin/bash
#SBATCH --job-name=v03_dual_dyn
#SBATCH --output=log/v03_dual_dynamic_steering.log
#SBATCH --error=log/v03_dual_dynamic_steering.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting Dual Dynamic Steering (proj_rank x Token Intensity Gating) v0.3 SLURM Job..."
date

python -u scratch/run_v03_dual_dynamic_steering.py

echo "Dual Dynamic Steering v0.3 Job finished successfully!"
date
