#!/bin/bash
#SBATCH --job-name=v03_prk_fall
#SBATCH --output=log/v03_proj_rank_fall.log
#SBATCH --error=log/v03_proj_rank_fall.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting proj_rank Fall Parameter Sweep (25 pairs) SLURM Job..."
date

python -u scratch/run_v03_proj_rank_fall_sweep.py

echo "proj_rank Fall Parameter Sweep Job finished successfully!"
date
