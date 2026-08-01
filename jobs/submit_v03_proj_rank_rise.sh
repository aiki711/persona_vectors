#!/bin/bash
#SBATCH --job-name=v03_prk_rise
#SBATCH --output=log/v03_proj_rank_rise.log
#SBATCH --error=log/v03_proj_rank_rise.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting proj_rank Rise Parameter Sweep (25 pairs) SLURM Job..."
date

python -u scratch/run_v03_proj_rank_rise_sweep.py

echo "proj_rank Rise Parameter Sweep Job finished successfully!"
date
