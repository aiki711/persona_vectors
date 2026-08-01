#!/bin/bash
#SBATCH --job-name=v03_prk_swp
#SBATCH --output=log/v03_proj_rank_sweeps.log
#SBATCH --error=log/v03_proj_rank_sweeps.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting proj_rank Rise & Fall Parameter Sweeps v0.3 SLURM Job..."
date

python -u scratch/run_v03_proj_rank_sweeps.py
python -u scratch/plot_v03_proj_rank_heatmaps.py

echo "proj_rank Parameter Sweeps Job finished successfully!"
date
