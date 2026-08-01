#!/bin/bash
#SBATCH --job-name=v03_lselect
#SBATCH --output=log/v03_layer_selection_3methods.log
#SBATCH --error=log/v03_layer_selection_3methods.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting 3 Dynamic Layer Selection Methods (alpha=5.0) v0.3 SLURM Job..."
date

python -u scratch/run_v03_layer_selection_3methods.py

echo "3 Dynamic Layer Selection Methods Job finished successfully!"
date
