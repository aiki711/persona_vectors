#!/bin/bash
#SBATCH --job-name=coordinate_step1
#SBATCH --output=log/coordinate_step1.log
#SBATCH --error=log/coordinate_step1.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Coordinate Descent Step 1 SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Execute the step 1 script
python scratch/run_coordinate_step1.py

echo "Job finished successfully!"
date
