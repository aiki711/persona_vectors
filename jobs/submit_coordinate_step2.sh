#!/bin/bash
#SBATCH --job-name=coordinate_step2
#SBATCH --output=log/coordinate_step2.log
#SBATCH --error=log/coordinate_step2.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Coordinate Descent Step 2 SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

# Execute the step 2 script
python scratch/run_coordinate_step2.py

echo "Job finished successfully!"
date
