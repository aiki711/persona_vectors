#!/bin/bash
#SBATCH --job-name=prep_vvec
#SBATCH --output=log/prepare_vocab_projected_vectors.log
#SBATCH --error=log/prepare_vocab_projected_vectors.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting Preparation and Verification of Vocab-Projected Vectors..."
date

python -u scratch/prepare_vocab_projected_vectors.py
python -u scratch/verify_vocab_projected_vectors.py

echo "Vocab-Projected Vectors Preparation and Verification Finished Successfully!"
date
