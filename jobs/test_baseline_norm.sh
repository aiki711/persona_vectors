#!/bin/bash
#SBATCH --job-name=test_baseline_norm
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=01:00:00
#SBATCH --output=log/test_baseline_norm.out
#SBATCH --error=log/test_baseline_norm.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"

persona_steering/bin/python3 scratch/test_baseline_norm.py
