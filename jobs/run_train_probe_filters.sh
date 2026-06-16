#!/bin/bash
#SBATCH --job-name=train_probe_filters
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:15:00
#SBATCH --output=log/train_probe_filters.out
#SBATCH --error=log/train_probe_filters.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Training probe filters..."
"$PYTHON_BIN" scripts/01_vectors/36_train_probe_filters.py --k 500
echo "Done."
