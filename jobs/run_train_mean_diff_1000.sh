#!/bin/bash
#SBATCH --job-name=train_mean_diff_1000
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=01:00:00
#SBATCH --output=log/train_mean_diff_1000.out
#SBATCH --error=log/train_mean_diff_1000.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Extracting 1000-sample hidden states for extraversion..."
"$PYTHON_BIN" scripts/01_vectors/30b_train_mean_diff.py \
    -c config/mistral_7b.yaml \
    --out_dir vectors \
    --axis extraversion

echo "Done."
