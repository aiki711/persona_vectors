#!/bin/bash
#SBATCH --job-name=train_mean_diff_1000_others
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=02:00:00
#SBATCH --output=log/train_mean_diff_1000_others.out
#SBATCH --error=log/train_mean_diff_1000_others.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

for AXIS in neuroticism openness conscientiousness agreeableness; do
    echo "Extracting 1000-sample hidden states for ${AXIS}..."
    "$PYTHON_BIN" scripts/01_vectors/30b_train_mean_diff.py \
        -c config/mistral_7b.yaml \
        --out_dir vectors \
        --axis "${AXIS}"
done

echo "All axes done."
