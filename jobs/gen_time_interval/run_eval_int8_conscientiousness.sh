#!/bin/bash
#SBATCH --job-name=eval_int8_conscientiousness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/eval_int8_conscientiousness.out
#SBATCH --error=log/eval_int8_conscientiousness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting batch evaluation for generation-time steering of conscientiousness (interval=8)..."

"$PYTHON_BIN" scripts/04_dyn_layer/115_batch_eval.py \
    --results_dir "exp_steering_dyn_gen_time_interval_raw/results_interval8/conscientiousness" \
    --axis "conscientiousness" \
    --quant "4bit"

echo "Batch evaluation completed for conscientiousness (interval=8)."
