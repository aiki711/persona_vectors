#!/bin/bash
#SBATCH --job-name=eval_test_dls_rank1000
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:30:00
#SBATCH --output=log/eval_test_dls_rank1000.out
#SBATCH --error=log/eval_test_dls_rank1000.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting evaluation of test_dls_rank1000..."

"$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
    --input exp_steering_dyn_layer_proj_prior/results_test_rank1000/extraversion/rank_only_Val5.0.jsonl \
    --output exp_steering_dyn_layer_proj_prior/results_test_rank1000/extraversion/scores_rank_only_Val5.0.csv \
    --axis extraversion \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --quant 4bit

echo "Done."
