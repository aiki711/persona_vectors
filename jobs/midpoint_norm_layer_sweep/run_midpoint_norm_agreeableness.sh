#!/bin/bash
#SBATCH --job-name=midpoint_norm_sweep_agreeableness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/midpoint_norm_sweep_agreeableness.out
#SBATCH --error=log/midpoint_norm_sweep_agreeableness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_layer_midpoint_norm/results"

echo "Running midpoint-norm scaled single-layer sweep for agreeableness..."

"$PYTHON_BIN" scripts/04_dyn_layer/106_run_online_norm_layer_sweep.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_dir "$OUT_DIR" \
    --axis "agreeableness" \
    --direction "high" \
    --judge_model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --judge_quant "4bit"

echo "Done: agreeableness"
