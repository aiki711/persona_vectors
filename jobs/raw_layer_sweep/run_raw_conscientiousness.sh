#!/bin/bash
#SBATCH --job-name=raw_sweep_conscientiousness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/raw_sweep_conscientiousness.out
#SBATCH --error=log/raw_sweep_conscientiousness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_layer_raw/results"

echo "Running raw single-layer sweep for conscientiousness..."

"$PYTHON_BIN" scripts/04_dyn_layer/103_run_raw_layer_sweep.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_dir "$OUT_DIR" \
    --axis "conscientiousness" \
    --direction "high" \
    --judge_model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --judge_quant "4bit"

echo "Done: conscientiousness"
