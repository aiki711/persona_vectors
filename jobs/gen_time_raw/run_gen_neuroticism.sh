#!/bin/bash
#SBATCH --job-name=gen_time_raw_neuroticism
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_time_raw_neuroticism.out
#SBATCH --error=log/gen_time_raw_neuroticism.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_dyn_gen_time_raw/results"

echo "Starting generation-time dynamic steering on test prompts for neuroticism..."

# Run full sweep internally in Python to avoid reloading model
"$PYTHON_BIN" scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_dir "$OUT_DIR" \
    --axis "neuroticism" \
    --direction "high" \
    --norm_mode "raw_norm" \
    --mask_bank "$MASK_BANK" \
    --update_interval 1 \
    --seed 42 \
    --sweep

echo "Generation-time dynamic steering completed on test prompts for neuroticism."
