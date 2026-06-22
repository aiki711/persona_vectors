#!/bin/bash
#SBATCH --job-name=gen_int8_openness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_int8_openness.out
#SBATCH --error=log/gen_int8_openness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_dyn_gen_time_interval_raw/results_interval8"

echo "Starting generation-time DLS on test prompts for openness (interval=8)..."

"$PYTHON_BIN" scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_dir "$OUT_DIR" \
    --axis "openness" \
    --direction "high" \
    --norm_mode "raw_norm" \
    --mask_bank "$MASK_BANK" \
    --update_interval 8 \
    --seed 42 \
    --sweep

echo "Generation-time DLS completed for openness (interval=8)."
