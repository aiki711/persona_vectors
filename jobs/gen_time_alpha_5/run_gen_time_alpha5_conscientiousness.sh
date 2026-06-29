#!/bin/bash
#SBATCH --job-name=gen_time_alpha5_conscientiousness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_time_alpha5_conscientiousness.out
#SBATCH --error=log/gen_time_alpha5_conscientiousness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/soft_probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_dyn_gen_time_raw/results"

echo "Starting generation-time dynamic steering (alpha=5.0) for conscientiousness..."

"$PYTHON_BIN" scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_dir "$OUT_DIR" \
    --axis "conscientiousness" \
    --direction "high" \
    --norm_mode "raw_norm" \
    --mask_bank "$MASK_BANK" \
    --update_interval 1 \
    --seed 42 \
    --sweep \
    --alphas "5.0"

echo "Generation-time dynamic steering (alpha=5.0) completed for conscientiousness."
