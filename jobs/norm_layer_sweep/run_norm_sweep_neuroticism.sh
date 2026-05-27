#!/bin/bash
#SBATCH --job-name=norm_sweep_neuroticism
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=log/norm_sweep_neuroticism.out
#SBATCH --error=log/norm_sweep_neuroticism.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp/exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
OUT_DIR="exp_steering_layer_norm/results"

echo "Running norm-scaled single-layer sweep for neuroticism..."

"$PYTHON_BIN" scripts/04_dyn_layer/93_run_norm_layer_sweep.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_dir "$OUT_DIR" \
    --axis "neuroticism" \
    --direction "high"

echo "Done: neuroticism"
