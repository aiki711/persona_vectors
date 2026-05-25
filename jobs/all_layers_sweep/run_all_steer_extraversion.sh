#!/bin/bash
#SBATCH --job-name=all_steer_extraversion
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=log/all_steer_extraversion.out
#SBATCH --error=log/all_steer_extraversion.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
OUT_DIR="exp_steering_layer_analysis/results"

echo "Running Full-Layer (32 layers) single-layer sweep for extraversion..."
"$PYTHON_BIN" scripts/02_base_steering/50_run_all_layers_sweep.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_dir "$OUT_DIR" \
    --axis "extraversion" \
    --direction "high"
