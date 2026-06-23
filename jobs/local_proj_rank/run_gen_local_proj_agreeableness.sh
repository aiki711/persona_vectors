#!/bin/bash
#SBATCH --job-name=gen_local_proj_agreeableness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_local_proj_agreeableness.out
#SBATCH --error=log/gen_local_proj_agreeableness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_raw/results"

echo "Starting Local Proj-Rank sweeps for agreeableness..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    # 1. Unmasked Local Proj-Rank
    JSONL_OUT="${OUT_DIR}/agreeableness/local_proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running Local Proj-Rank alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "local_proj_rank" \
            --seed 42
    fi

    # 2. Masked Local Proj-Rank (PDF)
    JSONL_OUT="${OUT_DIR}/agreeableness/masked_local_proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Local Proj-Rank alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "local_proj_rank" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi
done

echo "Local Proj-Rank sweeps completed for agreeableness."
