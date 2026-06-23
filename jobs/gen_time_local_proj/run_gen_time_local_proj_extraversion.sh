#!/bin/bash
#SBATCH --job-name=gen_time_local_proj_extraversion
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_time_local_proj_extraversion.out
#SBATCH --error=log/gen_time_local_proj_extraversion.err

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

echo "Starting generation-time Local Proj-Rank sweeps for extraversion..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    # 1. Unmasked Local Proj-Rank
    JSONL_OUT="${OUT_DIR}/extraversion/local_proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running Local Proj-Rank alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "local_proj_rank" \
            --update_interval 1 \
            --seed 42
    fi

    # 2. Masked Local Proj-Rank (PDF)
    JSONL_OUT="${OUT_DIR}/extraversion/masked_local_proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Local Proj-Rank alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "local_proj_rank" \
            --mask_bank "$MASK_BANK" \
            --update_interval 1 \
            --seed 42
    fi
done

echo "Generation-time Local Proj-Rank sweeps completed for extraversion."
