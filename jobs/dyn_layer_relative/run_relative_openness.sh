#!/bin/bash
#SBATCH --job-name=dyn_rel_openness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_rel_openness.out
#SBATCH --error=log/dyn_rel_openness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_layer_relative/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
# midpoint が含まれる再生成済みのベクトルバンクを使用
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

VALS=(1.0 2.0 4.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0)

for V in "${VALS[@]}"; do
    echo "Running Relative Anti-Alignment DLS: Trait=openness, Alpha=$V"
    JSONL_OUT="${OUT_DIR}/openness/relative_anti_alignment_Val${V}.jsonl"
    CSV_OUT="${OUT_DIR}/openness/scores_relative_anti_alignment_Val${V}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/65_run_dyn_layer_relative.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --out_dir "$OUT_DIR" \
            --axis "openness" \
            --alpha "$V" \
            --direction "high"
    else
        echo "  [SKIP] Generation already done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "openness" \
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation already done: $CSV_OUT"
    fi
done
