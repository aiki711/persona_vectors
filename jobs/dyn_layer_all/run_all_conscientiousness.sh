#!/bin/bash
#SBATCH --job-name=dyn_all_conscientiousness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_all_conscientiousness.out
#SBATCH --error=log/dyn_all_conscientiousness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_layer_all_layers/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
STATS="exp_steering_dyn_layer_all_layers/dls_calibration_stats_all.json"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

VALS=(1.0 2.0 4.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0)

for V in "${VALS[@]}"; do
    # ---------------- 1. Z-score Logit Diff ----------------
    echo "Running DLS Z-score logit_diff: Trait=conscientiousness, Alpha=$V"
    JSONL_OUT="${OUT_DIR}/conscientiousness/logit_diff_Val${V}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_logit_diff_Val${V}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/63_run_dyn_layer_zscore.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --stats_path "$STATS" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$V" \
            --direction "high" \
            --method "logit_diff" \
            --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    else
        echo "  [SKIP] Generation already done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation already done: $CSV_OUT"
    fi

    # ---------------- 2. Z-score Anti Alignment ----------------
    echo "Running DLS Z-score anti_alignment: Trait=conscientiousness, Alpha=$V"
    JSONL_OUT="${OUT_DIR}/conscientiousness/anti_alignment_Val${V}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_anti_alignment_Val${V}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/63_run_dyn_layer_zscore.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --stats_path "$STATS" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$V" \
            --direction "high" \
            --method "anti_alignment" \
            --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    else
        echo "  [SKIP] Generation already done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation already done: $CSV_OUT"
    fi

    # ---------------- 3. Relative Anti Alignment ----------------
    echo "Running DLS Relative Anti-alignment: Trait=conscientiousness, Alpha=$V"
    JSONL_OUT="${OUT_DIR}/conscientiousness/relative_anti_alignment_Val${V}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_relative_anti_alignment_Val${V}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/65_run_dyn_layer_relative.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$V" \
            --direction "high" \
            --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    else
        echo "  [SKIP] Generation already done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation already done: $CSV_OUT"
    fi
done
