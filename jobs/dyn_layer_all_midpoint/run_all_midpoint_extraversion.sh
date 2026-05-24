#!/bin/bash
#SBATCH --job-name=dyn_mid_all_extraversion
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_mid_all_extraversion.out
#SBATCH --error=log/dyn_mid_all_extraversion.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_layer_all_layers_midpoint/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
STATS="exp_steering_dyn_layer_all_layers_midpoint/dls_calibration_stats_all_midpoint.json"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

VALS=(0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0)

for V in "${VALS[@]}"; do
    # ---------------- 1. Z-score Logit Diff (Midpoint Normalized) ----------------
    echo "Running DLS Z-score logit_diff (Midpoint): Trait=extraversion, Alpha=$V"
    JSONL_OUT="${OUT_DIR}/extraversion/logit_diff_Val${V}.jsonl"
    CSV_OUT="${OUT_DIR}/extraversion/scores_logit_diff_Val${V}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/63_run_dyn_layer_zscore.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --stats_path "$STATS" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$V" \
            --direction "high" \
            --method "logit_diff" \
            --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
            --norm_mode "midpoint"
    else
        echo "  [SKIP] Generation already done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "extraversion" \
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation already done: $CSV_OUT"
    fi

    # ---------------- 2. Z-score Anti Alignment (Midpoint Normalized) ----------------
    echo "Running DLS Z-score anti_alignment (Midpoint): Trait=extraversion, Alpha=$V"
    JSONL_OUT="${OUT_DIR}/extraversion/anti_alignment_Val${V}.jsonl"
    CSV_OUT="${OUT_DIR}/extraversion/scores_anti_alignment_Val${V}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/63_run_dyn_layer_zscore.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --stats_path "$STATS" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$V" \
            --direction "high" \
            --method "anti_alignment" \
            --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
            --norm_mode "midpoint"
    else
        echo "  [SKIP] Generation already done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "extraversion" \
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation already done: $CSV_OUT"
    fi
done
