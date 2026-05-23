#!/bin/bash
#SBATCH --job-name=dyn_mid_openness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_mid_openness.out
#SBATCH --error=log/dyn_mid_openness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_ic_fusion_midpoint/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

AMAXES=(0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.5 2.0 3.0)
IC_MODES=(fixed sigmoid soft_plateau)

for MODE in "${IC_MODES[@]}"; do
    for AMAX in "${AMAXES[@]}"; do
        echo "Running DLS + IC Fusion (Midpoint): Trait=openness, Mode=$MODE, AlphaMax=$AMAX"
        JSONL_OUT="${OUT_DIR}/openness/fusion_${MODE}_Val${AMAX}.jsonl"
        CSV_OUT="${OUT_DIR}/openness/scores_fusion_${MODE}_Val${AMAX}.csv"

        if [ ! -f "$JSONL_OUT" ]; then
            "$PYTHON_BIN" scripts/04_dyn_layer/73_run_dyn_ic_fusion.py \
                --config "$CONFIG" \
                --vector_bank "$VECTOR_BANK" \
                --prompts "$PROMPT_IN" \
                --out_dir "$OUT_DIR" \
                --axis "openness" \
                --direction "high" \
                --alpha_max "$AMAX" \
                --ic_mode "$MODE" \
                --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
                --norm_mode "midpoint"
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
done
