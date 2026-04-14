#!/bin/bash
# jobs/run_steering_layer_sweep.sh
#
# Layer-Sweep 実験:
#   全5特性 x 11レイヤー（3層おき: 0,3,6,...,30）x 5alpha (0.03~0.15)
#   = 275条件 x (生成 + Constant + Adaptive + Llama-3 Judge評価)
#
# 使用法:
#   bash jobs/run_steering_layer_sweep.sh

set -euo pipefail

WORKDIR="$PWD"
RUN_ID="bash_$(date +%Y%m%d_%H%M%S)"

cd "$WORKDIR"
mkdir -p log exp_steering_layer_sweep/results exp_steering_layer_sweep/figures exp_steering_layer_sweep/summary_stats

LOG_FILE="log/steering_layer_sweep.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== STARTING STEERING LAYER-SWEEP EXPERIMENT ==="
echo "START TIME: $(date)"

# ==================== Project Setup ====================
export PROJECT_DIR="$WORKDIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:$PROJECT_DIR/scripts:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ==================== Tokens ====================
if [ -f "$PROJECT_DIR/.hf_token" ]; then
  export HUGGINGFACE_HUB_TOKEN="$(head -n1 "$PROJECT_DIR/.hf_token" | tr -d '\r\n' | sed 's/^Bearer[[:space:]]\+//')"
fi

# ==================== venv ====================
source "$PROJECT_DIR/persona_steering/bin/activate"
PYTHON_BIN="python"

# ==================== Params ====================
TRAITS=("extraversion" "neuroticism" "openness" "conscientiousness" "agreeableness")
LAYERS=(0 3 6 9 12 15 18 21 24 27 30)
PARAMS=(0.1 0.2 0.3 0.4 0.5)

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_adaptive_steering/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_adaptive_steering/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
BASE_OUT_DIR="exp_steering_layer_sweep/results"

TOTAL=$((${#TRAITS[@]} * ${#LAYERS[@]} * ${#PARAMS[@]}))
COUNT=0

echo "  Traits  : ${TRAITS[*]}"
echo "  Layers  : ${LAYERS[*]}"
echo "  Vals    : ${PARAMS[*]}"
echo "  Total   : $TOTAL conditions"
echo ""

# ==================== Main Loop ====================
# TRAITとLAYERを外側, VALを内側にしてモデル再ロードを最小化

for TRAIT in "${TRAITS[@]}"; do
    echo "=========================================="
    echo "  TRAIT: $TRAIT"
    echo "=========================================="

    OUT_DIR="${BASE_OUT_DIR}/${TRAIT}"
    mkdir -p "$OUT_DIR"

    for LAYER in "${LAYERS[@]}"; do
        echo "------------------------------------------"
        echo "  LAYER: $LAYER"
        echo "------------------------------------------"

        for VAL in "${PARAMS[@]}"; do
            COUNT=$((COUNT + 1))
            echo "### [$COUNT/$TOTAL] TRAIT=$TRAIT LAYER=$LAYER VAL=$VAL ###"

            JSONL_OUT="${OUT_DIR}/layer_${LAYER}_Val${VAL}.jsonl"
            CSV_OUT="${OUT_DIR}/scores_layer_${LAYER}_Val${VAL}.csv"

            # すでに評価済みならスキップ
            if [ -f "$CSV_OUT" ]; then
                echo "  [SKIP] Already evaluated: $CSV_OUT"
                continue
            fi

            # Step1: 生成（未済の場合のみ）
            if [ ! -f "$JSONL_OUT" ]; then
                "$PYTHON_BIN" scripts/40_run_layer_sweep.py \
                    --config "$CONFIG" \
                    --vector_bank "$VECTOR_BANK" \
                    --prompts "$PROMPT_IN" \
                    --out_dir "$OUT_DIR" \
                    --axis "$TRAIT" \
                    --target_layer "$LAYER" \
                    --tau "$VAL" \
                    --alpha "$VAL" \
                    --mode both
            else
                echo "  [SKIP] Generation already done: $JSONL_OUT"
            fi

            # Step2: Llama-3 ジャッジによる評価
            "$PYTHON_BIN" scripts/33_eval_adaptive_steering.py \
                --input "$JSONL_OUT" \
                --output "$CSV_OUT" \
                --axis "$TRAIT" \
                --model "$JUDGE_MODEL"

            echo "  Done: TRAIT=$TRAIT LAYER=$LAYER VAL=$VAL"
            echo ""
        done
    done
done

echo "=== STEERING LAYER-SWEEP EXPERIMENT COMPLETED ==="
echo "END TIME: $(date)"
echo "Results: $BASE_OUT_DIR"
