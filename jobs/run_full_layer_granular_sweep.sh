#!/bin/bash

set -euo pipefail

WORKDIR="$PWD"
RUN_ID="bash_$(date +%Y%m%d_%H%M%S)"

cd "$WORKDIR"
mkdir -p log exp_adaptive_steering/results/full_layer_granular

LOG_FILE="log/full_layer_granular.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== STARTING ULTRA-GRANULAR FULL-LAYER INVESTIGATION (0.01 - 0.1) ==="
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
CONFIG="config/mistral_7b.yaml"
PROMPT_IN="exp_adaptive_steering/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
BASE_OUT_DIR="exp_adaptive_steering/results/full_layer_granular"

# Sweep Matrix (0.03 to 0.15 with 0.03 step)
PARAMS=(0.03 0.06 0.09 0.12 0.15)

# ==================== Loop over Traits ====================
for TRAIT in "${TRAITS[@]}"; do
    echo "=================================================="
    echo "  TRAIT: $TRAIT"
    echo "=================================================="

    OUT_DIR="${BASE_OUT_DIR}/${TRAIT}"
    mkdir -p "$OUT_DIR"

    for VAL in "${PARAMS[@]}"; do
        echo "##################################################"
        echo "  SWEEP POINT: $TRAIT @ Val=$VAL"
        echo "##################################################"

        TAG="Granular_Val${VAL}"
        # Raw generation
        "$PYTHON_BIN" scripts/32b_run_full_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank exp_adaptive_steering/vectors/mean_diff_vectors.npz \
            --prompts "$PROMPT_IN" \
            --out_dir "$OUT_DIR" \
            --axis "$TRAIT" \
            --tau "$VAL" \
            --alpha "$VAL" \
            --mode both \
            --tag "$TAG"
            
        # The script saves to {out_dir}/investigation_{axis}_high_T{tau}_A{alpha}_{tag}.jsonl
        # Note: 32b_run_full_layer_steering.py uses Tau and Alpha in the filename.
        # Use a wider wildcard * to match both T0.1 and T0.10 if needed
        RAW_JSONL=$(ls "$OUT_DIR"/investigation_${TRAIT}_high_T${VAL}*_A${VAL}*_${TAG}.jsonl 2>/dev/null | head -n1)
        
        if [ -z "$RAW_JSONL" ]; then
            # Try a more aggressive fallback for T0.1 vs T0.10
            VAL_STRIP=$(echo "$VAL" | sed 's/0$//; s/\.$//')
            RAW_JSONL=$(ls "$OUT_DIR"/investigation_${TRAIT}_high_T${VAL_STRIP}*_A${VAL_STRIP}*_${TAG}.jsonl 2>/dev/null | head -n1)
        fi
        
        FINAL_JSONL="$OUT_DIR/investigation_${TRAIT}_Val${VAL}.jsonl"
        FINAL_CSV="$OUT_DIR/scores_${TRAIT}_Val${VAL}.csv"
        
        mv "$RAW_JSONL" "$FINAL_JSONL"

        # 2. Evaluate with Llama-3 Judge
        "$PYTHON_BIN" scripts/33_eval_adaptive_steering.py \
            --input "$FINAL_JSONL" \
            --output "$FINAL_CSV" \
            --axis "$TRAIT" \
            --model "$JUDGE_MODEL"

        echo "Finished $TRAIT Val=$VAL"
    done
done

echo "=== MULTI-TRAIT GRANULAR INVESTIGATION COMPLETED ==="
echo "END TIME: $(date)"
