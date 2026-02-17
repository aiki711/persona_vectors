#!/bin/bash
# run_raw_pca_mistral_v2.sh

set -euo pipefail

WORKDIR="/home/admin/work/s2550009/persona_vectors"
cd "$WORKDIR"

# プロジェクト設定
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
source persona_steering/bin/activate

# 実験パラメータ
PROMPT_FILE="probe_inputs/personality_expression_30.json"
OUTPUT_BASE="exp_raw_pca"
LAYERS="15,16,17,18,19,20,21,22,23,24,25"
ALPHAS="-5.0,-3.0,-1.0,0.0,1.0,3.0,5.0"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT="mistral_7b"
AXES_BANK="exp_raw_pca/mistral_7b/vectors/mistral_7b_raw_pca.npz"

TRAITS=("openness" "extraversion" "agreeableness" "conscientiousness" "neuroticism")

mkdir -p log

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log/raw_pca_mistral_v2_${TIMESTAMP}.log"

echo "=== RAW PCA MISTRAL V2 EXPERIMENT START: $(date) ===" | tee -a "$LOG_FILE"

for trait in "${TRAITS[@]}"; do
    echo ">>> TRAIT: $trait <<<" | tee -a "$LOG_FILE"
    
    RESULTS_DIR="$OUTPUT_BASE/${MODEL_SHORT}/results_v2"
    mkdir -p "$RESULTS_DIR"
    
    OUT_FILE="$RESULTS_DIR/${MODEL_SHORT}_raw_pca_${trait}_results.jsonl"
    
    python scripts/01_run_probe.py \
        --model="$MODEL_NAME" \
        --trait="$trait" \
        --layers="$LAYERS" \
        --alpha_list="$ALPHAS" \
        --prompt_file="$PROMPT_FILE" \
        --out="$OUT_FILE" \
        --axes_bank="$AXES_BANK" \
        --max_new_tokens=150 \
        --samples=100 \
        --seed=42 2>&1 | tee -a "$LOG_FILE"
done

echo "=== RAW PCA MISTRAL V2 EXPERIMENT FINISHED: $(date) ===" | tee -a "$LOG_FILE"
