#!/bin/bash
#PBS -N filtered_pca_expr
#PBS -q GPU-1
#PBS -o log/filtered_pca_expr.o%j
#PBS -e log/filtered_pca_expr.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
RUN_ID="${PBS_JOBID:-bash_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
mkdir -p log
LOG_FILE="log/filtered_pca_expr.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== FILTERED PCA PERSONALITY EXPERIMENT ==="
echo "START TIME: $(date)"

# ==================== Project Setup ====================
export PROJECT_DIR="$WORKDIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:$PROJECT_DIR/scripts:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Venv
VENV="$PROJECT_DIR/persona_steering"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHON_BIN="$VENV/bin/python"

# ==================== Experiment Params ====================
PROMPT_FILE="probe_inputs/personality_expression_30.json"
OUTPUT_BASE="exp_filtered_pca"
LAYERS="11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"
ALPHAS="-2.0 -1.33 -0.67 0.0 0.67 1.33 2.0"

# Models to test
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT="mistral_7b"

# Traits
TRAITS=("openness" "extraversion" "agreeableness" "conscientiousness" "neuroticism")

# Axes bank
AXES_BANK="vectors/mistral_7b_test_filtered.npz"

if [ ! -f "$AXES_BANK" ]; then
    echo "ERROR: Axes bank not found: $AXES_BANK"
    exit 1
fi

echo "AXES BANK: $AXES_BANK"

# ==================== Run All Traits ====================
for trait in "${TRAITS[@]}"; do
    echo ""
    echo "--- TRAIT: $trait ---"
    
    RESULTS_DIR="$OUTPUT_BASE/${MODEL_SHORT}/results_personality_expression"
    SCORES_DIR="$OUTPUT_BASE/${MODEL_SHORT}/scores"
    mkdir -p "$RESULTS_DIR" "$SCORES_DIR"
    
    # Calculate paths
    PROBE_RESULTS="$RESULTS_DIR/${MODEL_SHORT}_filtered_pca_${trait}_probe_results.jsonl"
    LLM_SCORES="$SCORES_DIR/personality_scores_llm_${trait}.csv"

    # Run steering
    echo "Generating steered text... $(date)"
    # Convert spaces to commas for python script
    LAYERS_CSV=$(echo $LAYERS | tr ' ' ',')
    ALPHAS_CSV=$(echo $ALPHAS | tr ' ' ',')
    
    $PYTHON_BIN scripts/01_run_probe.py \
        --model="$MODEL_NAME" \
        --trait="$trait" \
        --layers="$LAYERS_CSV" \
        --alpha_list="$ALPHAS_CSV" \
        --prompt_file="$PROMPT_FILE" \
        --out="$PROBE_RESULTS" \
        --axes_bank="$AXES_BANK" \
        --max_new_tokens=150 \
        --samples=100 \
        --seed=42
    
    # Calculate LLM scores
    if [ -f "$PROBE_RESULTS" ]; then
        echo "Calculating LLM scores... $(date)"
        $PYTHON_BIN scripts/14_calc_personality_score_llm.py \
            "$PROBE_RESULTS" \
            --output "$LLM_SCORES"
    fi
    
    echo "Completed $trait for $MODEL_SHORT"
done

echo ""
echo "=========================================="
echo "=== ALL TRAITS COMPLETE ==="
echo "=========================================="
echo "END TIME: $(date)"

# Correlation Summary
echo ""
echo "=== CORRELATION SUMMARY ==="
for trait in "${TRAITS[@]}"; do
    SCORES_FILE="$OUTPUT_BASE/${MODEL_SHORT}/scores/personality_scores_llm_${trait}.csv"
    if [ -f "$SCORES_FILE" ]; then
        corr=$($PYTHON_BIN -c "
import pandas as pd
try:
    df = pd.read_csv('$SCORES_FILE')
    print(f'{df[\"alpha_total\"].corr(df[\"raw_score_${trait}\"]):.4f}')
except Exception:
    print('N/A')
" 2>/dev/null || echo "N/A")
        echo "  $trait: r=$corr"
    fi
done

echo ""
echo "Results saved in: $OUTPUT_BASE/$MODEL_SHORT"
