#!/bin/bash
#PBS -N raw_pca_expr
#PBS -q GPU-1
#PBS -o log/raw_pca_expr.o%j
#PBS -e log/raw_pca_expr.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
RUN_ID="${PBS_JOBID:-bash_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
mkdir -p log
LOG_FILE="log/raw_pca_expr.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== RAW PCA PERSONALITY EXPERIMENT (PHASE 2) ==="
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
# Configuration
PROMPT_FILE="probe_inputs/personality_expression_30.json"
OUTPUT_BASE="exp_raw_pca"

# Layers: 15-25 (Focused Middle-Late Layers)
LAYERS="15 16 17 18 19 20 21 22 23 24 25"

# Alpha: Wider range for Raw vectors (-5 to +5)
ALPHAS="-5.0 -3.0 -1.0 0.0 1.0 3.0 5.0"

# Models
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT="mistral_7b"
YAML_CONFIG="config/mistral_7b.yaml" 

# Traits
TRAITS=("openness" "extraversion" "agreeableness" "conscientiousness" "neuroticism")

# Axes bank parameters
VECTORS_DIR="$OUTPUT_BASE/${MODEL_SHORT}/vectors"
AXES_BANK="$VECTORS_DIR/mistral_7b_raw_pca.npz"

mkdir -p "$VECTORS_DIR"

# ==================== Step 1: Generate Vectors ====================
echo "--- STEP 1: Generating Raw PCA Vectors ---"
if [ -f "$AXES_BANK" ]; then
    echo "Vector file already exists: $AXES_BANK"
    echo "Skipping generation. (Remove file to force regenerate)"
else
    echo "Running 00_prepare_vectors_raw_pca.py..."
    $PYTHON_BIN scripts/00_prepare_vectors_raw_pca.py \
        --config "$YAML_CONFIG" \
        --bank_path "$AXES_BANK" \
        --model_name "$MODEL_NAME" 
fi

if [ ! -f "$AXES_BANK" ]; then
    echo "ERROR: Axes bank generation failed!"
    exit 1
fi

echo "Vectors ready at: $AXES_BANK"


# ==================== Step 2: Run Experiments ====================
echo "--- STEP 2: Running Steering Experiments ---"

for trait in "${TRAITS[@]}"; do
    echo ""
    echo ">>> TRAIT: $trait <<<"
    
    # Directory Structure: exp_raw_pca/mistral_7b/{results, scores}
    RESULTS_DIR="$OUTPUT_BASE/${MODEL_SHORT}/results"
    SCORES_DIR="$OUTPUT_BASE/${MODEL_SHORT}/scores"
    mkdir -p "$RESULTS_DIR" "$SCORES_DIR"
    
    # Files
    PROBE_RESULTS="$RESULTS_DIR/${MODEL_SHORT}_raw_pca_${trait}_results.jsonl"
    LLM_SCORES="$SCORES_DIR/scores_${trait}.csv"

    # Convert lists to comma-separated strings for python args
    LAYERS_CSV=$(echo $LAYERS | tr ' ' ',')
    ALPHAS_CSV=$(echo $ALPHAS | tr ' ' ',')
    
    # 2.1 Run Probe
    if [ -f "$PROBE_RESULTS" ]; then
        echo "  Results file already exists: $PROBE_RESULTS"
        echo "  Skipping probe. (Remove file to force re-run)"
    else
        echo "  Generating steered text... $(date)"
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
    fi
    
    # 2.2 Calculate Scores
    if [ -f "$PROBE_RESULTS" ]; then
        echo "  Calculating LLM scores... $(date)"
        $PYTHON_BIN scripts/14_calc_personality_score_llm.py \
            "$PROBE_RESULTS" \
            --output "$LLM_SCORES"
    else
        echo "  ERROR: Results file not found: $PROBE_RESULTS"
    fi
    
    echo "  Completed $trait"
done

echo ""
echo "=========================================="
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "=========================================="
echo "END TIME: $(date)"

# ==================== Correlation Summary ====================
echo ""
echo "=== CORRELATION SNAPSHOT ==="
for trait in "${TRAITS[@]}"; do
    SCORES_FILE="$OUTPUT_BASE/${MODEL_SHORT}/scores/scores_${trait}.csv"
    if [ -f "$SCORES_FILE" ]; then
        corr=$($PYTHON_BIN -c "
import pandas as pd
try:
    df = pd.read_csv('$SCORES_FILE')
    # Valid columns? Usually 'alpha_total' and 'raw_score_{trait}'
    # But 01_run_probe output might change naming if logic differs? 
    # Assuming standard columns from 14_calc_personality_score_llm.py
    print(f'{df[\"alpha_total\"].corr(df[\"raw_score_${trait}\"]):.4f}')
except Exception as e:
    print('N/A')
" 2>/dev/null || echo "Error")
        echo "  $trait: r=$corr"
    fi
done

echo ""
echo "Output Directory: $OUTPUT_BASE/$MODEL_SHORT"
