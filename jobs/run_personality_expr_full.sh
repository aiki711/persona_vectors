#!/bin/bash
#PBS -N personality_expr_full
#PBS -q GPU-1
#PBS -o log/personality_expr_full.o%j
#PBS -e log/personality_expr_full.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=120:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
RUN_ID="${PBS_JOBID:-bash_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
mkdir -p log
LOG_FILE="log/personality_expr_full.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== PERSONALITY EXPRESSION PROMPTS FULL EXPERIMENT ==="
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

# Nvidia Libs
export LD_LIBRARY_PATH="$($PYTHON_BIN - <<'PY'
import site, glob, os
paths=[]
for sp in site.getsitepackages():
    paths += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
seen=set(); out=[]
for p in paths:
    if p not in seen:
        out.append(p); seen.add(p)
print(":".join(out))
PY
):${LD_LIBRARY_PATH:-}"

# ==================== Experiment Params ====================
PROMPT_FILE="probe_inputs/personality_expression_30.json"
OUTPUT_BASE="exp_personality_L10-30"
LAYERS="11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"
ALPHAS="-2.0 -1.33 -0.67 0.0 0.67 1.33 2.0"

# Models to test
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3:mistral_7b"
    "meta-llama/Meta-Llama-3-8B-Instruct:llama3_8b"
    "tiiuae/falcon-7b-instruct:falcon_7b"
)

# Traits
TRAITS=("openness" "extraversion" "agreeableness" "conscientiousness" "neuroticism")

# ==================== Run All Experiments ====================
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_short <<< "$model_spec"
    
    echo ""
    echo "=========================================="
    echo "MODEL: $model_name ($model_short)"
    echo "=========================================="
    
    for trait in "${TRAITS[@]}"; do
        echo ""
        echo "--- TRAIT: $trait ---"
        
        RESULTS_DIR="$OUTPUT_BASE/${model_short}/results_personality_expression"
        SCORES_DIR="$OUTPUT_BASE/${model_short}/scores"
        mkdir -p "$RESULTS_DIR" "$SCORES_DIR"
        
        # Find axes bank
        AXES_BANK=""
        # Try specific match first in model subdir
        AXES_BANK=$(find exp_pca_L10-30/${model_short} -name "*instruct*pca*.npz" | head -n 1)
        if [ -z "$AXES_BANK" ]; then
             AXES_BANK=$(find exp_pca_L10-30/${model_short} -name "*base*pca*.npz" | head -n 1)
        fi
        
        # Fallback broad search
        if [ -z "$AXES_BANK" ]; then
             echo "Warning: Specific PCA vector not found in exp_pca_L10-30/${model_short}. Searching broadly..."
             AXES_BANK=$(find . -path "*${model_short}*instruct*pca*.npz" | head -n 1)
        fi
        
        if [ -z "$AXES_BANK" ]; then 
             echo "ERROR: Axes bank not found for $model_short"
             continue
        fi
        
        echo "AXES BANK: $AXES_BANK"

        # Calculate paths
        PROBE_RESULTS="$RESULTS_DIR/${model_short}_base_${trait}_probe_results.jsonl"
        LLM_SCORES="$SCORES_DIR/personality_scores_llm_${trait}.csv"

        # Run steering
        echo "Generating steered text... $(date)"
        # Convert spaces to commas for python script
        LAYERS_CSV=$(echo $LAYERS | tr ' ' ',')
        ALPHAS_CSV=$(echo $ALPHAS | tr ' ' ',')
        
        $PYTHON_BIN scripts/01_run_probe.py \
            --model="$model_name" \
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
        
        echo "Completed $trait for $model_short"
    done
    
    echo ""
    echo "Completed all traits for $model_short"
done

# ==================== Final Summary ====================
echo ""
echo "=========================================="
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "=========================================="
echo "END TIME: $(date)"

echo ""
echo "=== CORRELATION SUMMARY ==="
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r _ model_short <<< "$model_spec"
    echo ""
    echo "Model: $model_short"
    
    for trait in "${TRAITS[@]}"; do
        SCORES_FILE="$OUTPUT_BASE/${model_short}/scores/personality_scores_llm_${trait}.csv"
        
        if [ -f "$SCORES_FILE" ]; then
            corr=$($PYTHON_BIN -c "
import pandas as pd
df = pd.read_csv('$SCORES_FILE')
print(f'{df[\"alpha_total\"].corr(df[\"raw_score_${trait}\"]):.4f}')
" 2>/dev/null || echo "N/A")
            echo "  $trait: r=$corr"
        fi
    done
done

echo ""
echo "Results saved in: $OUTPUT_BASE/"
