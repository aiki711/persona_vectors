#!/bin/bash
#PBS -N personality_expr_pilot
#PBS -q GPU-1
#PBS -o log/personality_expr_pilot.o%j
#PBS -e log/personality_expr_pilot.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=48:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
RUN_ID="${PBS_JOBID:-bash_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
mkdir -p log
LOG_FILE="log/personality_expr_pilot.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== PERSONALITY EXPRESSION PROMPTS PILOT EXPERIMENT ==="
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
# Pilot Test: Mistral-7B, Openness only
PROMPT_FILE="probe_inputs/personality_expression_30.json"
OUTPUT_BASE="exp_personality_L10-30"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT="mistral_7b"
TRAIT="openness"
# Comma-separated for python script
LAYERS="11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
ALPHAS="-2.0,-1.33,-0.67,0.0,0.67,1.33,2.0"

# Output files
RESULTS_DIR="$OUTPUT_BASE/${MODEL_SHORT}/results_personality_expression"
SCORES_DIR="$OUTPUT_BASE/${MODEL_SHORT}/scores"
mkdir -p "$RESULTS_DIR" "$SCORES_DIR"
PROBE_RESULTS="$RESULTS_DIR/${MODEL_SHORT}_base_${TRAIT}_probe_results.jsonl"

echo "PROMPT FILE: $PROMPT_FILE"
echo "OUTPUT FILE: $PROBE_RESULTS"
echo "MODEL: $MODEL_NAME"
echo "TRAIT: $TRAIT"
echo "LAYERS: $LAYERS"

# ==================== Run Steering Experiment ====================
echo ""
echo "=== GENERATING STEERED TEXT ==="
echo "$(date)"

# Need axes bank from prepare_vectors
# Look in exp_pca_L10-30/${MODEL_SHORT}
AXES_BANK=$(find exp_pca_L10-30/${MODEL_SHORT} -name "*instruct*pca*.npz" | head -n 1)

if [ -z "$AXES_BANK" ]; then
    echo "Warning: Instruct PCA vector not found. Trying base..."
    AXES_BANK=$(find exp_pca_L10-30/${MODEL_SHORT} -name "*base*pca*.npz" | head -n 1)
fi

if [ -z "$AXES_BANK" ]; then
    echo "Warning: Specific PCA vector not found in exp_pca_L10-30/${MODEL_SHORT}. Searching broadly..."
    # Search for any npz in the model folder, likely under legacy exp/ if not in exp_pca_L10-30
    AXES_BANK=$(find . -path "*${MODEL_SHORT}*instruct*pca*.npz" | head -n 1)
fi

if [ -z "$AXES_BANK" ]; then
    echo "ERROR: Axes bank not found anywhere for ${MODEL_SHORT}!"
    exit 1
fi

echo "AXES BANK: $AXES_BANK"

$PYTHON_BIN scripts/01_run_probe.py \
    --model="$MODEL_NAME" \
    --trait="$TRAIT" \
    --layers="$LAYERS" \
    --alpha_list="$ALPHAS" \
    --prompt_file="$PROMPT_FILE" \
    --out="$PROBE_RESULTS" \
    --axes_bank="$AXES_BANK" \
    --max_new_tokens=150 \
    --samples=100 \
    --seed=42

echo "Steering complete: $(date)"

# ==================== Calculate External Scores ====================
echo ""
echo "=== CALCULATING EXTERNAL SCORES (LLM JUDGE) ==="
echo "$(date)"

PROBE_RESULTS="$RESULTS_DIR/${MODEL_SHORT}_base_${TRAIT}_probe_results.jsonl"
LLM_SCORES="$SCORES_DIR/personality_scores_llm_${TRAIT}.csv"

if [ -f "$PROBE_RESULTS" ]; then
    $PYTHON_BIN scripts/14_calc_personality_score_llm.py \
        "$PROBE_RESULTS" \
        --output "$LLM_SCORES" \
        --trait "$TRAIT"
    
    echo "LLM scores saved to: $LLM_SCORES"
else
    echo "WARNING: Probe results not found at $PROBE_RESULTS"
fi

# ==================== Quick Analysis ====================
echo ""
echo "=== QUICK ANALYSIS ==="
echo "$(date)"

if [ -f "$LLM_SCORES" ]; then
    $PYTHON_BIN - <<ANALYSIS
import pandas as pd
import sys

try:
    df = pd.read_csv("$LLM_SCORES")
    
    # Summary by alpha
    summary = df.groupby('alpha_total')['raw_score_${TRAIT}'].agg(['mean', 'std', 'count'])
    print("\n=== Scores by Alpha ===")
    print(summary)
    
    # Check correlation
    corr = df['alpha_total'].corr(df['raw_score_${TRAIT}'])
    print(f"\nCorrelation (alpha vs score): {corr:.4f}")
    
    # Score range
    print(f"\nScore range: {df['raw_score_${TRAIT}'].min():.1f} to {df['raw_score_${TRAIT}'].max():.1f}")
    
    # Check if scores vary
    unique_scores = df['raw_score_${TRAIT}'].nunique()
    print(f"Unique score values: {unique_scores}")
    
    if unique_scores > 1 and abs(corr) > 0.3:
        print("\n✓ SUCCESS: Scores show variation and correlation with alpha!")
    else:
        print("\n⚠ WARNING: Scores may still be flat. Manual inspection needed.")
        
except Exception as e:
    print(f"Analysis failed: {e}")
    sys.exit(1)
ANALYSIS

fi

echo ""
echo "=== PILOT EXPERIMENT COMPLETE ==="
echo "END TIME: $(date)"
echo "Check results in: $OUTPUT_BASE/$MODEL_SHORT/"
