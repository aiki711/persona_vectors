#!/bin/bash
#PBS -N rerun_viz
#PBS -q GPU-1
#PBS -o log/rerun_viz.o%j
#PBS -e log/rerun_viz.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=120:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
RUN_ID="${PBS_JOBID:-rerun_viz_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
mkdir -p log
LOG_FILE="log/rerun_viz.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== RERUNNING VISUALIZATION ==="
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

# ==================== Constants ====================
TRAITS=("openness" "conscientiousness" "extraversion" "agreeableness" "neuroticism")

MODEL_SPECS=(
  "mistral_7b|mistralai/Mistral-7B-v0.3|mistralai/Mistral-7B-Instruct-v0.3|-2,-1.33,-0.67,0,0.67,1.33,2|-2,-1.33,-0.67,0,0.67,1.33,2"
  "llama3_8b|meta-llama/Meta-Llama-3-8B|meta-llama/Meta-Llama-3-8B-Instruct|-7,-4.7,-2.3,0,2.3,4.7,7|-7,-4.7,-2.3,0,2.3,4.7,7"
  "olmo3_7b|allenai/Olmo-3-1025-7B|allenai/Olmo-3-7B-Instruct|-15,-10,-5,0,5,10,15|-15,-10,-5,0,5,10,15"
  "qwen25_7b|Qwen/Qwen2.5-7B|Qwen/Qwen2.5-7B-Instruct|-50,-33,-17,0,17,33,50|-50,-33,-17,0,17,33,50"
  "gemma2_9b|google/gemma-2-9b|google/gemma-2-9b-it|-200,-133,-67,0,67,133,200|-200,-133,-67,0,67,133,200"
  "falcon3_7b|tiiuae/Falcon3-7B-Base|tiiuae/Falcon3-7B-Instruct|-100,-67,-33,0,33,67,100|-100,-67,-33,0,33,67,100"
)

PROMPT_SETS=(
  "mtbench|exp/01_probe_inputs/mtbench_50.json"
  "synthetic|exp/01_probe_inputs/synthetic_50.json"
  "ipip|exp/01_probe_inputs/ipip_50.json"
)

# ==================== Helper ====================
is_nonempty_file() { local f="$1"; [[ -f "$f" && -s "$f" ]]; }

run_viz_for_dir() {
    local tag="$1"
    local results_dir="$2"
    local suffix="$3" 
    
    local out_plots="$results_dir/plots"
    mkdir -p "$out_plots"
    
    echo "--- Viz for $tag / $suffix ---"
    
    echo "[Viz] 15_text_sensitivity"
    "$PYTHON_BIN" scripts/15_text_sensitivity_visualize.py \
        --metrics_glob "${results_dir}/*_text_metrics.csv" \
        --score_glob "${results_dir}/*_personality_scores.csv" \
        --out_dir "$out_plots" \
        --tag "${tag}_${suffix}"

    echo "[Viz] 18_visualize_scatter"
    "$PYTHON_BIN" scripts/18_visualize_scatter.py \
        --metrics_glob "${results_dir}/*_text_metrics.csv" \
        --score_glob "${results_dir}/*_personality_scores.csv" \
        --jsonl_glob "${results_dir}/*_probe_results.jsonl" \
        --out_dir "$out_plots" \
        --tag "${tag}_${suffix}"
    
    local slope_csv="$results_dir/slopes/slopes_${tag}_${suffix}.csv"
    mkdir -p "$(dirname "$slope_csv")"
    local base_all="${results_dir}/${tag}_base_alltraits.jsonl"
    local instr_all="${results_dir}/${tag}_instruct_alltraits.jsonl"
    
    # Recalculate slopes just in case data changed (or file missing)
    if is_nonempty_file "$base_all" && is_nonempty_file "$instr_all"; then
         echo "[Calc] 02_probe_slopes"
         "$PYTHON_BIN" scripts/02_probe_slopes_from_logs.py \
              --base_json  "$base_all" \
              --instr_json "$instr_all" \
              --out_csv    "$slope_csv" \
              --pooling asst \
              --axis_mode pairwise
        
        echo "[Viz] 16_combined_metrics"
        "$PYTHON_BIN" scripts/16_visualize_combined_metrics.py \
            --internal_csv "$slope_csv" \
            --external_csv "$out_plots/${tag}_${suffix}_text_sensitivities.csv" \
            --out_dir "$out_plots" \
            --tag "${tag}_${suffix}"
    fi
}

# ==================== MAIN ====================

for pset in "${PROMPT_SETS[@]}"; do
    IFS='|' read -r PNAME PFILE <<< "$pset"
    
    for spec in "${MODEL_SPECS[@]}"; do
        IFS='|' read -r TAG BASE_ID INSTR_ID ALPHAS_BASE ALPHAS_INSTR <<< "$spec"
        
        # REMOVE 'local' keyword here from previous version
        results_dir="exp/${TAG}/results_${PNAME}"
        if [ -d "$results_dir" ]; then
            run_viz_for_dir "$TAG" "$results_dir" "$PNAME"
        else
            echo "[Skip] Directory not found: $results_dir"
        fi
    done
done

echo "=== DONE ==="
