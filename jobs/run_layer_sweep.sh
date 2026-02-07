#!/bin/bash
#PBS -N layer_sweep
#PBS -q GPU-1
#PBS -o log/layer_sweep.o%j
#PBS -e log/layer_sweep.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=120:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
RUN_ID="${PBS_JOBID:-bash_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
mkdir -p log
LOG_FILE="log/layer_sweep.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== STARTING LAYER SWEEP EXPERIMENTS ==="
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
# Layer Range Configuration
LAYER_START=10
LAYER_END=30
LAYER_SUFFIX="_L${LAYER_START}-${LAYER_END}"

# Dynamic Experiment Root: exp -> exp_L10-30
EXP_OUTPUT_ROOT="exp${LAYER_SUFFIX}"

echo "LAYER CONFIG: ${LAYER_START} to ${LAYER_END} (Suffix: ${LAYER_SUFFIX})"
echo "OUTPUT ROOT: ${EXP_OUTPUT_ROOT}"

TRAITS=("openness" "conscientiousness" "extraversion" "agreeableness" "neuroticism")

MODEL_SPECS=(
  "mistral_7b|mistralai/Mistral-7B-v0.3|mistralai/Mistral-7B-Instruct-v0.3|-2,-1.33,-0.67,0,0.67,1.33,2|-2,-1.33,-0.67,0,0.67,1.33,2"
  "llama3_8b|meta-llama/Meta-Llama-3-8B|meta-llama/Meta-Llama-3-8B-Instruct|-7,-4.7,-2.3,0,2.3,4.7,7|-7,-4.7,-2.3,0,2.3,4.7,7"
  "olmo3_7b|allenai/Olmo-3-1025-7B|allenai/Olmo-3-7B-Instruct|-15,-10,-5,0,5,10,15|-15,-10,-5,0,5,10,15"
  "qwen25_7b|Qwen/Qwen2.5-7B|Qwen/Qwen2.5-7B-Instruct|-50,-33,-17,0,17,33,50|-50,-33,-17,0,17,33,50"
  "gemma2_9b|google/gemma-2-9b|google/gemma-2-9b-it|-200,-133,-67,0,67,133,200|-200,-133,-67,0,67,133,200"
  "falcon3_7b|tiiuae/Falcon3-7B-Base|tiiuae/Falcon3-7B-Instruct|-100,-67,-33,0,33,67,100|-100,-67,-33,0,33,67,100"
)

# New Prompt Sets
# "Name | JSON Path | Template"
PROMPT_SETS=(
  "writing|probe_inputs/writing_prompts_30.json|flexible"
  "advice|probe_inputs/opinion_advice_30.json|flexible"
)

# ==================== Helpers ====================
is_nonempty_file() { local f="$1"; [[ -f "$f" && -s "$f" ]]; }

prepare_axes_if_needed() {
  local model_id="$1"
  local ax_bank="$2"
  local config_file="exp/configs/big5_vectors.yaml" # Fixed config path
  
  if ! is_nonempty_file "$ax_bank"; then
      echo "[WARN] Axis bank $ax_bank missing. You might need to run main sweep first or ensure 00_prepare_vectors runs."
      # Just try running it
      "$PYTHON_BIN" scripts/00_prepare_vectors.py --config "$config_file" --bank_path "$ax_bank"
  fi
}

run_probe_if_needed() {
  local split="$1"
  local trait="$2"
  local model_id="$3"
  local axes_bank="$4"
  local out_jsonl="$5"
  local alpha_list="$6"
  local prompt_file="$7"
  local l_start="$8"
  local l_end="$9"
  local template_type="${10:-standard}"

  if is_nonempty_file "$out_jsonl"; then
    echo "[SKIP] probe exists: $out_jsonl"
    return 0
  fi

  echo "[RUN ] 01_run_probe.py -> $out_jsonl"
  "$PYTHON_BIN" scripts/01_run_probe.py \
    --model       "$model_id" \
    --axes_bank   "$axes_bank" \
    --trait       "$trait" \
    --alpha_list="$alpha_list" \
    --out         "$out_jsonl" \
    --prompt_file "$prompt_file" \
    --layer_start "$l_start" \
    --layer_end   "$l_end" \
    --samples     50 \
    --template_type "$template_type"
}

run_internal_analysis_if_needed() {
  local split="$1"
  local trait="$2"
  local model_id="$3"
  local axes_bank="$4"
  local out_csv="$5"
  local alpha_list="$6"
  local prompt_file="$7"
  local l_start="$8"
  local l_end="$9"

  if is_nonempty_file "$out_csv"; then
    echo "[SKIP] internal analysis exists: $out_csv"
    return 0
  fi

  echo "[RUN ] 25_analyze_internal_states.py -> $out_csv"
  "$PYTHON_BIN" scripts/25_analyze_internal_states.py \
    --model_id    "$model_id" \
    --vector_path "$axes_bank" \
    --trait       "$trait" \
    --alpha       "$alpha_list" \
    --prompt_file "$prompt_file" \
    --out_file    "$out_csv" \
    --steer_layers "${l_start}-${l_end}" \
    --limit       50
}

concat_alltraits() {
  local tag="$1"
  local results_dir="$2"
  local split="$3"
  local out_all="$4"

  rm -f "$out_all"
  for trait in "${TRAITS[@]}"; do
    local f="${results_dir}/${tag}_${split}_${trait}_probe_results.jsonl"
    if is_nonempty_file "$f"; then
      cat "$f" >> "$out_all"
    fi
  done
}

run_text_analysis() {
  local tag="$1"
  local results_dir="$2"
  local split="$3"
  
  local input_jsonl="${results_dir}/${tag}_${split}_alltraits.jsonl"
  if ! is_nonempty_file "$input_jsonl"; then return 0; fi

  # 13: Text Metrics
  local metrics_csv="${results_dir}/${tag}_${split}_text_metrics.csv"
  if ! is_nonempty_file "$metrics_csv"; then
    echo "[RUN ] 13_text_metrics_vs_alpha.py"
    "$PYTHON_BIN" scripts/13_text_metrics_vs_alpha.py "$input_jsonl" \
      --output "$metrics_csv" \
      --baseline 0 \
      --mode all
  fi

  # 14: Personality Score
  local score_csv="${results_dir}/${tag}_${split}_personality_scores.csv"
  if ! is_nonempty_file "$score_csv"; then
    echo "[RUN ] 14_calc_personality_score.py"
    "$PYTHON_BIN" scripts/14_calc_personality_score.py "$input_jsonl" \
      --output "$score_csv" \
      --batch_size 32 \
      --model "KevSun/Personality_LM"
  fi
}

run_viz_for_dir() {
    local tag="$1"
    local results_dir="$2"
    local suffix="$3" # e.g. "mtbench"
    
    local out_plots="$results_dir/plots"
    mkdir -p "$out_plots"
    
    echo "[Viz] 15_text_sensitivity ($suffix)"
    "$PYTHON_BIN" scripts/15_text_sensitivity_visualize.py \
        --metrics_glob "${results_dir}/*_text_metrics.csv" \
        --score_glob "${results_dir}/*_personality_scores.csv" \
        --out_dir "$out_plots" \
        --tag "${tag}_${suffix}"

    # 18: Scatter Plots (Alpha vs Score/Distance)
    echo "[Viz] 18_visualize_scatter ($suffix)"
    "$PYTHON_BIN" scripts/18_visualize_scatter.py \
        --metrics_glob "${results_dir}/*_text_metrics.csv" \
        --score_glob "${results_dir}/*_personality_scores.csv" \
        --jsonl_glob "${results_dir}/*_probe_results.jsonl" \
        --out_dir "$out_plots" \
        --tag "${tag}_${suffix}"
    
    # Combined Metrics (Skip slopes calc if file missing)
    local slope_csv="$results_dir/slopes/slopes_${tag}_${suffix}.csv"
    mkdir -p "$(dirname "$slope_csv")"
    local base_all="${results_dir}/${tag}_base_alltraits.jsonl"
    local instr_all="${results_dir}/${tag}_instruct_alltraits.jsonl"
    
    if is_nonempty_file "$base_all" && is_nonempty_file "$instr_all"; then
        if ! is_nonempty_file "$slope_csv"; then
             "$PYTHON_BIN" scripts/02_probe_slopes_from_logs.py \
              --base_json  "$base_all" \
              --instr_json "$instr_all" \
              --out_csv    "$slope_csv" \
              --pooling asst \
              --axis_mode pairwise
        fi
        
        echo "[Viz] 16_combined_metrics ($suffix)"
        "$PYTHON_BIN" scripts/16_visualize_combined_metrics.py \
            --internal_csv "$slope_csv" \
            --external_csv "$out_plots/${tag}_${suffix}_text_sensitivities.csv" \
            --out_dir "$out_plots" \
            --tag "${tag}_${suffix}"
    fi
}

run_experiment_set() {
    local set_name="$1"
    local prompt_file="$2"
    local template_type="$3"
    
    echo ">>>>>>> START PROMPT SET: $set_name (Layers ${LAYER_START}-${LAYER_END}) [Template: $template_type] <<<<<<<"
    
    for spec in "${MODEL_SPECS[@]}"; do
        IFS='|' read -r TAG BASE_ID INSTR_ID ALPHAS_BASE ALPHAS_INSTR <<< "$spec"
        
        # Suffix removed from dir name inside the root, since root has suffix
        # Old: exp/${TAG}/results_${set_name}${LAYER_SUFFIX}
        # New: "${EXP_OUTPUT_ROOT}/${TAG}/results_${set_name}"
        local results_dir="${EXP_OUTPUT_ROOT}/${TAG}/results_${set_name}"
        mkdir -p "$results_dir"
        
        echo "=== Model: $TAG | Set: $set_name | Layers: $LAYER_START-$LAYER_END ==="
        
        # Base (Input Axes still in exp/)
        local ax_base="exp/${TAG}/axes_base_asst_pairwise.npz"
        prepare_axes_if_needed "$BASE_ID" "$ax_base"
        for trait in "${TRAITS[@]}"; do
            run_probe_if_needed "base" "$trait" "$BASE_ID" "$ax_base" \
                "${results_dir}/${TAG}_base_${trait}_probe_results.jsonl" "$ALPHAS_BASE" "$prompt_file" \
                "$LAYER_START" "$LAYER_END" "$template_type"
            
            run_internal_analysis_if_needed "base" "$trait" "$BASE_ID" "$ax_base" \
                "${results_dir}/${TAG}_base_${trait}_internal_states.csv" "$ALPHAS_BASE" "$prompt_file" \
                "$LAYER_START" "$LAYER_END"
        done
        concat_alltraits "$TAG" "$results_dir" "base" "${results_dir}/${TAG}_base_alltraits.jsonl"
        run_text_analysis "$TAG" "$results_dir" "base"
        
        # Instruct
        local ax_instr="exp/${TAG}/axes_instruct_asst_pairwise.npz"
        prepare_axes_if_needed "$INSTR_ID" "$ax_instr"
        for trait in "${TRAITS[@]}"; do
             run_probe_if_needed "instruct" "$trait" "$INSTR_ID" "$ax_instr" \
                "${results_dir}/${TAG}_instruct_${trait}_probe_results.jsonl" "$ALPHAS_INSTR" "$prompt_file" \
                "$LAYER_START" "$LAYER_END" "$template_type"
             
             run_internal_analysis_if_needed "instruct" "$trait" "$INSTR_ID" "$ax_instr" \
                "${results_dir}/${TAG}_instruct_${trait}_internal_states.csv" "$ALPHAS_INSTR" "$prompt_file" \
                "$LAYER_START" "$LAYER_END"
        done
        concat_alltraits "$TAG" "$results_dir" "instruct" "${results_dir}/${TAG}_instruct_alltraits.jsonl"
        run_text_analysis "$TAG" "$results_dir" "instruct"
        
        # Visualization
        run_viz_for_dir "$TAG" "$results_dir" "$set_name"
    done
}

# ==================== MAIN ====================

for pset in "${PROMPT_SETS[@]}"; do
    IFS='|' read -r PNAME PFILE PTEMPLATE <<< "$pset"
    run_experiment_set "$PNAME" "$PFILE" "${PTEMPLATE:-standard}"
done

echo "=== GLOBAL ANALYSIS (THESIS PLOTS) FOR LAYERED EXPERIMENTS ==="
for pset in "${PROMPT_SETS[@]}"; do
    IFS='|' read -r PNAME PFILE <<< "$pset"
    OUT_DIR="analysis_results/thesis_plots_${PNAME}${LAYER_SUFFIX}"
    echo "[Thesis Viz] $PNAME (Saved to $OUT_DIR)"
    
    # Update globs to use EXP_OUTPUT_ROOT
    "$PYTHON_BIN" scripts/21_thesis_analysis_plots.py \
        --score_glob "${EXP_OUTPUT_ROOT}/*/results_${PNAME}/*_personality_scores.csv" \
        --metrics_glob "${EXP_OUTPUT_ROOT}/*/results_${PNAME}/*_text_metrics.csv" \
        --jsonl_glob "${EXP_OUTPUT_ROOT}/*/results_${PNAME}/*_probe_results.jsonl" \
        --out_dir "$OUT_DIR"
done

echo "=== DONE ==="