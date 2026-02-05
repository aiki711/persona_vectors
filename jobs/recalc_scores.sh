#!/bin/bash
#PBS -N recalc_scores
#PBS -q GPU-1
#PBS -o log/recalc_scores.o%j
#PBS -e log/recalc_scores.e%j
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
cd "$WORKDIR"

# Project Setup
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

echo "=== STARTING SCORE RECALCULATION ==="
echo "Model: KevSun/Personality_LM"
echo "Target: exp/**/*_alltraits.jsonl"

# Find all target jsonl files
# output example: exp/mistral_7b/results_mtbench/mistral_7b_base_alltraits.jsonl
find exp -name "*_alltraits.jsonl" | sort | while read -r input_jsonl; do
    echo "Processing: $input_jsonl"
    
    # Define output csv path (replace .jsonl with _personality_scores.csv)
    # But adhere to the naming convention used in other scripts if possible.
    # Usually: tag_split_personality_scores.csv
    # The input is usually: tag_split_alltraits.jsonl
    
    dir_name=$(dirname "$input_jsonl")
    base_name=$(basename "$input_jsonl" .jsonl)
    
    # base_name is like: mistral_7b_base_alltraits
    # we want output: mistral_7b_base_personality_scores.csv
    
    # Replace 'alltraits' with 'personality_scores'
    out_name="${base_name/alltraits/personality_scores}.csv"
    output_csv="$dir_name/$out_name"
    
    echo "  -> Output: $output_csv"
    
    "$PYTHON_BIN" scripts/14_calc_personality_score.py "$input_jsonl" \
      --output "$output_csv" \
      --batch_size 32 \
      --model "KevSun/Personality_LM"
      
    echo "  [Done]"
done

echo "=== RECALCULATION COMPLETED ==="
