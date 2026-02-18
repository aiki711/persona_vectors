#!/bin/bash
# 21_run_big5_experiment.sh
# Big5全軸拡張 ＆ 複数層介入戦略（Golden vs Top-5）の比較実験

set -e

# Config
BASE_CONFIG="exp/configs/verification_subspace.yaml"
VECTORS="exp/vectors/mistral_subspace_full.npz"
SENSITIVITY="exp/sensitivity/mistral_subspace_full.json"
ACTIVATE="/home/admin/work/s2550009/persona_vectors/persona_steering/bin/activate"

source $ACTIVATE
export PYTHONPATH=/home/admin/work/s2550009/persona_vectors/src:$PYTHONPATH

echo "=== Phase 1: Sensitivity Scan for all Big5 Traits ==="
# Note: 00_prepare_vectors_subspace.py must have finished by now
if [ ! -f "$VECTORS" ]; then
    echo "ERROR: Full vectors not found. Please wait for the prep script to finish."
    exit 1
fi

python scripts/19_layer_sensitivity_scan.py \
    --config $BASE_CONFIG \
    --vectors $VECTORS \
    --output $SENSITIVITY \
    --alpha 5.0 --limit 5

echo "=== Phase 2: Comparative Layer Strategy Experiments ==="
# Trait: extraversion (for direct comparison with previous results)
TRAIT="extraversion"
ALPHAS="-2.0,0.0,2.0"

# 1. Golden Layer Strategy (Baseline for this phase)
echo "Running Strategy: Golden Layer (Single Layer)"
python scripts/01_run_probe.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --axes_bank $VECTORS \
    --trait $TRAIT \
    --alpha_list="$ALPHAS" \
    --dynamic_layer_json $SENSITIVITY \
    --layer_strategy golden \
    --alpha_scale \
    --calc_ppl \
    --out exp/big5_extra_golden.jsonl \
    --samples 10

# 2. Top-5 Layers Strategy (Additive Mid-Layer Intervention)
echo "Running Strategy: Top-5 Layers (Additive L10-25)"
python scripts/01_run_probe.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --axes_bank $VECTORS \
    --trait $TRAIT \
    --alpha_list="$ALPHAS" \
    --dynamic_layer_json $SENSITIVITY \
    --layer_strategy top5 \
    --alpha_mode additive \
    --layer_start 10 --layer_end 25 \
    --alpha_scale \
    --calc_ppl \
    --out exp/big5_extra_top5.jsonl \
    --samples 10

echo "=== Phase 3: Multi-Trait Simultaneous Steering (e.g., E+ / A+) ==="
# Mix Extraversion and Agreeableness
echo "Running Strategy: Multi-Trait (E+1.5 & A+1.5)"
python scripts/01_run_probe.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --axes_bank $VECTORS \
    --trait extraversion \
    --traits "extraversion,agreeableness" \
    --alphas "1.5,1.5" \
    --alpha_list="0.0,3.0" \
    --dynamic_layer_json $SENSITIVITY \
    --layer_strategy top5 \
    --alpha_mode additive \
    --layer_start 10 --layer_end 25 \
    --alpha_scale \
    --calc_ppl \
    --out exp/big5_multi_trait_mixed.jsonl \
    --samples 10

echo "=== Phase 4: LLM Scoring and Analysis ==="
python scripts/14_calc_personality_score_llm.py exp/big5_extra_golden.jsonl --output exp/big5_extra_golden_scores.csv
python scripts/14_calc_personality_score_llm.py exp/big5_extra_top5.jsonl --output exp/big5_extra_top5_scores.csv

echo "=== Experiment Ready for Evaluation ==="
