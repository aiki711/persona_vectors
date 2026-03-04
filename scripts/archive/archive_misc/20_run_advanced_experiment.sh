#!/bin/bash
# 20_run_advanced_experiment.sh
# 複合実験：KLスケーリング、多軸直交化、レイヤー比較

set -e

# Config
BASE_CONFIG="exp/configs/verification_subspace.yaml"
VECTORS="exp/vectors/mistral_subspace.npz"
SENSITIVITY="exp/sensitivity/mistral_subspace.json"
ACTIVATE="/home/admin/work/s2550009/persona_vectors/persona_steering/bin/activate"

source $ACTIVATE
export PYTHONPATH=$PYTHONPATH:/home/admin/work/s2550009/persona_vectors/src

echo "=== Phase 1: Sensitivity Scan (Required for KL Scaling) ==="
if [ ! -f "$SENSITIVITY" ]; then
    python scripts/19_layer_sensitivity_scan.py \
        --config $BASE_CONFIG \
        --vectors $VECTORS \
        --output $SENSITIVITY \
        --alpha 5.0 --limit 5
fi

echo "=== Phase 2: Comparative Steering Experiments ==="

# 1. Golden Layer + KL Scaling
echo "Running Strategy: Golden Layer + KL Scaling"
python scripts/01_run_probe.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --axes_bank $VECTORS \
    --trait extraversion \
    --alpha_list="-5.0,0.0,5.0" \
    --dynamic_layer_json $SENSITIVITY \
    --alpha_scale \
    --calc_ppl \
    --out exp/adv_extra_golden_scaled.jsonl \
    --samples 5

# 2. Multi-Trait Steering (Extraversion + Agreeableness)
# Note: This will perform Orthogonalization in live_axes.py
echo "Running Strategy: Multi-Trait (Extraversion + Agreeableness)"
python scripts/01_run_probe.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --axes_bank $VECTORS \
    --trait extraversion \
    --traits="extraversion,agreeableness" \
    --alphas="3.0,3.0" \
    --alpha_list="0.0,5.0" \
    --dynamic_layer_json $SENSITIVITY \
    --alpha_scale \
    --calc_ppl \
    --out exp/adv_multi_trait.jsonl \
    --samples 5

echo "=== Phase 3: Final LLM Scoring ==="
python scripts/14_calc_personality_score_llm.py exp/adv_extra_golden_scaled.jsonl --output exp/adv_extra_golden_scaled_scores.csv
python scripts/14_calc_personality_score_llm.py exp/adv_multi_trait.jsonl --output exp/adv_multi_trait_scores.csv

echo "=== Experiment Complete ==="
