#!/bin/bash
# 37_run_multi_layer_bidirectional.sh
# 
# Run adaptive steering on multiple layers simultaneously.
# Usage: bash scripts/02_base_steering/37_run_multi_layer_bidirectional.sh <trait> <layers>

set -e

# Activation
source persona_steering/bin/activate
export PYTHONPATH=$PYTHONPATH:src

# Configuration
TRAIT=$1
LAYERS=$2
CONFIG="config/mistral_7b.yaml"
BOUNDARY="exp_adaptive_steering/vectors/boundary_vectors.npz"
PROMPTS="exp_adaptive_steering/results/test_prompts_archive/test_prompts_10.jsonl"
OUT_DIR="exp_adaptive_steering/results"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

if [ -z "$TRAIT" ] || [ -z "$LAYERS" ]; then
    echo "Usage: $0 <trait> <layers>"
    exit 1
fi

echo "=================================================="
echo " Starting Multi-layer Bidirectional Steering Test "
echo " Trait : $TRAIT"
echo " Layers: $LAYERS "
echo "=================================================="

for DIRECTION in high low; do
    echo ">> Direction: $DIRECTION"
    
    # 1. Text Generation
    python3 scripts/02_base_steering/32_run_adaptive_steering.py \
        --config $CONFIG \
        --boundary_bank $BOUNDARY \
        --prompts $PROMPTS \
        --out_dir $OUT_DIR \
        --axis $TRAIT \
        --layer $LAYERS \
        --direction $DIRECTION \
        --tau 2.0 \
        --max_alpha 5.0 \
        --constant_alpha 5.0
        
    # 2. Evaluation
    LAYER_STR=$(echo $LAYERS | tr ',' '_')
    JSONL_FILE="${OUT_DIR}/adaptive_${TRAIT}_${DIRECTION}_L${LAYER_STR}.jsonl"
    CSV_FILE="${OUT_DIR}/scores_adaptive_${TRAIT}_${DIRECTION}_L${LAYER_STR}.csv"
    
    python3 scripts/02_base_steering/33_eval_adaptive_steering.py \
        --input $JSONL_FILE \
        --output $CSV_FILE \
        --axis $TRAIT \
        --model $JUDGE_MODEL
done

echo "Finished bidirectional test for $TRAIT (Layers: $LAYERS)."
