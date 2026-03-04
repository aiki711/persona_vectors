#!/bin/bash
# 37_run_multi_layer_bidirectional.sh
# 
# Run adaptive steering on multiple layers simultaneously (e.g., L10, L15, L20),
# and test both positive (High) and negative (Low) directions.

set -e

# Configuration
CONFIG="config/mistral_7b.yaml"
BOUNDARY="exp_adaptive_steering/vectors/boundary_vectors.npz"
PROMPTS="exp_adaptive_steering/results/test_prompts_10.jsonl"
OUT_DIR="exp_adaptive_steering/results"
LAYERS="10,15,20" # Target layers
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

echo "=================================================="
echo " Starting Multi-layer Bidirectional Steering Test "
echo " Layers: $LAYERS "
echo "=================================================="

for TRAIT in extraversion neuroticism agreeableness conscientiousness openness; do
    echo "=================================================="
    echo " Processing Trait: $TRAIT"
    echo "=================================================="
    
    for DIRECTION in high low; do
        echo ">> Direction: $DIRECTION"
        
        # 1. Text Generation
        python scripts/32_run_adaptive_steering.py \
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
        # Replace commas with underscores for the filename
        LAYER_STR=$(echo $LAYERS | tr ',' '_')
        JSONL_FILE="${OUT_DIR}/adaptive_${TRAIT}_${DIRECTION}_L${LAYER_STR}.jsonl"
        CSV_FILE="${OUT_DIR}/scores_adaptive_${TRAIT}_${DIRECTION}_L${LAYER_STR}.csv"
        
        python scripts/33_eval_adaptive_steering.py \
            --input $JSONL_FILE \
            --output $CSV_FILE \
            --axis $TRAIT \
            --model $JUDGE_MODEL
    done
done

echo "Finished all multi-layer bidirectional traits."
