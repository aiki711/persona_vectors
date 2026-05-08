#!/bin/bash
# 38_run_trait_specific_steering.sh
# Runs specific steering for each trait with its optimized layer set.

echo "=================================================="
echo " Starting Trait-Specific Optimized Steering Test "
echo "=================================================="

# Selection based on vocabulary projection analysis
# Format: trait "layers"
declare -A TRAIT_LAYERS
TRAIT_LAYERS=(
    ["extraversion"]="14,16,18,20,22"
    ["neuroticism"]="10,12,14,16,18"
    ["agreeableness"]="18,20,22,24,26"
    ["conscientiousness"]="12,14,16,18,20"
    ["openness"]="15,18,21,24,27"
)

TRAITS=("extraversion" "neuroticism" "agreeableness" "conscientiousness" "openness")

for trait in "${TRAITS[@]}"; do
    layers=${TRAIT_LAYERS[$trait]}
    echo "=================================================="
    echo " Processing Trait: $trait (Layers: $layers)"
    echo "=================================================="
    
    # Run bidirectional (High and Low)
    bash scripts/37_run_multi_layer_bidirectional.sh "$trait" "$layers"
    
    # Move results to target folder
    mkdir -p exp_adaptive_steering/results/phase5_trait_specific_optimized
    layer_str=$(echo $layers | tr ',' '_')
    mv exp_adaptive_steering/results/*_${trait}_*_${layer_str}.* exp_adaptive_steering/results/phase5_trait_specific_optimized/ 2>/dev/null || true
    mv exp_adaptive_steering/results/scores_*_${trait}_*_${layer_str}.* exp_adaptive_steering/results/phase5_trait_specific_optimized/ 2>/dev/null || true
done

echo "Finished all trait-specific optimized steering tests."
