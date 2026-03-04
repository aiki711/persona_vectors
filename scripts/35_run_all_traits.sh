#!/bin/bash
# 35_run_all_traits.sh
# Runs the full Adaptive Steering pipeline for all 5 Big5 traits.

set -e

# Default parameters
LAYER=15
TAU=2.0
MAX_ALPHA=5.0
CONST_ALPHA=5.0
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

TRAITS=("extraversion" "neuroticism" "agreeableness" "conscientiousness" "openness")

echo "Starting Adaptive Steering Phase 3 Evaluation for all 5 traits..."

for TRAIT in "${TRAITS[@]}"; do
    echo "=================================================="
    echo "Processing Trait: ${TRAIT^^}"
    echo "=================================================="

    # 1. Train Boundary
    echo ">> [1/5] Training Boundary..."
    python scripts/30_train_boundary.py --config config/mistral_7b.yaml --out_dir exp_adaptive_steering/vectors --axis $TRAIT
    
    # 2. Visualize Boundary
    echo ">> [2/5] Visualizing Boundary..."
    python scripts/31_visualize_boundary.py --config config/mistral_7b.yaml --boundary_bank exp_adaptive_steering/vectors/boundary_vectors.npz --out_dir exp_adaptive_steering/figures --axis $TRAIT --layer $LAYER
    
    # 3. Run Adaptive Steering Text Generation
    echo ">> [3/5] Generating Text (Base, Constant, Adaptive)..."
    python scripts/32_run_adaptive_steering.py --config config/mistral_7b.yaml --boundary_bank exp_adaptive_steering/vectors/boundary_vectors.npz --prompts exp_adaptive_steering/results/test_prompts_10.jsonl --out_dir exp_adaptive_steering/results --axis $TRAIT --layer $LAYER --tau $TAU --max_alpha $MAX_ALPHA --constant_alpha $CONST_ALPHA
    
    # 4. Evaluate with LLM Judge
    echo ">> [4/5] Evaluating with LLM Judge..."
    python scripts/33_eval_adaptive_steering.py --input exp_adaptive_steering/results/adaptive_${TRAIT}_L${LAYER}.jsonl --output exp_adaptive_steering/results/scores_adaptive_${TRAIT}_L${LAYER}.csv --axis $TRAIT --model $JUDGE_MODEL
    
    # 5. Plot Trade-off
    echo ">> [5/5] Plotting Trade-off..."
    python scripts/34_plot_adaptive_tradeoff.py --input exp_adaptive_steering/results/scores_adaptive_${TRAIT}_L${LAYER}.csv --output exp_adaptive_steering/figures/tradeoff_${TRAIT}_L${LAYER}.png --axis $TRAIT
    
    echo "Finished processing ${TRAIT^^}."
    echo ""
done

echo "=================================================="
echo "All traits processed successfully! Phase 3 complete."
echo "=================================================="
