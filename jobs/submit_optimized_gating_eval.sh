#!/bin/bash
#SBATCH --job-name=optimized_gating_eval
#SBATCH --output=log/optimized_gating_eval.log
#SBATCH --error=log/optimized_gating_eval.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Optimized Gating Evaluation SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

TRAITS=("extraversion" "neuroticism" "openness" "conscientiousness" "agreeableness")

# Step 1: Run generation for all 5 traits using our ideal gating curve parameters
for TRAIT in "${TRAITS[@]}"; do
    echo "=========================================="
    echo "Running generation for trait: ${TRAIT}"
    echo "=========================================="
    python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --mask_bank vectors/soft_probe_masks.npz \
        --out_dir exp_token_intensity/exp_sensitivity_analysis \
        --axis "${TRAIT}" \
        --alpha_max 5.0 \
        --gating_mode plateau \
        --static_layer \
        --theta_lo 2.0 \
        --theta_hi 7.0 \
        --k_lo 1.0 \
        --k_hi 4.0 \
        --num_prompts 10
done

# Step 2: Run Llama-3-70B judge evaluation on the generated files
for TRAIT in "${TRAITS[@]}"; do
    echo "=========================================="
    echo "Running judge evaluation for trait: ${TRAIT}"
    echo "=========================================="
    python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
        --results_dir exp_token_intensity/exp_sensitivity_analysis/"${TRAIT}" \
        --axis "${TRAIT}" \
        --quant 4bit
done

# Step 3: Run summary python script to print final aggregated score and PPL tables
python scratch/summarize_optimized_gating.py

echo "Job finished successfully!"
date
