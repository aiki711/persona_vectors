#!/bin/bash
#SBATCH --job-name=anticipatory_gating
#SBATCH --output=log/anticipatory_gating.log
#SBATCH --error=log/anticipatory_gating.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -e

# Make sure log dir exists
mkdir -p log

echo "Starting Anticipatory Gating SLURM Job..."
date

# Activate virtual environment
source persona_steering/bin/activate

TRAITS=("extraversion" "neuroticism" "openness" "conscientiousness" "agreeableness")

# 1. P-Conf 3 (Resampled & Fixed)
echo "=========================================="
echo "Running P-Conf 3 (Fixed & Resampled)"
echo "=========================================="
for TRAIT in "${TRAITS[@]}"; do
    # Resampled
    python scratch/run_anticipatory_gating.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --mask_bank vectors/soft_probe_masks.npz \
        --out_dir exp_token_intensity/exp_anticipatory_gating \
        --axis "${TRAIT}" \
        --alpha_max 5.0 \
        --gating_mode plateau \
        --theta_lo 1.0 \
        --theta_hi 9.0 \
        --k_lo 2.0 \
        --k_hi 2.0 \
        --resample \
        --num_prompts 10

    # Fixed (no_resample)
    python scratch/run_anticipatory_gating.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --mask_bank vectors/soft_probe_masks.npz \
        --out_dir exp_token_intensity/exp_anticipatory_gating \
        --axis "${TRAIT}" \
        --alpha_max 5.0 \
        --gating_mode plateau \
        --theta_lo 1.0 \
        --theta_hi 9.0 \
        --k_lo 2.0 \
        --k_hi 2.0 \
        --no_resample \
        --num_prompts 10
done

# 2. P-Conf 6 (Resampled & Fixed)
echo "=========================================="
echo "Running P-Conf 6 (Fixed & Resampled)"
echo "=========================================="
for TRAIT in "${TRAITS[@]}"; do
    # Resampled
    python scratch/run_anticipatory_gating.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --mask_bank vectors/soft_probe_masks.npz \
        --out_dir exp_token_intensity/exp_anticipatory_gating \
        --axis "${TRAIT}" \
        --alpha_max 5.0 \
        --gating_mode plateau \
        --theta_lo 3.0 \
        --theta_hi 7.0 \
        --k_lo 0.5 \
        --k_hi 0.5 \
        --resample \
        --num_prompts 10

    # Fixed (no_resample)
    python scratch/run_anticipatory_gating.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --mask_bank vectors/soft_probe_masks.npz \
        --out_dir exp_token_intensity/exp_anticipatory_gating \
        --axis "${TRAIT}" \
        --alpha_max 5.0 \
        --gating_mode plateau \
        --theta_lo 3.0 \
        --theta_hi 7.0 \
        --k_lo 0.5 \
        --k_hi 0.5 \
        --no_resample \
        --num_prompts 10
done

# 3. A-Conf 3 (Resampled & Fixed)
echo "=========================================="
echo "Running A-Conf 3 (Fixed & Resampled)"
echo "=========================================="
for TRAIT in "${TRAITS[@]}"; do
    # Resampled
    python scratch/run_anticipatory_gating.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --mask_bank vectors/soft_probe_masks.npz \
        --out_dir exp_token_intensity/exp_anticipatory_gating \
        --axis "${TRAIT}" \
        --alpha_max 5.0 \
        --gating_mode plateau \
        --theta_lo 1.0 \
        --theta_hi 5.0 \
        --k_lo 1.0 \
        --k_hi 4.0 \
        --resample \
        --num_prompts 10

    # Fixed (no_resample)
    python scratch/run_anticipatory_gating.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --mask_bank vectors/soft_probe_masks.npz \
        --out_dir exp_token_intensity/exp_anticipatory_gating \
        --axis "${TRAIT}" \
        --alpha_max 5.0 \
        --gating_mode plateau \
        --theta_lo 1.0 \
        --theta_hi 5.0 \
        --k_lo 1.0 \
        --k_hi 4.0 \
        --no_resample \
        --num_prompts 10
done

# 4. Evaluation (batch_eval.py) for all traits and configs
echo "=========================================="
echo "Running evaluations..."
echo "=========================================="
for TRAIT in "${TRAITS[@]}"; do
    python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
        --results_dir exp_token_intensity/exp_anticipatory_gating/"${TRAIT}" \
        --axis "${TRAIT}" \
        --quant 4bit
done

echo "Anticipatory Gating experiments completed successfully!"
date
