#!/bin/bash
#SBATCH --job-name=sl_pa_c_masked_proj_cosine_agreeableness
#SBATCH --output=/home/s2550009/persona_vectors/log/static_layer_plateau_asym_custom/sl_pa_c_masked_proj_cosine_agreeableness.log
#SBATCH --error=/home/s2550009/persona_vectors/log/static_layer_plateau_asym_custom/sl_pa_c_masked_proj_cosine_agreeableness.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=2:00:00

cd /home/s2550009/persona_vectors
source persona_steering/bin/activate

# Gating configuration: a_conf5 (theta: 2.0-7.0, k: 1.0-4.0, mode: max_normalized)
echo "=========================================="
echo "Starting configuration a_conf5..."
echo "=========================================="

if [ ! -f "/home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_2.0_7.0_k_1.0_4.0_max_norm_Val5.0.jsonl" ]; then
    python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results \
        --axis agreeableness \
        --alpha_max 5.0 \
        --score_mode proj_cosine \
        --theta_lo 2.0 \
        --theta_hi 7.0 \
        --k_lo 1.0 \
        --k_hi 4.0 \
        --gating_mode max_normalized \
        --num_prompts 10 \
        --static_layer --mask_bank vectors/soft_probe_masks.npz
else
    echo "Output JSONL file already exists. Skipping generation."
fi

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_2.0_7.0_k_1.0_4.0_max_norm_Val5.0.jsonl \
    --axis agreeableness \
    --quant 4bit

# Gating configuration: a_conf6 (theta: 1.0-4.0, k: 0.5-6.0, mode: max_normalized)
echo "=========================================="
echo "Starting configuration a_conf6..."
echo "=========================================="

if [ ! -f "/home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_1.0_4.0_k_0.5_6.0_max_norm_Val5.0.jsonl" ]; then
    python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results \
        --axis agreeableness \
        --alpha_max 5.0 \
        --score_mode proj_cosine \
        --theta_lo 1.0 \
        --theta_hi 4.0 \
        --k_lo 0.5 \
        --k_hi 6.0 \
        --gating_mode max_normalized \
        --num_prompts 10 \
        --static_layer --mask_bank vectors/soft_probe_masks.npz
else
    echo "Output JSONL file already exists. Skipping generation."
fi

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_1.0_4.0_k_0.5_6.0_max_norm_Val5.0.jsonl \
    --axis agreeableness \
    --quant 4bit

# Gating configuration: a_conf7 (theta: 2.0-8.0, k: 0.8-5.0, mode: max_normalized)
echo "=========================================="
echo "Starting configuration a_conf7..."
echo "=========================================="

if [ ! -f "/home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_2.0_8.0_k_0.8_5.0_max_norm_Val5.0.jsonl" ]; then
    python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results \
        --axis agreeableness \
        --alpha_max 5.0 \
        --score_mode proj_cosine \
        --theta_lo 2.0 \
        --theta_hi 8.0 \
        --k_lo 0.8 \
        --k_hi 5.0 \
        --gating_mode max_normalized \
        --num_prompts 10 \
        --static_layer --mask_bank vectors/soft_probe_masks.npz
else
    echo "Output JSONL file already exists. Skipping generation."
fi

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_2.0_8.0_k_0.8_5.0_max_norm_Val5.0.jsonl \
    --axis agreeableness \
    --quant 4bit

# Gating configuration: a_conf8 (theta: 2.0-6.0, k: 0.5-8.0, mode: plateau)
echo "=========================================="
echo "Starting configuration a_conf8..."
echo "=========================================="

if [ ! -f "/home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_2.0_6.0_k_0.5_8.0_plateau_Val5.0.jsonl" ]; then
    python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
        --config configs/mistral_7b.yaml \
        --vector_bank vectors/mean_diff_vectors.npz \
        --prompts inputs/eval_prompts_10.jsonl \
        --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results \
        --axis agreeableness \
        --alpha_max 5.0 \
        --score_mode proj_cosine \
        --theta_lo 2.0 \
        --theta_hi 6.0 \
        --k_lo 0.5 \
        --k_hi 8.0 \
        --gating_mode plateau \
        --num_prompts 10 \
        --static_layer --mask_bank vectors/soft_probe_masks.npz
else
    echo "Output JSONL file already exists. Skipping generation."
fi

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym_custom/results/agreeableness/masked_proj_cosine_theta_2.0_6.0_k_0.5_8.0_plateau_Val5.0.jsonl \
    --axis agreeableness \
    --quant 4bit
