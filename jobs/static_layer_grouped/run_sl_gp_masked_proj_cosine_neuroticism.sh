#!/bin/bash
#SBATCH --job-name=sl_gp_masked_proj_cosine_neuroticism
#SBATCH --output=/home/s2550009/persona_vectors/log/static_layer_grouped/sl_gp_masked_proj_cosine_neuroticism.log
#SBATCH --error=/home/s2550009/persona_vectors/log/static_layer_grouped/sl_gp_masked_proj_cosine_neuroticism.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=3:00:00

cd /home/s2550009/persona_vectors
source persona_steering/bin/activate

# Gating configuration: conf1 (theta: 0.0-99.0, k: 1.0-1.0)
echo "=========================================="
echo "Starting configuration conf1..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results \
    --axis neuroticism \
    --alpha_max 5.0 \
    --score_mode proj_cosine \
    --theta_lo 0.0 \
    --theta_hi 99.0 \
    --k_lo 1.0 \
    --k_hi 1.0 \
    --gating_mode standard \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results/neuroticism/masked_proj_cosine_theta_0.0_99.0_k_1.0_1.0_Val5.0.jsonl \
    --axis neuroticism \
    --quant 4bit

# Gating configuration: conf2 (theta: 3.0-7.0, k: 2.0-2.0)
echo "=========================================="
echo "Starting configuration conf2..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results \
    --axis neuroticism \
    --alpha_max 5.0 \
    --score_mode proj_cosine \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 2.0 \
    --k_hi 2.0 \
    --gating_mode standard \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results/neuroticism/masked_proj_cosine_theta_3.0_7.0_k_2.0_2.0_Val5.0.jsonl \
    --axis neuroticism \
    --quant 4bit

# Gating configuration: conf3 (theta: 1.0-9.0, k: 2.0-2.0)
echo "=========================================="
echo "Starting configuration conf3..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results \
    --axis neuroticism \
    --alpha_max 5.0 \
    --score_mode proj_cosine \
    --theta_lo 1.0 \
    --theta_hi 9.0 \
    --k_lo 2.0 \
    --k_hi 2.0 \
    --gating_mode standard \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results/neuroticism/masked_proj_cosine_theta_1.0_9.0_k_2.0_2.0_Val5.0.jsonl \
    --axis neuroticism \
    --quant 4bit

# Gating configuration: conf4 (theta: 4.0-6.0, k: 2.0-2.0)
echo "=========================================="
echo "Starting configuration conf4..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results \
    --axis neuroticism \
    --alpha_max 5.0 \
    --score_mode proj_cosine \
    --theta_lo 4.0 \
    --theta_hi 6.0 \
    --k_lo 2.0 \
    --k_hi 2.0 \
    --gating_mode standard \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results/neuroticism/masked_proj_cosine_theta_4.0_6.0_k_2.0_2.0_Val5.0.jsonl \
    --axis neuroticism \
    --quant 4bit

# Gating configuration: conf5 (theta: 3.0-7.0, k: 8.0-8.0)
echo "=========================================="
echo "Starting configuration conf5..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results \
    --axis neuroticism \
    --alpha_max 5.0 \
    --score_mode proj_cosine \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 8.0 \
    --k_hi 8.0 \
    --gating_mode standard \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results/neuroticism/masked_proj_cosine_theta_3.0_7.0_k_8.0_8.0_Val5.0.jsonl \
    --axis neuroticism \
    --quant 4bit

# Gating configuration: conf6 (theta: 3.0-7.0, k: 0.5-0.5)
echo "=========================================="
echo "Starting configuration conf6..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results \
    --axis neuroticism \
    --alpha_max 5.0 \
    --score_mode proj_cosine \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 0.5 \
    --k_hi 0.5 \
    --gating_mode standard \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results/neuroticism/masked_proj_cosine_theta_3.0_7.0_k_0.5_0.5_Val5.0.jsonl \
    --axis neuroticism \
    --quant 4bit
