#!/bin/bash
#SBATCH --job-name=sl_pa_masked_proj_rank_conscientiousness
#SBATCH --output=/home/s2550009/persona_vectors/log/static_layer_plateau_asym/sl_pa_masked_proj_rank_conscientiousness.log
#SBATCH --error=/home/s2550009/persona_vectors/log/static_layer_plateau_asym/sl_pa_masked_proj_rank_conscientiousness.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=4:00:00

cd /home/s2550009/persona_vectors
source persona_steering/bin/activate

# Gating configuration: p_conf2 (theta: 3.0-7.0, k: 2.0-2.0, mode: plateau)
echo "=========================================="
echo "Starting configuration p_conf2..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 2.0 \
    --k_hi 2.0 \
    --gating_mode plateau \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_3.0_7.0_k_2.0_2.0_plateau_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: p_conf3 (theta: 1.0-9.0, k: 2.0-2.0, mode: plateau)
echo "=========================================="
echo "Starting configuration p_conf3..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 1.0 \
    --theta_hi 9.0 \
    --k_lo 2.0 \
    --k_hi 2.0 \
    --gating_mode plateau \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_1.0_9.0_k_2.0_2.0_plateau_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: p_conf4 (theta: 4.0-6.0, k: 2.0-2.0, mode: plateau)
echo "=========================================="
echo "Starting configuration p_conf4..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 4.0 \
    --theta_hi 6.0 \
    --k_lo 2.0 \
    --k_hi 2.0 \
    --gating_mode plateau \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_4.0_6.0_k_2.0_2.0_plateau_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: p_conf5 (theta: 3.0-7.0, k: 8.0-8.0, mode: plateau)
echo "=========================================="
echo "Starting configuration p_conf5..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 8.0 \
    --k_hi 8.0 \
    --gating_mode plateau \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_3.0_7.0_k_8.0_8.0_plateau_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: p_conf6 (theta: 3.0-7.0, k: 0.5-0.5, mode: plateau)
echo "=========================================="
echo "Starting configuration p_conf6..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 0.5 \
    --k_hi 0.5 \
    --gating_mode plateau \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_3.0_7.0_k_0.5_0.5_plateau_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: a_conf1 (theta: 3.0-7.0, k: 0.5-8.0, mode: max_normalized)
echo "=========================================="
echo "Starting configuration a_conf1..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 0.5 \
    --k_hi 8.0 \
    --gating_mode max_normalized \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_3.0_7.0_k_0.5_8.0_max_norm_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: a_conf2 (theta: 3.0-7.0, k: 8.0-0.5, mode: max_normalized)
echo "=========================================="
echo "Starting configuration a_conf2..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 8.0 \
    --k_hi 0.5 \
    --gating_mode max_normalized \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_3.0_7.0_k_8.0_0.5_max_norm_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: a_conf3 (theta: 1.0-5.0, k: 1.0-4.0, mode: max_normalized)
echo "=========================================="
echo "Starting configuration a_conf3..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 1.0 \
    --theta_hi 5.0 \
    --k_lo 1.0 \
    --k_hi 4.0 \
    --gating_mode max_normalized \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_1.0_5.0_k_1.0_4.0_max_norm_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit

# Gating configuration: a_conf4 (theta: 5.0-9.0, k: 4.0-1.0, mode: max_normalized)
echo "=========================================="
echo "Starting configuration a_conf4..."
echo "=========================================="

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 5.0 \
    --theta_hi 9.0 \
    --k_lo 4.0 \
    --k_hi 1.0 \
    --gating_mode max_normalized \
    --num_prompts 10 \
    --static_layer --mask_bank vectors/soft_probe_masks.npz

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer_plateau_asym/results/conscientiousness/masked_proj_rank_theta_5.0_9.0_k_4.0_1.0_max_norm_Val5.0.jsonl \
    --axis conscientiousness \
    --quant 4bit
