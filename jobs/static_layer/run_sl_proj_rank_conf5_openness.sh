#!/bin/bash
#SBATCH --job-name=sl_proj_rank_conf5_openness
#SBATCH --output=/home/s2550009/persona_vectors/log/static_layer/sl_proj_rank_conf5_openness.log
#SBATCH --error=/home/s2550009/persona_vectors/log/static_layer/sl_proj_rank_conf5_openness.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=0:30:00

cd /home/s2550009/persona_vectors
source persona_steering/bin/activate

# 1. Run generation with --static_layer
python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results \
    --axis openness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 8.0 \
    --k_hi 8.0 \
    --gating_mode standard \
    --num_prompts 10 \
    --static_layer 

# 2. Run judge evaluation immediately
python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --file /home/s2550009/persona_vectors/exp_token_intensity/exp_static_layer/results/openness/proj_rank_theta_3.0_7.0_k_8.0_8.0_Val5.0.jsonl \
    --axis openness \
    --quant 4bit
