#!/bin/bash
#SBATCH --job-name=dlis_conscientiousness_conf6_proj_rank
#SBATCH --output=/home/s2550009/persona_vectors/log/02_token_intensity/dlis_conscientiousness_conf6_proj_rank.log
#SBATCH --error=/home/s2550009/persona_vectors/log/02_token_intensity/dlis_conscientiousness_conf6_proj_rank.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=0:30:00

cd /home/s2550009/persona_vectors
source persona_steering/bin/activate

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/results \
    --axis conscientiousness \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 0.5 \
    --k_hi 0.5 \
    --num_prompts 10 \
    
