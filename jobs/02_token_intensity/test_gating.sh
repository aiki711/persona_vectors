#!/bin/bash
#SBATCH --job-name=test_dlis
#SBATCH --output=log/02_token_intensity/test_dlis.log
#SBATCH --error=log/02_token_intensity/test_dlis.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=0:10:00

cd /home/s2550009/persona_vectors
source persona_steering/bin/activate

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \
    --config configs/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --out_dir exp_token_intensity/results_test \
    --axis extraversion \
    --alpha_max 5.0 \
    --score_mode proj_rank \
    --theta_lo 3.0 \
    --theta_hi 7.0 \
    --k_lo 2.0 \
    --k_hi 2.0 \
    --num_prompts 1
