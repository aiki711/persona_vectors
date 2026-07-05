#!/bin/bash
#SBATCH --job-name=dlis_dyn_ng_neuroticism_a1.0
#SBATCH --output=/home/s2550009/persona_vectors/log/02_token_intensity_dynamic_no_gating/dlis_dyn_ng_neuroticism_a1.0.log
#SBATCH --error=/home/s2550009/persona_vectors/log/02_token_intensity_dynamic_no_gating/dlis_dyn_ng_neuroticism_a1.0.err
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
    --out_dir /home/s2550009/persona_vectors/exp_token_intensity/exp_symmetric/results \
    --axis neuroticism \
    --alpha_max 1.0 \
    --score_mode proj_rank \
    --theta_lo 0.0 \
    --theta_hi 99.0 \
    --k_lo 1.0 \
    --k_hi 1.0 \
    --gating_mode standard \
    --num_prompts 10 \
    --mask_bank vectors/soft_probe_masks.npz
