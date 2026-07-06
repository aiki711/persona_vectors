#!/bin/bash
#SBATCH --job-name=eval_dlis_openness
#SBATCH --output=/home/s2550009/persona_vectors/log/02_token_intensity/eval_dlis_openness.log
#SBATCH --error=/home/s2550009/persona_vectors/log/02_token_intensity/eval_dlis_openness.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=2:00:00

cd /home/s2550009/persona_vectors
source persona_steering/bin/activate

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \
    --results_dir exp_token_intensity/exp_symmetric/results/openness \
    --axis openness \
    --quant 4bit
