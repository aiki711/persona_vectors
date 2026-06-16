#!/bin/bash
#SBATCH --job-name=dls_masked_test
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:30:00
#SBATCH --output=log/test_masked.out
#SBATCH --error=log/test_masked.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Running DLS with probe masks (proj_cosine mode)..."
"$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
    --config config/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --mask_bank vectors/probe_masks.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --axis extraversion \
    --alpha 5.0 \
    --norm_mode raw_norm \
    --score_mode proj_cosine \
    --out_dir exp_steering_dyn_layer_proj_prior/results_test_masked

echo "Running DLS with probe masks (proj_rank mode)..."
"$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
    --config config/mistral_7b.yaml \
    --vector_bank vectors/mean_diff_vectors.npz \
    --mask_bank vectors/probe_masks.npz \
    --prompts inputs/eval_prompts_10.jsonl \
    --axis extraversion \
    --alpha 5.0 \
    --norm_mode raw_norm \
    --score_mode proj_rank \
    --out_dir exp_steering_dyn_layer_proj_prior/results_test_masked

echo "Done."
