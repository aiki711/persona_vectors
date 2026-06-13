#!/bin/bash
#SBATCH --job-name=dls_rank_only_openness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/dls_rank_only_openness.out
#SBATCH --error=log/dls_rank_only_openness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_proj_prior/results"

echo "Running rank-Only DLS sweep (midpoint logic) for openness..."

# Loop over values and run the script with --score_mode rank and --no_prior
for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    echo "=== Running alpha=$val ==="
    "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py         --config "$CONFIG"         --vector_bank "$VECTOR_BANK"         --prompts "$PROMPT_IN"         --input_dir "$INPUT_DIR"         --out_dir "$OUT_DIR"         --axis "openness"         --alpha "$val"         --direction "high"         --norm_mode "raw_norm"         --score_mode "rank"         --no_prior
done
