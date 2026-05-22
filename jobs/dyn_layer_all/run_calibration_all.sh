#!/bin/bash
#SBATCH --job-name=dls_calib_all
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=log/dls_calib_all.out
#SBATCH --error=log/dls_calib_all.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
OUT_STATS="exp_steering_dyn_layer_all_layers/dls_calibration_stats_all.json"

mkdir -p "exp_steering_dyn_layer_all_layers"

echo "Starting Calibration for all 32 layers..."
"$PYTHON_BIN" scripts/04_dyn_layer/64_calibrate_dls_stats.py \
    --config "$CONFIG" \
    --vector_bank "$VECTOR_BANK" \
    --prompts "$PROMPT_IN" \
    --out_file "$OUT_STATS" \
    --num_prompts 50 \
    --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
