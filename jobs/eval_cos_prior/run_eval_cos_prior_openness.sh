#!/bin/bash
#SBATCH --job-name=eval_cos_prior_openness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=08:00:00
#SBATCH --output=log/eval_cos_prior_openness.out
#SBATCH --error=log/eval_cos_prior_openness.err


WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting evaluation for Cos-Prior openness..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    JSONL_OUT="exp_steering_dyn_layer_proj_prior/results/openness/cos_prior_Val${val}.jsonl"
    CSV_OUT="exp_steering_dyn_layer_proj_prior/results/openness/scores_cos_prior_Val${val}.csv"
    
    if [ -f "$JSONL_OUT" ]; then
        if [ ! -f "$CSV_OUT" ]; then
            echo "Evaluating alpha=$val..."
            "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
                --input "$JSONL_OUT" \
                --output "$CSV_OUT" \
                --axis "openness" \
                --model "meta-llama/Meta-Llama-3-70B-Instruct" \
                --quant "4bit"
        else
            echo "Already evaluated: $CSV_OUT"
        fi
    else
        echo "Warning: input file not found: $JSONL_OUT"
    fi
done

echo "Evaluation completed for Cos-Prior openness."
