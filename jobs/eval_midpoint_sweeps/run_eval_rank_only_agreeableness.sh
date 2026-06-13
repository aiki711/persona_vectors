#!/bin/bash
#SBATCH --job-name=eval_rank_only_agreeableness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=08:00:00
#SBATCH --output=log/eval_rank_only_agreeableness.out
#SBATCH --error=log/eval_rank_only_agreeableness.err
#SBATCH --dependency=afterok:204682

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting evaluation for rank-Only agreeableness..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    JSONL_OUT="exp_steering_dyn_layer_proj_prior/results/agreeableness/rank_only_Val${val}.jsonl"
    CSV_OUT="exp_steering_dyn_layer_proj_prior/results/agreeableness/scores_rank_only_Val${val}.csv"
    
    # Check both float version and integer version for paths
    if [ ! -f "$JSONL_OUT" ]; then
        # Check integer format fallback (e.g. Val5 instead of Val5.0)
        JSONL_OUT_INT="exp_steering_dyn_layer_proj_prior/results/agreeableness/rank_only_Val${val}.jsonl"
        if [ -f "$JSONL_OUT_INT" ]; then
            JSONL_OUT="$JSONL_OUT_INT"
        fi
    fi

    if [ -f "$JSONL_OUT" ]; then
        # We overwrite existing scores since the underlying steering layer logic has changed!
        echo "Evaluating alpha=${val}..."
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "agreeableness" \
            --model "meta-llama/Meta-Llama-3-70B-Instruct" \
            --quant "4bit"
    else
        echo "Warning: input file not found: $JSONL_OUT"
    fi
done

echo "Evaluation completed for rank-Only agreeableness."
