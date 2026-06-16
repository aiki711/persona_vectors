#!/bin/bash
#SBATCH --job-name=dls_norm_conscientiousness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/dls_norm_conscientiousness.out
#SBATCH --error=log/dls_norm_conscientiousness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_norm/results"
JUDGE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

echo "Starting text generation with norm DLS for conscientiousness..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    # 1. Cos-Only
    JSONL_OUT="${OUT_DIR}/conscientiousness/cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "cosine" \
            --no_prior
    fi

    # 2. Rank-Only
    JSONL_OUT="${OUT_DIR}/conscientiousness/rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "rank" \
            --no_prior
    fi

    # 3. Logit-Diff
    JSONL_OUT="${OUT_DIR}/conscientiousness/logit_diff_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "logit_diff" \
            --no_prior
    fi
done

echo "=== Running Batch Evaluation ==="
"$PYTHON_BIN" scripts/04_dyn_layer/88_batch_eval_pdf.py --axis conscientiousness --model "$JUDGE_MODEL" --results_dir "$OUT_DIR" --methods cos_only rank_only logit_diff

echo "Completed DLS norm run and evaluation for conscientiousness."
