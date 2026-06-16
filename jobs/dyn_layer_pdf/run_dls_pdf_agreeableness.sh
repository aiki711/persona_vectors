#!/bin/bash
#SBATCH --job-name=dls_pdf_agreeableness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=06:00:00
#SBATCH --output=log/dls_pdf_agreeableness.out
#SBATCH --error=log/dls_pdf_agreeableness.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_pdf/results"
JUDGE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

echo "Starting text generation with PDF for agreeableness..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    # 1. Cos-Only DLS
    echo "=== Running Masked Cos-Only DLS alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/agreeableness/masked_cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --mask_bank "$MASK_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "cosine" \
            --no_prior
    fi

    # 2. Rank-Only DLS
    echo "=== Running Masked Rank-Only DLS alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/agreeableness/masked_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --mask_bank "$MASK_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "rank" \
            --no_prior
    fi

    # 3. Proj-Cos-Only DLS
    echo "=== Running Masked Proj-Cos-Only DLS alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/agreeableness/masked_proj_cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --mask_bank "$MASK_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_cosine" \
            --no_prior
    fi

    # 4. Proj-Rank-Only DLS
    echo "=== Running Masked Proj-Rank-Only DLS alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/agreeableness/masked_proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --mask_bank "$MASK_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank" \
            --no_prior
    fi

    # 5. Proj-Cos-Prior DLS
    echo "=== Running Masked Proj-Cos-Prior DLS alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/agreeableness/masked_proj_cos_prior_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --mask_bank "$MASK_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_cosine"
    fi

    # 6. Proj-Rank-Prior DLS
    echo "=== Running Masked Proj-Rank-Prior DLS alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/agreeableness/masked_proj_rank_prior_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --mask_bank "$MASK_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "agreeableness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank"
    fi
done

echo "=== Running Batch Evaluation (Llama-3-70B) ==="
"$PYTHON_BIN" scripts/04_dyn_layer/88_batch_eval_pdf.py --axis agreeableness --model "$JUDGE_MODEL"

echo "Completed DLS PDF run and evaluation for agreeableness."
