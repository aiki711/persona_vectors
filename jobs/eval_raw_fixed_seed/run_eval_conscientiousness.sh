#!/bin/bash
#SBATCH --job-name=eval_raw_fixed_seed_conscientiousness
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/eval_raw_fixed_seed_conscientiousness.out
#SBATCH --error=log/eval_raw_fixed_seed_conscientiousness.err

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
OUT_DIR="exp_steering_dyn_layer_raw/results"
JUDGE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

echo "Starting evaluation on test prompts for conscientiousness..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    # ------------------ 1. UNMASKED METHODS ------------------
    # 1.1 Logit-Diff
    echo "=== Running Logit-Diff alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/logit_diff_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_logit_diff_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
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
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # 1.2 Cos-Only
    echo "=== Running Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/cos_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_cos_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
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
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # 1.3 Rank-Only
    echo "=== Running Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/rank_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_rank_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
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
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # 1.4 Proj-Cos-Only
    echo "=== Running Proj-Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/proj_cos_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_proj_cos_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_cosine" \
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # 1.5 Proj-Rank-Only
    echo "=== Running Proj-Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/proj_rank_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_proj_rank_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank" \
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # ------------------ 2. PDF MASKED METHODS ------------------
    # 2.1 PDF Cos-Only
    echo "=== Running PDF Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/masked_cos_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_masked_cos_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
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
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # 2.2 PDF Rank-Only
    echo "=== Running PDF Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/masked_rank_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_masked_rank_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
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
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # 2.3 PDF Proj-Cos-Only
    echo "=== Running PDF Proj-Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/masked_proj_cos_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_masked_proj_cos_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_cosine" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi

    # 2.4 PDF Proj-Rank-Only
    echo "=== Running PDF Proj-Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/conscientiousness/masked_proj_rank_only_Val${val}.jsonl"
    CSV_OUT="${OUT_DIR}/conscientiousness/scores_masked_proj_rank_only_Val${val}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "conscientiousness" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \
            --input "$JSONL_OUT" \
            --output "$CSV_OUT" \
            --axis "conscientiousness" \
            --model "$JUDGE_MODEL" \
            --quant "4bit"
    fi
done

echo "Evaluation completed on test prompts for conscientiousness."
