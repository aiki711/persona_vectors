#!/bin/bash
#SBATCH --job-name=gen_raw_fixed_seed_neuroticism
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_raw_fixed_seed_neuroticism.out
#SBATCH --error=log/gen_raw_fixed_seed_neuroticism.err

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

echo "Starting text generation on test prompts for neuroticism..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0; do
    # ------------------ 1. UNMASKED METHODS ------------------
    # 1.1 Logit-Diff
    echo "=== Running Logit-Diff alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/logit_diff_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "logit_diff" \
            --seed 42
    fi

    # 1.2 Cos-Only
    echo "=== Running Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "cosine" \
            --seed 42
    fi

    # 1.3 Rank-Only
    echo "=== Running Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "rank" \
            --seed 42
    fi

    # 1.4 Proj-Cos-Only
    echo "=== Running Proj-Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/proj_cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_cosine" \
            --seed 42
    fi

    # 1.5 Proj-Rank-Only
    echo "=== Running Proj-Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank" \
            --seed 42
    fi

    # ------------------ 2. PDF MASKED METHODS ------------------
    # 2.1 PDF Cos-Only
    echo "=== Running PDF Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/masked_cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "cosine" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi

    # 2.2 PDF Rank-Only
    echo "=== Running PDF Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/masked_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "rank" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi

    # 2.3 PDF Proj-Cos-Only
    echo "=== Running PDF Proj-Cos-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/masked_proj_cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_cosine" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi

    # 2.4 PDF Proj-Rank-Only
    echo "=== Running PDF Proj-Rank-Only alpha=$val ==="
    JSONL_OUT="${OUT_DIR}/neuroticism/masked_proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "neuroticism" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi
done

echo "Text generation completed on test prompts for neuroticism."
