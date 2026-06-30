#!/bin/bash
#SBATCH --job-name=gen_sim_rank_extraversion
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/gen_sim_rank_extraversion.out
#SBATCH --error=log/gen_sim_rank_extraversion.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/soft_probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_raw/results"

echo "Starting Similarity-based Rank/Proj-Rank sweeps for extraversion..."

for val in 0.5 1.0 2.0 4.0 5.0 6.0 8.0 10.0 15.0 20.0; do
    # 1. Rank-Only (unmasked)
    JSONL_OUT="${OUT_DIR}/extraversion/rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running Rank-Only alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "rank" \
            --seed 42
    fi

    # 2. PDF Rank-Only (masked)
    JSONL_OUT="${OUT_DIR}/extraversion/masked_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Rank-Only alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "rank" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi

    # 3. Proj Rank-Only (unmasked)
    JSONL_OUT="${OUT_DIR}/extraversion/proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running Proj Rank-Only alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank" \
            --seed 42
    fi

    # 4. PDF Proj Rank-Only (masked)
    JSONL_OUT="${OUT_DIR}/extraversion/masked_proj_rank_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Proj Rank-Only alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_rank" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi

    # 5. PDF Cos-Only (masked)
    JSONL_OUT="${OUT_DIR}/extraversion/masked_cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Cos-Only alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "cosine" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi

    # 6. PDF Proj Cos-Only (masked)
    JSONL_OUT="${OUT_DIR}/extraversion/masked_proj_cos_only_Val${val}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Proj Cos-Only alpha=$val ==="
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \
            --config "$CONFIG" \
            --vector_bank "$VECTOR_BANK" \
            --prompts "$PROMPT_IN" \
            --input_dir "$INPUT_DIR" \
            --out_dir "$OUT_DIR" \
            --axis "extraversion" \
            --alpha "$val" \
            --direction "high" \
            --norm_mode "raw_norm" \
            --score_mode "proj_cosine" \
            --mask_bank "$MASK_BANK" \
            --seed 42
    fi
done

echo "Similarity-based Rank/Proj-Rank sweeps completed for extraversion."
