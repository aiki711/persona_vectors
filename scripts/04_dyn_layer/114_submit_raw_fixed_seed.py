#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 114_submit_raw_fixed_seed.py
#
# Submits evaluation jobs for all 9 dynamic layer steering methods (5 unmasked, 4 PDF-masked)
# on the test set using a fixed seed (42) and raw-norm scaling (raw difference vector norm).
#
# Output: exp_steering_dyn_layer_raw/results/{trait}/{method}_Val{alpha}.jsonl
#         exp_steering_dyn_layer_raw/results/{trait}/scores_{method}_Val{alpha}.csv
#

import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_raw_fixed_seed_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/eval_raw_fixed_seed_{trait}.out
#SBATCH --error=log/eval_raw_fixed_seed_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_raw/results"
JUDGE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

echo "Starting evaluation on test prompts for {trait}..."

for val in {vals_list}; do
    # ------------------ 1. UNMASKED METHODS ------------------
    # 1.1 Logit-Diff
    echo "=== Running Logit-Diff alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/logit_diff_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_logit_diff_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "logit_diff" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 1.2 Cos-Only
    echo "=== Running Cos-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/cos_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_cos_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "cosine" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 1.3 Rank-Only
    echo "=== Running Rank-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/rank_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_rank_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "rank" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 1.4 Proj-Cos-Only
    echo "=== Running Proj-Cos-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/proj_cos_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_proj_cos_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "proj_cosine" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 1.5 Proj-Rank-Only
    echo "=== Running Proj-Rank-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/proj_rank_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_proj_rank_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "proj_rank" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # ------------------ 2. PDF MASKED METHODS ------------------
    # 2.1 PDF Cos-Only
    echo "=== Running PDF Cos-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_cos_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_masked_cos_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "cosine" \\
            --mask_bank "$MASK_BANK" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 2.2 PDF Rank-Only
    echo "=== Running PDF Rank-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_rank_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_masked_rank_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "rank" \\
            --mask_bank "$MASK_BANK" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 2.3 PDF Proj-Cos-Only
    echo "=== Running PDF Proj-Cos-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_proj_cos_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_masked_proj_cos_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "proj_cosine" \\
            --mask_bank "$MASK_BANK" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 2.4 PDF Proj-Rank-Only
    echo "=== Running PDF Proj-Rank-Only alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_proj_rank_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_masked_proj_rank_only_Val${{val}}.csv"
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "proj_rank" \\
            --mask_bank "$MASK_BANK" \\
            --seed 42
    fi
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi
done

echo "Evaluation completed on test prompts for {trait}."
"""

def main():
    job_dir = Path("jobs/eval_raw_fixed_seed")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(str(v) for v in VALS)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait, vals_list=vals_str)
        pbs_file = job_dir / f"run_eval_{trait}.sh"
        with open(pbs_file, "w", encoding="utf-8") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        cmd = ["sbatch", str(pbs_file)]
        print(f"Submitting job for {trait}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  Stdout: {res.stdout.strip()}")
        print(f"  Stderr: {res.stderr.strip()}")

if __name__ == "__main__":
    main()
