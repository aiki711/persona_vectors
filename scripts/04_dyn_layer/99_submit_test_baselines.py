#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 99_submit_test_baselines.py
#
# Submits evaluation jobs for Logit-Diff, Cos-Only, and Cos-Prior DLS on the new unseen test set.
#

import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_dls_test_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/eval_dls_test_{trait}.out
#SBATCH --error=log/eval_dls_test_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_proj_prior/results_test_unseen"
JUDGE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

echo "Starting evaluation on unseen test prompts for {trait}..."

for val in {vals_list}; do
    # 1. Logit-Diff DLS (Bhandari et al. baseline, no prior)
    echo "=== Running Logit-Diff DLS alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/logit_diff_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_logit_diff_Val${{val}}.csv"
    
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \\
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
            --no_prior
    fi
    
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 2. Cos-Only DLS (Cosine similarity, no prior)
    echo "=== Running Cos-Only DLS alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/cos_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_cos_only_Val${{val}}.csv"
    
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \\
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
            --no_prior
    fi
    
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 3. Cos-Prior DLS (Cosine similarity, with validation prior)
    echo "=== Running Cos-Prior DLS alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/cos_prior_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_cos_prior_Val${{val}}.csv"
    
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --input_dir "$INPUT_DIR" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$val" \\
            --direction "high" \\
            --norm_mode "raw_norm" \\
            --score_mode "cosine"
    fi
    
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 4. Rank-Only DLS (Ranking-based cosine, no prior)
    echo "=== Running Rank-Only DLS alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/rank_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_rank_only_Val${{val}}.csv"
    
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \\
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
            --no_prior
    fi
    
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 5. Proj-Rank-Only DLS (Projection ranking, no prior)
    echo "=== Running Proj-Rank-Only DLS alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/proj_rank_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_proj_rank_only_Val${{val}}.csv"
    
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \\
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
            --no_prior
    fi
    
    if [ -f "$JSONL_OUT" ] && [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL" \\
            --quant "4bit"
    fi

    # 6. Proj-Cos-Only DLS (Projection cosine alignment, no prior)
    echo "=== Running Proj-Cos-Only DLS alpha=$val ==="
    JSONL_OUT="${{OUT_DIR}}/{trait}/proj_cos_only_Val${{val}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_proj_cos_only_Val${{val}}.csv"
    
    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \\
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
            --no_prior
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

echo "Evaluation completed on unseen test prompts for {trait}."
"""

def main():
    job_dir = Path("jobs/eval_test_unseen")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(str(v) for v in VALS)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait, vals_list=vals_str)
        pbs_file = job_dir / f"run_eval_test_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        cmd = ["sbatch", str(pbs_file)]
        print(f"Submitting eval test job for {trait}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  {res.stdout.strip()} {res.stderr.strip()}")

if __name__ == "__main__":
    main()
