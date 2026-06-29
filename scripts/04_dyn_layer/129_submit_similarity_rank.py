#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/129_submit_similarity_rank.py
#
# Submits text generation sweeps and Llama-3 evaluations for the new similarity-based Rank and Proj-Rank methods.
# Cleans up old results first to avoid skip triggers.
# Uses SLURM dependencies to chain evaluation after generation completes.
#

import subprocess
import os
import sys
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]

GEN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=gen_sim_rank_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/gen_sim_rank_{trait}.out
#SBATCH --error=log/gen_sim_rank_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/soft_probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_raw/results"

echo "Starting Similarity-based Rank/Proj-Rank sweeps for {trait}..."

for val in {vals_list}; do
    # 1. Rank-Only (unmasked)
    JSONL_OUT="${{OUT_DIR}}/{trait}/rank_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running Rank-Only alpha=$val ==="
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

    # 2. PDF Rank-Only (masked)
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_rank_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Rank-Only alpha=$val ==="
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

    # 3. Proj Rank-Only (unmasked)
    JSONL_OUT="${{OUT_DIR}}/{trait}/proj_rank_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running Proj Rank-Only alpha=$val ==="
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

    # 4. PDF Proj Rank-Only (masked)
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_proj_rank_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Proj Rank-Only alpha=$val ==="
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

    # 5. PDF Cos-Only (masked)
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_cos_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Cos-Only alpha=$val ==="
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

    # 6. PDF Proj Cos-Only (masked)
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_proj_cos_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Proj Cos-Only alpha=$val ==="
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
done

echo "Similarity-based Rank/Proj-Rank sweeps completed for {trait}."
"""

EVAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_sim_rank_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=02:00:00
#SBATCH --output=log/eval_sim_rank_{trait}.out
#SBATCH --error=log/eval_sim_rank_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting evaluation of Similarity-based Rank/Proj-Rank results for {trait}..."

"$PYTHON_BIN" scripts/04_dyn_layer/115_batch_eval.py \\
    --results_dir "exp_steering_dyn_layer_raw/results/{trait}" \\
    --axis "{trait}" \\
    --quant "4bit"

echo "Evaluation completed for {trait}."
"""

def submit_job(script_path):
    cmd = ["sbatch", str(script_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error submitting {script_path}: {res.stderr}")
        return None
    stdout = res.stdout.strip()
    print(f"  {stdout}")
    parts = stdout.split()
    if parts:
        return parts[-1]
    return None

def submit_job_with_dependency(script_path, dep_job_id):
    cmd = ["sbatch", f"--dependency=afterok:{dep_job_id}", str(script_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error submitting {script_path} with dependency: {res.stderr}")
        return None
    stdout = res.stdout.strip()
    print(f"  {stdout} (dependent on {dep_job_id})")
    parts = stdout.split()
    if parts:
        return parts[-1]
    return None

def clean_old_files(results_dir: Path, trait: str):
    trait_dir = results_dir / trait
    if not trait_dir.exists():
        return
    print(f"Cleaning old rank/proj_rank/masked files in {trait_dir}...")
    patterns = [
        "rank_only_Val*",
        "scores_rank_only_Val*",
        "masked_rank_only_Val*",
        "scores_masked_rank_only_Val*",
        "proj_rank_only_Val*",
        "scores_proj_rank_only_Val*",
        "masked_proj_rank_only_Val*",
        "scores_masked_proj_rank_only_Val*",
        "masked_cos_only_Val*",
        "scores_masked_cos_only_Val*",
        "masked_proj_cos_only_Val*",
        "scores_masked_proj_cos_only_Val*"
    ]
    for pattern in patterns:
        for f in trait_dir.glob(pattern):
            try:
                f.unlink()
                print(f"  Deleted: {f.name}")
            except Exception as e:
                print(f"  Failed to delete {f.name}: {e}")

def main():
    results_dir = Path("exp_steering_dyn_layer_raw/results")
    
    # Clean old files first
    for trait in TRAITS:
        clean_old_files(results_dir, trait)

    job_dir = Path("jobs/similarity_rank")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(str(v) for v in VALS)

    print("=== Submitting Similarity-based Rank/Proj-Rank Sweep & Evaluation Jobs ===")

    for trait in TRAITS:
        print(f"\nTrait: {trait}")
        
        # 1. Write Generation script
        gen_content = GEN_TEMPLATE.format(trait=trait, vals_list=vals_str)
        gen_file = job_dir / f"run_gen_sim_rank_{trait}.sh"
        with open(gen_file, "w", encoding="utf-8") as f:
            f.write(gen_content)
        gen_file.chmod(0o755)

        # 2. Write Evaluation script
        eval_content = EVAL_TEMPLATE.format(trait=trait)
        eval_file = job_dir / f"run_eval_sim_rank_{trait}.sh"
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write(eval_content)
        eval_file.chmod(0o755)

        # 3. Submit Gen Job
        print(f"Submitting Gen Job for {trait}...")
        gen_job_id = submit_job(gen_file)

        if gen_job_id:
            # 4. Submit dependent Eval Job
            print(f"Submitting dependent Eval Job for {trait}...")
            submit_job_with_dependency(eval_file, gen_job_id)
        else:
            print(f"Warning: Failed to submit generation job, skipping dependent eval.")

if __name__ == "__main__":
    main()
