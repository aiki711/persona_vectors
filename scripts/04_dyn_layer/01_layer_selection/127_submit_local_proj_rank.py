#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/127_submit_local_proj_rank.py
#
# Submits text generation sweeps and Llama-3 evaluations for the new Local Proj-Rank methods.
# Uses SLURM dependencies to chain evaluation after generation completes.
#

import subprocess
import os
import sys
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

GEN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=gen_local_proj_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_local_proj_{trait}.out
#SBATCH --error=log/gen_local_proj_{trait}.err

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

echo "Starting Local Proj-Rank sweeps for {trait}..."

for val in {vals_list}; do
    # 1. Unmasked Local Proj-Rank
    JSONL_OUT="${{OUT_DIR}}/{trait}/local_proj_rank_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running Local Proj-Rank alpha=$val ==="
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
            --score_mode "local_proj_rank" \\
            --seed 42
    fi

    # 2. Masked Local Proj-Rank (PDF)
    JSONL_OUT="${{OUT_DIR}}/{trait}/masked_local_proj_rank_only_Val${{val}}.jsonl"
    if [ ! -f "$JSONL_OUT" ]; then
        echo "=== Running PDF Local Proj-Rank alpha=$val ==="
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
            --score_mode "local_proj_rank" \\
            --mask_bank "$MASK_BANK" \\
            --seed 42
    fi
done

echo "Local Proj-Rank sweeps completed for {trait}."
"""

EVAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_local_proj_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/eval_local_proj_{trait}.out
#SBATCH --error=log/eval_local_proj_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting evaluation of Local Proj-Rank results for {trait}..."

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

def main():
    job_dir = Path("jobs/local_proj_rank")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(str(v) for v in VALS)

    print("=== Submitting Local Proj-Rank DLS Sweep & Evaluation Jobs ===")

    for trait in TRAITS:
        print(f"\nTrait: {trait}")
        
        # 1. Write Generation script
        gen_content = GEN_TEMPLATE.format(trait=trait, vals_list=vals_str)
        gen_file = job_dir / f"run_gen_local_proj_{trait}.sh"
        with open(gen_file, "w", encoding="utf-8") as f:
            f.write(gen_content)
        gen_file.chmod(0o755)

        # 2. Write Evaluation script
        eval_content = EVAL_TEMPLATE.format(trait=trait)
        eval_file = job_dir / f"run_eval_local_proj_{trait}.sh"
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
