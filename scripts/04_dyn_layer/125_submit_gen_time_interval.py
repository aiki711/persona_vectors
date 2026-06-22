#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/125_submit_gen_time_interval.py
#
# Submits generation-time DLS sweeps and evaluation jobs for update_interval = 4 and 8.
# Uses SLURM dependencies (--dependency=afterok) to chain evaluation after generation.
#

import subprocess
import os
import sys
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
INTERVALS = [4, 8]

GEN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=gen_int{interval}_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_int{interval}_{trait}.out
#SBATCH --error=log/gen_int{interval}_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_dyn_gen_time_interval_raw/results_interval{interval}"

echo "Starting generation-time DLS on test prompts for {trait} (interval={interval})..."

"$PYTHON_BIN" scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py \\
    --config "$CONFIG" \\
    --vector_bank "$VECTOR_BANK" \\
    --prompts "$PROMPT_IN" \\
    --out_dir "$OUT_DIR" \\
    --axis "{trait}" \\
    --direction "high" \\
    --norm_mode "raw_norm" \\
    --mask_bank "$MASK_BANK" \\
    --update_interval {interval} \\
    --seed 42 \\
    --sweep

echo "Generation-time DLS completed for {trait} (interval={interval})."
"""

EVAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_int{interval}_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/eval_int{interval}_{trait}.out
#SBATCH --error=log/eval_int{interval}_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting batch evaluation for generation-time steering of {trait} (interval={interval})..."

"$PYTHON_BIN" scripts/04_dyn_layer/115_batch_eval.py \\
    --results_dir "exp_steering_dyn_gen_time_interval_raw/results_interval{interval}/{trait}" \\
    --axis "{trait}" \\
    --quant "4bit"

echo "Batch evaluation completed for {trait} (interval={interval})."
"""

def submit_job(script_path):
    cmd = ["sbatch", str(script_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error submitting {script_path}: {res.stderr}")
        return None
    # Parse Job ID from stdout: "Submitted batch job 123456"
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
    job_dir = Path("jobs/gen_time_interval")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== Submitting Generation-Time Dynamic Steering Interval Sweep Jobs ===")

    for interval in INTERVALS:
        print(f"\n--- Interval: {interval} ---")
        for trait in TRAITS:
            # 1. Write Generation script
            gen_content = GEN_TEMPLATE.format(trait=trait, interval=interval)
            gen_file = job_dir / f"run_gen_int{interval}_{trait}.sh"
            with open(gen_file, "w", encoding="utf-8") as f:
                f.write(gen_content)
            gen_file.chmod(0o755)

            # 2. Write Evaluation script
            eval_content = EVAL_TEMPLATE.format(trait=trait, interval=interval)
            eval_file = job_dir / f"run_eval_int{interval}_{trait}.sh"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write(eval_content)
            eval_file.chmod(0o755)

            # 3. Submit Generation Job
            print(f"Submitting Generation Job for {trait} (interval={interval})...")
            gen_job_id = submit_job(gen_file)

            if gen_job_id:
                # 4. Submit dependent Evaluation Job
                print(f"Submitting Evaluation Job for {trait} (interval={interval})...")
                submit_job_with_dependency(eval_file, gen_job_id)
            else:
                print(f"Warning: Failed to submit generation job, skipping dependent eval job.")

if __name__ == "__main__":
    main()
