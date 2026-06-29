#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/131_submit_gen_time_alpha_5.py
#
# Submits text generation sweeps and Llama-3 evaluations for generation-time DLS at fixed alpha=5.0.
# Cleans up old results first to avoid skip triggers.
# Uses SLURM dependencies to chain evaluation after generation completes.
#

import subprocess
import os
import sys
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

GEN_TEMPLATE = """#!/bash
#SBATCH --job-name=gen_time_alpha5_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_time_alpha5_{trait}.out
#SBATCH --error=log/gen_time_alpha5_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
MASK_BANK="vectors/probe_masks.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_dyn_gen_time_raw/results"

echo "Starting generation-time dynamic steering (alpha=5.0) for {trait}..."

"$PYTHON_BIN" scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py \\
    --config "$CONFIG" \\
    --vector_bank "$VECTOR_BANK" \\
    --prompts "$PROMPT_IN" \\
    --out_dir "$OUT_DIR" \\
    --axis "{trait}" \\
    --direction "high" \\
    --norm_mode "raw_norm" \\
    --mask_bank "$MASK_BANK" \\
    --update_interval 1 \\
    --seed 42 \\
    --sweep \\
    --alphas "5.0"

echo "Generation-time dynamic steering (alpha=5.0) completed for {trait}."
"""

EVAL_TEMPLATE = """#!/bash
#SBATCH --job-name=eval_gen_time_alpha5_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=02:00:00
#SBATCH --output=log/eval_gen_time_alpha5_{trait}.out
#SBATCH --error=log/eval_gen_time_alpha5_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting evaluation of Similarity-based Rank/Proj-Rank results for {trait}..."

"$PYTHON_BIN" scripts/04_dyn_layer/115_batch_eval.py \\
    --results_dir "exp_steering_dyn_gen_time_raw/results/{trait}" \\
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
    print(f"Cleaning old alpha=5.0 files in {trait_dir}...")
    patterns = [
        "*_Val5.0.jsonl",
        "scores_*_Val5.0.csv"
    ]
    for pattern in patterns:
        for f in trait_dir.glob(pattern):
            try:
                f.unlink()
                print(f"  Deleted: {f.name}")
            except Exception as e:
                print(f"  Failed to delete {f.name}: {e}")

def main():
    results_dir = Path("exp_steering_dyn_gen_time_raw/results")
    
    # Clean old files first
    for trait in TRAITS:
        clean_old_files(results_dir, trait)

    job_dir = Path("jobs/gen_time_alpha_5")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== Submitting Gen-Time DLS at Fixed Alpha=5.0 Jobs ===")

    for trait in TRAITS:
        print(f"\nTrait: {trait}")
        
        # 1. Write Generation script (replace shebang placeholder ##!/bash with ##!/bin/bash)
        gen_content = GEN_TEMPLATE.replace("#!/bash", "#!/bin/bash").format(trait=trait)
        gen_file = job_dir / f"run_gen_time_alpha5_{trait}.sh"
        with open(gen_file, "w", encoding="utf-8") as f:
            f.write(gen_content)
        gen_file.chmod(0o755)

        # 2. Write Evaluation script
        eval_content = EVAL_TEMPLATE.replace("#!/bash", "#!/bin/bash").format(trait=trait)
        eval_file = job_dir / f"run_eval_gen_time_alpha5_{trait}.sh"
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
