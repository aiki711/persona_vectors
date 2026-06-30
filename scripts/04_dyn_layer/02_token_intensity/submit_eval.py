#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/submit_eval.py
#
# Submits evaluation jobs for completed DLIS results on SLURM.
#

import argparse
import os
import subprocess
from pathlib import Path

# Paths
WORKSPACE = Path("/home/s2550009/persona_vectors")
JOBS_DIR = WORKSPACE / "jobs/02_token_intensity"
RESULTS_DIR = WORKSPACE / "exp_token_intensity/results"
LOG_DIR = WORKSPACE / "log/02_token_intensity"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_dlis_{trait}
#SBATCH --output={log_dir}/eval_dlis_{trait}.log
#SBATCH --error={log_dir}/eval_dlis_{trait}.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=2:00:00

cd {workspace}
source persona_steering/bin/activate

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \\
    --results_dir exp_token_intensity/results/{trait} \\
    --axis {trait} \\
    --quant 4bit
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness", "all"], default="all")
    args = ap.parse_args()

    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    if args.axis != "all":
        traits = [args.axis]

    for trait in traits:
        # Check if directory exists and has files
        trait_dir = RESULTS_DIR / trait
        if not trait_dir.exists():
            print(f"Skipping {trait} because results directory does not exist yet.")
            continue
            
        jsonl_files = list(trait_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"Skipping {trait} because no jsonl files are present yet.")
            continue

        print(f"Submitting evaluation job for {trait} ({len(jsonl_files)} files found)...")
        script_content = TEMPLATE.format(
            workspace=WORKSPACE,
            log_dir=LOG_DIR,
            trait=trait
        )
        
        script_path = JOBS_DIR / f"run_eval_dlis_{trait}.sh"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        res = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Submitted: run_eval_dlis_{trait}.sh -> {res.stdout.strip()}")
        else:
            print(f"[ERROR] Failed to submit run_eval_dlis_{trait}.sh: {res.stderr}")

if __name__ == "__main__":
    main()
