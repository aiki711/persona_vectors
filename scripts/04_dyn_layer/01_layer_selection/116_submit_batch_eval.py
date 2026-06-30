#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 116_submit_batch_eval.py
#
# Detects running/pending generation jobs in SLURM, and submits batch evaluation jobs
# with SLURM dependencies (`--dependency=afterok:<jobid>`) so they automatically run
# after the corresponding generation job completes.
#

import subprocess
import re
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_raw_fixed_seed_batch_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/eval_raw_fixed_seed_batch_{trait}.out
#SBATCH --error=log/eval_raw_fixed_seed_batch_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting batch evaluation for {trait}..."
"$PYTHON_BIN" scripts/04_dyn_layer/115_batch_eval.py \\
    --results_dir "exp_steering_dyn_layer_raw/results/{trait}" \\
    --axis "{trait}" \\
    --quant "4bit"

echo "Batch evaluation completed for {trait}."
"""

def get_active_jobs():
    """Queries squeue and returns a map of job_name -> job_id."""
    import os
    username = os.environ.get('USER', 's2550009')
    try:
        res = subprocess.run(["squeue", "-u", username, "--format=%i %j"], capture_output=True, text=True)
        jobs = {}
        for line in res.stdout.strip().split("\n")[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                job_id = parts[0]
                job_name = parts[1]
                jobs[job_name] = job_id
        return jobs
    except Exception as e:
        print(f"Warning: failed to query squeue: {e}")
        return {}

def main():
    job_dir = Path("jobs/eval_raw_fixed_seed_batch")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    active_jobs = get_active_jobs()
    print(f"Active/queued jobs: {active_jobs}")

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait)
        pbs_file = job_dir / f"run_eval_batch_{trait}.sh"
        with open(pbs_file, "w", encoding="utf-8") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        # Detect dependency
        gen_job_name = f"gen_raw_fixed_seed_{trait}"
        # If job name was truncated in squeue output, try prefix matching
        gen_job_id = None
        for name, jid in active_jobs.items():
            if name.startswith(gen_job_name) or gen_job_name.startswith(name):
                gen_job_id = jid
                break

        cmd = ["sbatch"]
        if gen_job_id:
            cmd.append(f"--dependency=afterok:{gen_job_id}")
            print(f"Submitting eval job for {trait} with dependency on gen job {gen_job_id} ({gen_job_name})...")
        else:
            print(f"No active gen job found for {trait}. Submitting eval job directly...")
            
        cmd.append(str(pbs_file))
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  Stdout: {res.stdout.strip()}")
        print(f"  Stderr: {res.stderr.strip()}")

if __name__ == "__main__":
    main()
