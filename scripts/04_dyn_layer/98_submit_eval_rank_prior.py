#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 98_submit_eval_rank_prior.py
#
# Submits evaluation jobs for Rank-Prior and Zscore-Prior DLS results.
# Queries active SLURM jobs to automatically set up job dependencies.
#

import subprocess
import sys
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_{mode}_only_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=08:00:00
#SBATCH --output=log/eval_{mode}_only_{trait}.out
#SBATCH --error=log/eval_{mode}_only_{trait}.err
{dependency_line}

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting evaluation for {mode}-Only {trait}..."

for val in {vals_list}; do
    JSONL_OUT="exp_steering_dyn_layer_proj_prior/results/{trait}/{mode}_only_Val${{val}}.jsonl"
    CSV_OUT="exp_steering_dyn_layer_proj_prior/results/{trait}/scores_{mode}_only_Val${{val}}.csv"
    
    if [ -f "$JSONL_OUT" ]; then
        if [ ! -f "$CSV_OUT" ]; then
            echo "Evaluating alpha=$val..."
            "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
                --input "$JSONL_OUT" \\
                --output "$CSV_OUT" \\
                --axis "{trait}" \\
                --model "{model_name}" \\
                --quant "4bit"
        else
            echo "Already evaluated: $CSV_OUT"
        fi
    else
        echo "Warning: input file not found: $JSONL_OUT"
    fi
done

echo "Evaluation completed for {mode}-Only {trait}."
"""

def get_active_jobs():
    try:
        res = subprocess.run(
            ["squeue", "-u", "s2550009", "--format=%i %j"],
            capture_output=True, text=True, check=True
        )
        jobs = {}
        for line in res.stdout.strip().split("\n")[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                job_id, job_name = parts[0], parts[1]
                jobs[job_name] = job_id
        return jobs
    except Exception as e:
        print(f"Warning: failed to query squeue: {e}", file=sys.stderr)
        return {}

def main():
    JUDGE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
    job_dir = Path("jobs/eval_rank_zscore_prior")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(str(v) for v in VALS)
    active_jobs = get_active_jobs()
    print("Detected active jobs:", active_jobs)

    for mode in ["rank"]:
        for trait in TRAITS:
            gen_job_name = f"dls_{mode}_only_{trait}"
            dependency_line = ""
            if gen_job_name in active_jobs:
                job_id = active_jobs[gen_job_name]
                dependency_line = f"#SBATCH --dependency=afterok:{job_id}"
                print(f"Setting dependency for eval_{mode}_only_{trait} on generation job {job_id} ({gen_job_name})")
            else:
                print(f"No active generation job found for {gen_job_name}, submitting evaluation without dependency.")

            pbs_content = PBS_TEMPLATE.format(
                trait=trait,
                mode=mode,
                vals_list=vals_str,
                dependency_line=dependency_line,
                model_name=JUDGE_MODEL
            )
            pbs_file = job_dir / f"run_eval_{mode}_only_{trait}.sh"
            with open(pbs_file, "w") as f:
                f.write(pbs_content)
            pbs_file.chmod(0o755)

            cmd = ["sbatch", str(pbs_file)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(f"Submitting eval job for {mode} {trait}:")
            print(f"  {res.stdout.strip()} {res.stderr.strip()}")

if __name__ == "__main__":
    main()
