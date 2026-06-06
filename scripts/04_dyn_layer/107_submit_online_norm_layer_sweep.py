#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 107_submit_online_norm_layer_sweep.py
#
# Submits midpoint-norm scaled single-layer steering sweep jobs to SLURM.
#

import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=midpoint_norm_sweep_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/midpoint_norm_sweep_{trait}.out
#SBATCH --error=log/midpoint_norm_sweep_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="inputs/eval_prompts_10.jsonl"
OUT_DIR="exp_steering_layer_midpoint_norm/results"

echo "Running midpoint-norm scaled single-layer sweep for {trait}..."

"$PYTHON_BIN" scripts/04_dyn_layer/106_run_online_norm_layer_sweep.py \\
    --config "$CONFIG" \\
    --vector_bank "$VECTOR_BANK" \\
    --prompts "$PROMPT_IN" \\
    --out_dir "$OUT_DIR" \\
    --axis "{trait}" \\
    --direction "high" \\
    --judge_model "meta-llama/Meta-Llama-3-70B-Instruct" \\
    --judge_quant "4bit"

echo "Done: {trait}"
"""


def main():
    job_dir = Path("jobs/midpoint_norm_layer_sweep")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait)
        pbs_file = job_dir / f"run_midpoint_norm_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        cmd = ["sbatch", str(pbs_file)]
        print(f"Submitting midpoint-norm layer sweep job for {trait}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  {res.stdout.strip()} {res.stderr.strip()}")


if __name__ == "__main__":
    main()
