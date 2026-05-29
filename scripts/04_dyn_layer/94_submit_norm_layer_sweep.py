#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 94_submit_norm_layer_sweep.py
#
# Submits norm-scaled single-layer steering sweep jobs to SLURM.
# Uses steering vectors normalized to each layer's raw difference vector norm.
#

import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=norm_sweep_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=log/norm_sweep_{trait}.out
#SBATCH --error=log/norm_sweep_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
OUT_DIR="exp_steering_layer_norm/results"

echo "Running norm-scaled single-layer sweep for {trait}..."

"$PYTHON_BIN" scripts/04_dyn_layer/93_run_norm_layer_sweep.py \\
    --config "$CONFIG" \\
    --vector_bank "$VECTOR_BANK" \\
    --prompts "$PROMPT_IN" \\
    --out_dir "$OUT_DIR" \\
    --axis "{trait}" \\
    --direction "high"

echo "Done: {trait}"
"""


def main():
    job_dir = Path("jobs/norm_layer_sweep")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait)
        pbs_file = job_dir / f"run_norm_sweep_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        cmd = ["sbatch", str(pbs_file)]
        print(f"Submitting norm-scaled layer sweep job for {trait}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  {res.stdout.strip()} {res.stderr.strip()}")


if __name__ == "__main__":
    main()
