#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/121_submit_gen_time_generation.py
#
# Submits text generation jobs for the 8 generation-time dynamic layer steering methods
# on the test set using a fixed seed (42) and raw-norm scaling.
# Delegates the loops to the Python script to avoid redundant model loading.
#

import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=gen_time_raw_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=03:00:00
#SBATCH --output=log/gen_time_raw_{trait}.out
#SBATCH --error=log/gen_time_raw_{trait}.err

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

echo "Starting generation-time dynamic steering on test prompts for {trait}..."

# Run full sweep internally in Python to avoid reloading model
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
    --sweep

echo "Generation-time dynamic steering completed on test prompts for {trait}."
"""

def main():
    job_dir = Path("jobs/gen_time_raw")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait)
        pbs_file = job_dir / f"run_gen_{trait}.sh"
        with open(pbs_file, "w", encoding="utf-8") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        cmd = ["sbatch", str(pbs_file)]
        print(f"Submitting generation job for {trait}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  Stdout: {res.stdout.strip()}")
        print(f"  Stderr: {res.stderr.strip()}")

if __name__ == "__main__":
    main()
