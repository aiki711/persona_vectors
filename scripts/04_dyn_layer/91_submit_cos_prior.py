#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 91_submit_cos_prior.py
#
# Submits the Cosine Similarity & Safety Prior DLS method sweeps to the SLURM queue.
# Generates batch files and submits them.
#

import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dls_cos_prior_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=log/dls_cos_prior_{trait}.out
#SBATCH --error=log/dls_cos_prior_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
INPUT_DIR="exp_steering_layer_analysis/results"
OUT_DIR="exp_steering_dyn_layer_proj_prior/results"

echo "Running Cos-Prior DLS sweep for {trait}..."

# Loop over values and run the script with --score_mode cosine
for val in {vals_list}; do
    echo "=== Running alpha=$val ==="
    "$PYTHON_BIN" scripts/04_dyn_layer/82_run_dyn_layer_proj_prior.py \\
        --config "$CONFIG" \\
        --vector_bank "$VECTOR_BANK" \\
        --prompts "$PROMPT_IN" \\
        --input_dir "$INPUT_DIR" \\
        --out_dir "$OUT_DIR" \\
        --axis "{trait}" \\
        --alpha "$val" \\
        --direction "high" \\
        --norm_mode "midpoint" \\
        --score_mode "cosine"
done
"""

def main():
    job_dir = Path("jobs/dls_cos_prior")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(str(v) for v in VALS)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait, vals_list=vals_str)
        pbs_file = job_dir / f"run_dls_cos_prior_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        cmd = ["sbatch", str(pbs_file)]
        print(f"Submitting Cos-Prior DLS job for {trait}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  {res.stdout.strip()} {res.stderr.strip()}")

if __name__ == "__main__":
    main()
