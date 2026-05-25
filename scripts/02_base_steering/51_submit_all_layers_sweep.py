#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 51_submit_all_layers_sweep.py
#
# SLURM に全32層の単層スイープ実験ジョブを投入するスクリプト。
#

import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=all_steer_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=log/all_steer_{trait}.out
#SBATCH --error=log/all_steer_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
OUT_DIR="exp_steering_layer_analysis/results"

echo "Running Full-Layer (32 layers) single-layer sweep for {trait}..."
"$PYTHON_BIN" scripts/02_base_steering/50_run_all_layers_sweep.py \\
    --config "$CONFIG" \\
    --vector_bank "$VECTOR_BANK" \\
    --prompts "$PROMPT_IN" \\
    --out_dir "$OUT_DIR" \\
    --axis "{trait}" \\
    --direction "high"
"""

def main():
    job_dir = Path("jobs/all_layers_sweep")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait)
        pbs_file = job_dir / f"run_all_steer_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        print(f"Submitting full single-layer sweep job for {trait}...")
        res = subprocess.run(["sbatch", str(pbs_file)], capture_output=True, text=True)
        print(f"  {res.stdout.strip()} {res.stderr.strip()}")

if __name__ == "__main__":
    main()
