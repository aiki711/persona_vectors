#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 96_submit_eval_baselines.py
#
# Submits evaluation jobs for Baseline (logit_diff & anti_alignment) results.
#

import subprocess
import sys
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
METHODS = ["logit_diff"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_baseline_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=12:00:00
#SBATCH --output=log/eval_baseline_{trait}.out
#SBATCH --error=log/eval_baseline_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting baseline evaluation for {trait}..."

for method in {methods_list}; do
    for val in {vals_list}; do
        JSONL_OUT="exp_steering_dyn_layer_all_layers_midpoint/results/{trait}/${{method}}_Val${{val}}.jsonl"
        CSV_OUT="exp_steering_dyn_layer_all_layers_midpoint/results/{trait}/scores_${{method}}_Val${{val}}.csv"
        
        if [ -f "$JSONL_OUT" ]; then
            if [ ! -f "$CSV_OUT" ]; then
                echo "Evaluating ${{method}} alpha=$val..."
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
done

echo "Baseline evaluation completed for {trait}."
"""

def main():
    JUDGE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
    job_dir = Path("jobs/eval_baseline")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(str(v) for v in VALS)
    methods_str = " ".join(METHODS)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(
            trait=trait,
            vals_list=vals_str,
            methods_list=methods_str,
            model_name=JUDGE_MODEL
        )
        pbs_file = job_dir / f"run_eval_baseline_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        cmd = ["sbatch", str(pbs_file)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Submitting eval job for {trait}:")
        print(f"  {res.stdout.strip()} {res.stderr.strip()}")

if __name__ == "__main__":
    main()
