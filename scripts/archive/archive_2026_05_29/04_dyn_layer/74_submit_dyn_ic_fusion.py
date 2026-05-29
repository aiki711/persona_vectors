#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 74_submit_dyn_ic_fusion.py
#
# DLS (Relative Anti-alignment) と IC-Adaptive Steering (Sigmoid / Soft-Plateau) の
# 融合実験ジョブを SLURM に投入するスクリプト。
#
# 実験フォルダ: exp_steering_dyn_ic_fusion/
#

import os
import subprocess
from pathlib import Path

TRAITS      = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHA_MAXES = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
IC_MODES    = ["sigmoid", "soft_plateau"]
ALL_LAYERS  = ",".join(map(str, range(32)))

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dyn_ic_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_ic_{trait}.out
#SBATCH --error=log/dyn_ic_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_ic_fusion/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

AMAXES=({amax_str})
IC_MODES=({ic_modes_str})

for MODE in "${{IC_MODES[@]}}"; do
    for AMAX in "${{AMAXES[@]}}"; do
        echo "Running DLS + IC Fusion: Trait={trait}, Mode=$MODE, AlphaMax=$AMAX"
        JSONL_OUT="${{OUT_DIR}}/{trait}/fusion_${{MODE}}_Val${{AMAX}}.jsonl"
        CSV_OUT="${{OUT_DIR}}/{trait}/scores_fusion_${{MODE}}_Val${{AMAX}}.csv"

        if [ ! -f "$JSONL_OUT" ]; then
            "$PYTHON_BIN" scripts/04_dyn_layer/73_run_dyn_ic_fusion.py \\
                --config "$CONFIG" \\
                --vector_bank "$VECTOR_BANK" \\
                --prompts "$PROMPT_IN" \\
                --out_dir "$OUT_DIR" \\
                --axis "{trait}" \\
                --direction "high" \\
                --alpha_max "$AMAX" \\
                --ic_mode "$MODE" \\
                --layers "{layers}"
        else
            echo "  [SKIP] Generation already done: $JSONL_OUT"
        fi

        if [ ! -f "$CSV_OUT" ]; then
            "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
                --input "$JSONL_OUT" \\
                --output "$CSV_OUT" \\
                --axis "{trait}" \\
                --model "$JUDGE_MODEL"
        else
            echo "  [SKIP] Evaluation already done: $CSV_OUT"
        fi
    done
done
"""

def main():
    job_dir = Path("jobs/dyn_ic_fusion")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    amax_str = " ".join(map(str, ALPHA_MAXES))
    ic_modes_str = " ".join(IC_MODES)

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(
            trait=trait,
            amax_str=amax_str,
            ic_modes_str=ic_modes_str,
            layers=ALL_LAYERS,
        )

        pbs_file = job_dir / f"run_fusion_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        print(f"Submitting fusion job for {trait}...")
        res = subprocess.run(["sbatch", str(pbs_file)], capture_output=True, text=True)
        print(f"  {res.stdout.strip()} {res.stderr.strip()}")

if __name__ == "__main__":
    main()
