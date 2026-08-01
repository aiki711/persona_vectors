#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_real_logit_diff.py
# Real Logit-Diff (Dynamic Layer Selection via score_mode=logit_diff) Baseline on Mistral-7B-v0.3
# Output Directory: exp_token_intensity/exp_v03_baselines/logit_diff
# (Alpha fixed to 5.0)
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_baselines/logit_diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHAS = ["5.0"]

def main():
    print("=======================================================")
    print("Starting Real Logit-Diff (score_mode=logit_diff) Evaluation on Mistral-7B-v0.3 (alpha=5.0)")
    print(f"Output Directory: {OUT_DIR}")
    print("=======================================================")

    for val in ALPHAS:
        print(f"\n==========================================")
        print(f"Running Real Logit-Diff for alpha={val}")
        print(f"==========================================")

        for trait in TRAITS:
            csv_name = f"scores_logit_diff_Val{val}.csv"
            csv_path = OUT_DIR / trait / csv_name
            if csv_path.exists():
                print(f"Skipping generation for {trait} (alpha={val}): Already evaluated.")
                continue

            cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/01_layer_selection/82_run_dyn_layer_steering.py",
                "--config", "configs/mistral_7b.yaml",
                "--vector_bank", "vectors/mean_diff_vectors.npz",
                "--prompts", "inputs/eval_prompts_10.jsonl",
                "--mask_bank", "vectors/probe_masks.npz",
                "--out_dir", str(OUT_DIR),
                "--axis", trait,
                "--alpha", val,
                "--direction", "high",
                "--norm_mode", "raw_norm",
                "--score_mode", "logit_diff"
            ]
            print(f"Generating Logit-Diff text for {trait} (alpha={val})...")
            subprocess.run(cmd, cwd=WORKSPACE, check=True)

        for trait in TRAITS:
            csv_name = f"scores_logit_diff_Val{val}.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                eval_cmd = [
                    sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                    "--results_dir", str(OUT_DIR / trait),
                    "--axis", trait,
                    "--quant", "4bit"
                ]
                print(f"Running LLM Judge eval for {trait} (alpha={val})...")
                subprocess.run(eval_cmd, cwd=WORKSPACE, check=True)

    print("\n-------------------------------------------------------")
    print("Real Logit-Diff (score_mode=logit_diff, alpha=5.0) Evaluation Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
