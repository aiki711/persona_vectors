#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_baselines.py
# Runs baseline methods (No Steering, Logit-diff/DLS, No Gating/Static Steering) on Mistral-7B-Instruct-v0.3
# Output Directory: exp_token_intensity/exp_v03_baselines
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

def main():
    print("=======================================================")
    print("Starting Job A: Baseline Comparisons on Mistral-7B-Instruct-v0.3")
    print(f"Output Directory: {OUT_DIR}")
    print("=======================================================")

    # 1. M1: No Steering (alpha = 0.0)
    print("\n--- Running M1: No Steering (alpha=0.0) ---")
    for trait in TRAITS:
        cmd = [
            sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py",
            "--config", "configs/mistral_7b.yaml",
            "--vector_bank", "vectors/mean_diff_vectors.npz",
            "--prompts", "inputs/eval_prompts_10.jsonl",
            "--mask_bank", "vectors/soft_probe_masks.npz",
            "--out_dir", str(OUT_DIR / "no_steering"),
            "--axis", trait,
            "--alpha_max", "0.0",
            "--gating_mode", "entropy_plateau",
            "--static_layer",
            "--theta_lo", "0.0",
            "--theta_hi", "99.0",
            "--k_lo", "1.0",
            "--k_hi", "1.0",
            "--num_prompts", "10"
        ]
        subprocess.run(cmd, cwd=WORKSPACE)

        eval_cmd = [
            sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
            "--results_dir", str(OUT_DIR / "no_steering" / trait),
            "--axis", trait,
            "--quant", "4bit"
        ]
        subprocess.run(eval_cmd, cwd=WORKSPACE)

    # 2. M3: No Gating / Static Steering (alpha_max = 5.0, full tokens)
    print("\n--- Running M3: Static No Gating (alpha_max=5.0) ---")
    for trait in TRAITS:
        cmd = [
            sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py",
            "--config", "configs/mistral_7b.yaml",
            "--vector_bank", "vectors/mean_diff_vectors.npz",
            "--prompts", "inputs/eval_prompts_10.jsonl",
            "--mask_bank", "vectors/soft_probe_masks.npz",
            "--out_dir", str(OUT_DIR / "static_no_gating"),
            "--axis", trait,
            "--alpha_max", "5.0",
            "--gating_mode", "entropy_plateau",
            "--static_layer",
            "--theta_lo", "0.0",
            "--theta_hi", "99.0",
            "--k_lo", "1.0",
            "--k_hi", "1.0",
            "--num_prompts", "10"
        ]
        subprocess.run(cmd, cwd=WORKSPACE)

        eval_cmd = [
            sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
            "--results_dir", str(OUT_DIR / "static_no_gating" / trait),
            "--axis", trait,
            "--quant", "4bit"
        ]
        subprocess.run(eval_cmd, cwd=WORKSPACE)

    print("\n-------------------------------------------------------")
    print("Job A: Baseline Comparisons Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
