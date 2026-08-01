#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_logit_diff.py
# Re-evaluation of Logit-Diff Baseline on Mistral-7B-Instruct-v0.3
# Output Directory: exp_token_intensity/exp_v03_baselines/logit_diff
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_baselines/logit_diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHAS = ["2.0", "5.0", "8.0"]

def main():
    print("=======================================================")
    print("Starting Logit-Diff Re-Evaluation on Mistral-7B-Instruct-v0.3")
    print(f"Output Directory: {OUT_DIR}")
    print("=======================================================")

    for alpha in ALPHAS:
        print(f"\n==========================================")
        print(f"Running Logit-Diff Baseline for alpha={alpha}")
        print(f"==========================================")

        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_0.0_99.0_k_1.0_1.0_entropy_plateau_Val{alpha}.csv"
            csv_path = OUT_DIR / trait / csv_name

            if csv_path.exists():
                print(f"Skipping generation for {trait} (alpha={alpha}): Already evaluated.")
                continue

            cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py",
                "--config", "configs/mistral_7b.yaml",
                "--vector_bank", "vectors/mean_diff_vectors.npz",
                "--prompts", "inputs/eval_prompts_10.jsonl",
                "--mask_bank", "vectors/soft_probe_masks.npz",
                "--out_dir", str(OUT_DIR),
                "--axis", trait,
                "--alpha_max", alpha,
                "--gating_mode", "entropy_plateau",
                "--static_layer",
                "--theta_lo", "0.0",
                "--theta_hi", "99.0",
                "--k_lo", "1.0",
                "--k_hi", "1.0",
                "--num_prompts", "10"
            ]
            print(f"Generating for {trait} (alpha={alpha})...")
            subprocess.run(cmd, cwd=WORKSPACE, check=True)

        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            print(f"Running LLM Judge eval for {trait} (alpha={alpha})...")
            subprocess.run(eval_cmd, cwd=WORKSPACE, check=True)

    print("\n-------------------------------------------------------")
    print("Logit-Diff Re-Evaluation on Mistral-7B-v0.3 Finished Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
