#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_dual_dynamic_steering.py
# Dual Dynamic Steering: Best Layer Selection (proj_rank) x Token Intensity Dynamic Gating (Rise, Fall, Combined)
# Model & Vectors: Mistral-7B-Instruct-v0.3
# Output Directory: exp_token_intensity/exp_v03_dual_dynamic_steering
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_dual_dynamic_steering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHA_MAX = "5.0"

CONFIGS = [
    # (Name, theta_lo, k_lo, theta_hi, k_hi)
    ("Rise_Dynamic", "0.2", "4.0", "99.0", "1.0"),
    ("Fall_Dynamic", "0.0", "1.0", "2.0", "0.20"),
    ("Combined_Config1_HighScore_HighScore", "0.2", "4.0", "2.0", "0.20"),
    ("Combined_Config2_HighScore_LowPPL", "0.2", "4.0", "3.0", "0.30"),
    ("Combined_Config3_LowPPL_HighScore", "0.3", "3.0", "2.0", "0.20"),
    ("Combined_Config4_LowPPL_LowPPL", "0.3", "3.0", "3.0", "0.30"),
]

def main():
    print("=======================================================")
    print("Starting Dual Dynamic Steering (proj_rank x Token Intensity Gating) Evaluation on Mistral-7B-v0.3")
    print(f"Alpha Max: {ALPHA_MAX}")
    print(f"Output Directory: {OUT_DIR}")
    print("=======================================================")

    for name, theta_lo, k_lo, theta_hi, k_hi in CONFIGS:
        print(f"\n==========================================")
        print(f"Running Dual Dynamic Steering: {name}")
        print(f"Params: theta_lo={theta_lo}, k_lo={k_lo}, theta_hi={theta_hi}, k_hi={k_hi}")
        print(f"==========================================")

        for trait in TRAITS:
            trait_out_dir = OUT_DIR / trait
            trait_out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py",
                "--config", "configs/mistral_7b.yaml",
                "--vector_bank", "vectors/mean_diff_vectors.npz",
                "--prompts", "inputs/eval_prompts_10.jsonl",
                "--mask_bank", "vectors/probe_masks.npz",
                "--out_dir", str(OUT_DIR),
                "--axis", trait,
                "--alpha_max", ALPHA_MAX,
                "--score_mode", "proj_rank",
                "--gating_mode", "entropy_plateau",
                "--theta_lo", theta_lo,
                "--k_lo", k_lo,
                "--theta_hi", theta_hi,
                "--k_hi", k_hi
            ]
            print(f"Generating for {trait} ({name})...")
            subprocess.run(cmd, cwd=WORKSPACE, check=True)

        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            print(f"Running LLM Judge eval for {trait} ({name})...")
            subprocess.run(eval_cmd, cwd=WORKSPACE, check=True)

    print("\n-------------------------------------------------------")
    print("Dual Dynamic Steering Evaluation Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
