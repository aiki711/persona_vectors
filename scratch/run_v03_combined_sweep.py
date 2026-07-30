#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_combined_sweep.py
# Combined Rise & Fall Dynamic Gating (Trapezoidal Gating) Evaluation on Mistral-7B-Instruct-v0.3
# Output Directory: exp_token_intensity/exp_v03_combined_sweep
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_combined_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

# 2x2 Elite Configurations based on Rise & Fall sweeps
COMBINED_CONFIGS = [
    {"name": "high_score_x_high_score", "th_lo": 0.2, "k_lo": 4.0, "th_hi": 2.0, "k_hi": 0.20},
    {"name": "high_score_x_low_ppl",   "th_lo": 0.2, "k_lo": 4.0, "th_hi": 3.0, "k_hi": 0.30},
    {"name": "low_ppl_x_high_score",   "th_lo": 0.3, "k_lo": 3.0, "th_hi": 2.0, "k_hi": 0.20},
    {"name": "low_ppl_x_low_ppl",      "th_lo": 0.3, "k_lo": 3.0, "th_hi": 3.0, "k_hi": 0.30},
]

def main():
    print("=======================================================")
    print("Starting Combined Rise & Fall Dynamic Gating Evaluation (Mistral-7B-v0.3)")
    print(f"Output Directory: {OUT_DIR}")
    print(f"Total Combined Configurations: {len(COMBINED_CONFIGS)}")
    print("=======================================================")

    for idx, cfg in enumerate(COMBINED_CONFIGS, 1):
        th_lo = cfg["th_lo"]
        k_lo = cfg["k_lo"]
        th_hi = cfg["th_hi"]
        k_hi = cfg["k_hi"]
        cfg_name = cfg["name"]

        print(f"\n==========================================")
        print(f"[{idx}/{len(COMBINED_CONFIGS)}] Running Config: {cfg_name}")
        print(f"Rise: (theta_lo={th_lo}, k_lo={k_lo}) | Fall: (theta_hi={th_hi}, k_hi={k_hi})")
        print(f"==========================================")

        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_lo:.1f}_{k_hi:.2f}_entropy_plateau_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_lo}_{k_hi}_entropy_plateau_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name

            if csv_path.exists():
                print(f"Skipping generation for {trait} in {cfg_name}: Already evaluated.")
                continue

            cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py",
                "--config", "configs/mistral_7b.yaml",
                "--vector_bank", "vectors/mean_diff_vectors.npz",
                "--prompts", "inputs/eval_prompts_10.jsonl",
                "--mask_bank", "vectors/soft_probe_masks.npz",
                "--out_dir", str(OUT_DIR),
                "--axis", trait,
                "--alpha_max", "5.0",
                "--gating_mode", "entropy_plateau",
                "--static_layer",
                "--theta_lo", str(th_lo),
                "--theta_hi", str(th_hi),
                "--k_lo", str(k_lo),
                "--k_hi", str(k_hi),
                "--num_prompts", "10"
            ]
            print(f"Running generation for {trait}...")
            subprocess.run(cmd, cwd=WORKSPACE, check=True)

        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            print(f"Running LLM Judge eval for {trait}...")
            subprocess.run(eval_cmd, cwd=WORKSPACE, check=True)

    print("\n-------------------------------------------------------")
    print("Combined Rise & Fall Dynamic Gating Evaluation Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
