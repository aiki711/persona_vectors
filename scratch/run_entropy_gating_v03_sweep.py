#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_entropy_gating_v03_sweep.py
# Re-running the full Entropy Gating Sweep using the CORRECT model: Mistral-7B-Instruct-v0.3.
# Evaluates baseline, static steering, and plateau gating parameters (theta_hi: 4.0~9.0, k_hi: 0.5~2.0).
#

import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

THETA_LO = 1.2
K_LO = 1.5

THETA_HI_LIST = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
K_HI_LIST = [0.5, 1.0, 1.5, 2.0]

def main():
    print("Starting Mistral-7B-Instruct-v0.3 Entropy Gating Sweep Re-evaluation...")
    
    configs = []
    for theta_hi in THETA_HI_LIST:
        for k_hi in K_HI_LIST:
            configs.append((THETA_LO, theta_hi, K_LO, k_hi))
                
    print(f"Total Configurations to Evaluate on Mistral-7B-Instruct-v0.3: {len(configs)}")
    
    for idx, (th_lo, th_hi, k_l, k_h) in enumerate(configs, 1):
        config_name = f"v03-klo{k_l}-thi{th_hi}-khi{k_h}"
        
        # 1. Steering Generation for all 5 traits
        print(f"\n==========================================")
        print(f"[{idx}/{len(configs)}] Running Config: {config_name}")
        print(f"Model: Mistral-7B-Instruct-v0.3 | theta_lo={th_lo}, theta_hi={th_hi}, k_lo={k_l}, k_hi={k_h}")
        print(f"==========================================")
        
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_l:.1f}_{k_h:.1f}_entropy_plateau_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_l}_{k_h}_entropy_plateau_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name

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
                "--k_lo", str(k_l),
                "--k_hi", str(k_h),
                "--num_prompts", "10"
            ]
            res = subprocess.run(cmd, cwd=WORKSPACE)
            if res.returncode != 0:
                print(f"[ERROR] Steering generation failed for {trait} in {config_name}")
                
        # 2. 70B Judge Evaluation for each trait
        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            res_eval = subprocess.run(eval_cmd, cwd=WORKSPACE)
            if res_eval.returncode != 0:
                print(f"[ERROR] Evaluation failed for {trait} in {config_name}")

    # 3. Generate Summary Heatmaps and Report
    print("\n------------------------------------------")
    print("Generating Final Mistral-7B-v0.3 Summary Report & Heatmaps...")
    print("------------------------------------------")
    try:
        subprocess.run([sys.executable, "scratch/plot_entropy_gating_phase3_heatmaps.py"], cwd=WORKSPACE)
        subprocess.run([sys.executable, "scratch/plot_entropy_gating_tradeoff_scatter.py"], cwd=WORKSPACE)
        subprocess.run([sys.executable, "scratch/plot_optimal_alpha_function.py"], cwd=WORKSPACE)
    except Exception as e:
        print(f"Failed to run plot scripts: {e}")

if __name__ == "__main__":
    main()
