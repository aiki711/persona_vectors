#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_entropy_gating_v03_full_sweep.py
# Full Dual Sweep (Rise-stage & Fall-stage) on Mistral-7B-Instruct-v0.3
# Evaluates combinations of theta_lo (0.8~1.5), k_lo (1.0~2.0), theta_hi (4.0~9.0), and k_hi (0.5~1.5).
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

# Grid setup for rise-stage and fall-stage
THETA_LO_LIST = [0.8, 1.0, 1.2, 1.5]
K_LO_LIST = [1.0, 1.5, 2.0]

THETA_HI_LIST = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
K_HI_LIST = [0.5, 1.0, 1.5]

def main():
    print("Starting Comprehensive Rise & Fall Stage Entropy Gating Sweep on Mistral-7B-Instruct-v0.3...")
    
    # Core grid selection:
    # 1. Broad Rise-stage sweep with optimal fall-stage (theta_hi=6.0, k_hi=1.0)
    # 2. Broad Fall-stage sweep with optimal rise-stage (theta_lo=1.2, k_lo=1.5)
    # 3. Key dual combinations
    configs = []

    # 1. Rise-stage focus configs
    for th_lo in THETA_LO_LIST:
        for k_l in K_LO_LIST:
            configs.append((th_lo, 6.0, k_l, 1.0))
            configs.append((th_lo, 7.0, k_l, 1.0))

    # 2. Fall-stage focus configs
    for th_hi in THETA_HI_LIST:
        for k_h in K_HI_LIST:
            configs.append((1.2, th_hi, 1.5, k_h))
            configs.append((1.0, th_hi, 1.5, k_h))

    # Remove duplicates while maintaining order
    unique_configs = []
    seen = set()
    for item in configs:
        if item not in seen:
            seen.add(item)
            unique_configs.append(item)

    print(f"Total Unique Configurations to Evaluate: {len(unique_configs)}")

    for idx, (th_lo, th_hi, k_l, k_h) in enumerate(unique_configs, 1):
        config_name = f"v03-tlo{th_lo}-klo{k_l}-thi{th_hi}-khi{k_h}"
        
        print(f"\n==========================================")
        print(f"[{idx}/{len(unique_configs)}] Running Config: {config_name}")
        print(f"Model: Mistral-7B-Instruct-v0.3 | theta_lo={th_lo}, theta_hi={th_hi}, k_lo={k_l}, k_hi={k_h}")
        print(f"==========================================")
        
        # 1. Generation
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_l:.1f}_{k_h:.1f}_entropy_plateau_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_l}_{k_h}_entropy_plateau_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name

            if csv_path.exists():
                print(f"Skipping generation for {trait} in {config_name}: CSV already exists.")
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
                "--k_lo", str(k_l),
                "--k_hi", str(k_h),
                "--num_prompts", "10"
            ]
            res = subprocess.run(cmd, cwd=WORKSPACE)
            if res.returncode != 0:
                print(f"[ERROR] Steering generation failed for {trait} in {config_name}")

        # 2. Evaluation
        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            subprocess.run(eval_cmd, cwd=WORKSPACE)

    # 3. Plots and summaries
    print("\n------------------------------------------")
    print("Generating Summary Heatmaps and Reports...")
    print("------------------------------------------")
    try:
        subprocess.run([sys.executable, "scratch/plot_entropy_gating_phase3_heatmaps.py"], cwd=WORKSPACE)
        subprocess.run([sys.executable, "scratch/plot_entropy_gating_tradeoff_scatter.py"], cwd=WORKSPACE)
        subprocess.run([sys.executable, "scratch/plot_optimal_alpha_function.py"], cwd=WORKSPACE)
    except Exception as e:
        print(f"Failed to run plot scripts: {e}")

if __name__ == "__main__":
    main()
