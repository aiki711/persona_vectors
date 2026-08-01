#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_proj_rank_sweeps.py
# Full Rise & Fall Parameter Sweeps (25 pairs each) on proj_rank Dynamic Layer Selection Axis
# Model & Vectors: Mistral-7B-Instruct-v0.3
# Output Directory: exp_token_intensity/exp_v03_proj_rank_sweeps
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_proj_rank_sweeps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHA_MAX = "5.0"

# Grids
THETA_LO_LIST = ["0.1", "0.2", "0.3", "0.4", "0.6"]
K_LO_LIST = ["2.0", "3.0", "4.0", "5.0", "6.0"]

THETA_HI_LIST = ["1.5", "2.0", "2.5", "3.0", "3.5"]
K_HI_LIST = ["0.05", "0.1", "0.2", "0.3", "0.4"]

def run_sweep(stage_name, theta_lo_list, k_lo_list, theta_hi_list, k_hi_list):
    print(f"\n=======================================================")
    print(f"Starting {stage_name} Parameter Sweep on proj_rank Axis")
    print(f"=======================================================")

    for t_lo in theta_lo_list:
        for k_lo in k_lo_list:
            for t_hi in theta_hi_list:
                for k_hi in k_hi_list:
                    print(f"\n--- Running Pair: theta_lo={t_lo}, k_lo={k_lo}, theta_hi={t_hi}, k_hi={k_hi} ---")
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
                            "--theta_lo", t_lo,
                            "--k_lo", k_lo,
                            "--theta_hi", t_hi,
                            "--k_hi", k_hi
                        ]
                        subprocess.run(cmd, cwd=WORKSPACE, check=True)

                    for trait in TRAITS:
                        eval_cmd = [
                            sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                            "--results_dir", str(OUT_DIR / trait),
                            "--axis", trait,
                            "--quant", "4bit"
                        ]
                        subprocess.run(eval_cmd, cwd=WORKSPACE, check=True)

def main():
    # 1. Rise Stage Sweep (theta_hi=99.0, k_hi=1.0)
    run_sweep(
        stage_name="Rise Stage (25 pairs)",
        theta_lo_list=THETA_LO_LIST,
        k_lo_list=K_LO_LIST,
        theta_hi_list=["99.0"],
        k_hi_list=["1.0"]
    )

    # 2. Fall Stage Sweep (theta_lo=0.0, k_lo=1.0)
    run_sweep(
        stage_name="Fall Stage (25 pairs)",
        theta_lo_list=["0.0"],
        k_lo_list=["1.0"],
        theta_hi_list=THETA_HI_LIST,
        k_hi_list=K_HI_LIST
    )

    print("\n-------------------------------------------------------")
    print("All proj_rank Rise & Fall Sweeps Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
