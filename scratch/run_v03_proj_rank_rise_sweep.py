#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_proj_rank_rise_sweep.py
# proj_rank Axis Rise Stage Parameter Sweep (25 pairs)
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

THETA_LO_LIST = ["0.1", "0.2", "0.3", "0.4", "0.6"]
K_LO_LIST = ["2.0", "3.0", "4.0", "5.0", "6.0"]

def main():
    print("=======================================================")
    print("Starting Rise Stage Parameter Sweep (25 pairs) on proj_rank Axis")
    print(f"Output Directory: {OUT_DIR}")
    print("=======================================================")

    for t_lo in THETA_LO_LIST:
        for k_lo in K_LO_LIST:
            print(f"\n--- Running Rise Pair: theta_lo={t_lo}, k_lo={k_lo} ---")
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
                    "--theta_hi", "99.0",
                    "--k_hi", "1.0"
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

    print("\nPlotting Rise heatmaps...")
    plot_cmd = [sys.executable, "-u", "scratch/plot_v03_proj_rank_heatmaps.py"]
    subprocess.run(plot_cmd, cwd=WORKSPACE, check=False)

    print("\n-------------------------------------------------------")
    print("proj_rank Rise Stage Sweep Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
