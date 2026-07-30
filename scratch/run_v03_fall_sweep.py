#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_fall_sweep.py
# Extended Fall-Stage (theta_hi, k_hi) Fine-Grained Sweep on Mistral-7B-Instruct-v0.3
# Output Directory: exp_token_intensity/exp_v03_fall_sweep
# Rise-stage is kept UNCONTROLLED (theta_lo = 0.0)
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_fall_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

# Updated and extended Fall-stage grid based on heatmap analysis:
# theta_hi shifted smaller (1.5 to 3.5), k_hi shifted smaller (0.05 to 0.4)
THETA_HI_LIST = [1.5, 2.0, 2.5, 3.0, 3.5]
K_HI_LIST = [0.05, 0.1, 0.2, 0.3, 0.4]

def main():
    print("=======================================================")
    print("Starting Job C: Extended Fall-Stage Sweep on Mistral-7B-Instruct-v0.3")
    print(f"Output Directory: {OUT_DIR}")
    print("Rise-stage fixed at theta_lo = 0.0 (No rise-stage gating)")
    print("=======================================================")

    configs = []
    for th_hi in THETA_HI_LIST:
        for k_h in K_HI_LIST:
            configs.append((th_hi, k_h))

    print(f"Total Fall-Stage Configurations to Evaluate: {len(configs)}")

    for idx, (th_hi, k_h) in enumerate(configs, 1):
        config_name = f"fall-thi{th_hi:.1f}-khi{k_h:.2f}"
        print(f"\n==========================================")
        print(f"[{idx}/{len(configs)}] Running Config: {config_name}")
        print(f"Model: Mistral-7B-Instruct-v0.3 | theta_hi={th_hi}, k_hi={k_h}")
        print(f"==========================================")

        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_0.0_{th_hi:.1f}_k_1.0_{k_h:.2f}_entropy_plateau_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_0.0_{th_hi:.1f}_k_1.0_{k_h:.1f}_entropy_plateau_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_0.0_{th_hi}_k_1.0_{k_h}_entropy_plateau_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name

            if csv_path.exists():
                print(f"Skipping generation for {trait} in {config_name}: Already evaluated.")
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
                "--theta_lo", "0.0",
                "--theta_hi", str(th_hi),
                "--k_lo", "1.0",
                "--k_hi", str(k_h),
                "--num_prompts", "10"
            ]
            subprocess.run(cmd, cwd=WORKSPACE)

        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            subprocess.run(eval_cmd, cwd=WORKSPACE)

    print("\n-------------------------------------------------------")
    print("Job C: Extended Fall-Stage Sweep Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
