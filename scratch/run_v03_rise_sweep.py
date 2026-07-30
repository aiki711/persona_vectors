#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_rise_sweep.py
# Extended Rise-Stage (theta_lo, k_lo) Fine-Grained Sweep on Mistral-7B-Instruct-v0.3
# Output Directory: exp_token_intensity/exp_v03_rise_sweep
# Fall-stage is kept UNCONTROLLED (theta_hi = 99.0)
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_rise_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

# Updated and extended Rise-stage grid based on heatmap analysis:
# theta_lo shifted smaller (0.1 to 0.6), k_lo shifted larger (2.0 to 6.0)
THETA_LO_LIST = [0.1, 0.2, 0.3, 0.4, 0.6]
K_LO_LIST = [2.0, 3.0, 4.0, 5.0, 6.0]

def main():
    print("=======================================================")
    print("Starting Job B: Extended Rise-Stage Sweep on Mistral-7B-Instruct-v0.3")
    print(f"Output Directory: {OUT_DIR}")
    print("Fall-stage fixed at theta_hi = 99.0 (No fall-stage gating)")
    print("=======================================================")

    configs = []
    for th_lo in THETA_LO_LIST:
        for k_l in K_LO_LIST:
            configs.append((th_lo, k_l))

    print(f"Total Rise-Stage Configurations to Evaluate: {len(configs)}")

    for idx, (th_lo, k_l) in enumerate(configs, 1):
        config_name = f"rise-tlo{th_lo:.1f}-klo{k_l:.1f}"
        print(f"\n==========================================")
        print(f"[{idx}/{len(configs)}] Running Config: {config_name}")
        print(f"Model: Mistral-7B-Instruct-v0.3 | theta_lo={th_lo}, k_lo={k_l}")
        print(f"==========================================")

        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_99.0_k_{k_l:.1f}_1.0_entropy_plateau_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_99.0_k_{k_l}_1.0_entropy_plateau_Val5.0.csv"
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
                "--theta_lo", str(th_lo),
                "--theta_hi", "99.0",
                "--k_lo", str(k_l),
                "--k_hi", "1.0",
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
    print("Job B: Extended Rise-Stage Sweep Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
