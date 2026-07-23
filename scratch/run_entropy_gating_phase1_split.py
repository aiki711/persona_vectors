#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_entropy_gating_phase1_split.py
#

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def run_cmd(cmd):
    print(f"\nRunning command: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: Command failed with code {res.returncode}")
        print(f"Stdout:\n{res.stdout}")
        print(f"Stderr:\n{res.stderr}")
    else:
        print("Completed successfully.")
    return res

def main():
    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    out_base_dir = Path("exp_token_intensity/exp_entropy_gating")
    out_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Split Grid for Phase 1 (Rise stage - Back half)
    theta_lo_vals = [1.6, 1.8]
    k_lo_vals = [1.5, 4.0, 8.0]
    
    results = []
    
    total_runs = len(theta_lo_vals) * len(k_lo_vals)
    current_run = 0
    
    print(f"Starting Split Rise-Stage Entropy Gating Sweep ({total_runs} configurations)...")
    
    for t_lo in theta_lo_vals:
        for k_lo in k_lo_vals:
            current_run += 1
            config_name = f"Entropy-Rise-{t_lo}-k-{k_lo}"
            print(f"\n[{current_run}/{total_runs}] Running Config: {config_name}")
            
            t_hi = 7.0
            k_hi = 2.0
            
            # 1. Run generation for all 5 traits
            for trait in traits:
                cmd_gen = (
                    f"python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py "
                    f"--config configs/mistral_7b.yaml "
                    f"--vector_bank vectors/mean_diff_vectors.npz "
                    f"--prompts inputs/eval_prompts_10.jsonl "
                    f"--mask_bank vectors/soft_probe_masks.npz "
                    f"--out_dir {out_base_dir} "
                    f"--axis {trait} "
                    f"--alpha_max 5.0 "
                    f"--gating_mode entropy "
                    f"--static_layer "
                    f"--theta_lo {t_lo} --theta_hi {t_hi} "
                    f"--k_lo {k_lo} --k_hi {k_hi} "
                    f"--num_prompts 10"
                )
                run_cmd(cmd_gen)
                
            # 2. Run judge evaluation for all 5 traits
            for trait in traits:
                cmd_eval = (
                    f"python scripts/04_dyn_layer/02_token_intensity/batch_eval.py "
                    f"--results_dir {out_base_dir}/{trait} "
                    f"--axis {trait} "
                    f"--quant 4bit"
                )
                run_cmd(cmd_eval)
                
            # 3. Aggregate results (mostly dummy here, full report will be generated after both jobs finish)
            trait_scores = []
            trait_ppls = []
            for trait in traits:
                csv_path = out_base_dir / trait / f"scores_masked_proj_rank_theta_{t_lo}_{t_hi}_k_{k_lo}_{k_hi}_entropy_Val5.0.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        trait_scores.append(df['dyn_score'].mean())
                        trait_ppls.append(df['dyn_ppl'].mean())
                    except Exception as e:
                        print(f"Error loading {csv_path}: {e}")
                else:
                    print(f"Warning: CSV not found: {csv_path}")

if __name__ == "__main__":
    main()
