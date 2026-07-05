#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/submit_improved.py
#
# Generates and submits SLURM jobs for the improved gating methods (max_normalized and plateau).
# Saves all results to exp_token_intensity/results_improved/{trait}.
#

import argparse
import os
import subprocess
from pathlib import Path

# Paths
WORKSPACE = Path("/home/s2550009/persona_vectors")
JOBS_DIR = WORKSPACE / "jobs/02_token_intensity_asymmetric"
OUT_DIR = WORKSPACE / "exp_token_intensity/asymmetric/results"
LOG_DIR = WORKSPACE / "log/02_token_intensity_asymmetric"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHAS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
GATING_MODES = ["max_normalized", "plateau"]

# Gating config for Gentle Gating (Conf 6)
THETA_LO = 3.0
THETA_HI = 7.0
K_LO = 0.5
K_HI = 0.5

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dlis_asym_{trait}_{gating_mode}_a{alpha}
#SBATCH --output={log_dir}/dlis_asym_{trait}_{gating_mode}_a{alpha}.log
#SBATCH --error={log_dir}/dlis_asym_{trait}_{gating_mode}_a{alpha}.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=0:30:00

cd {workspace}
source persona_steering/bin/activate

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \\
    --config configs/mistral_7b.yaml \\
    --vector_bank vectors/mean_diff_vectors.npz \\
    --prompts inputs/eval_prompts_10.jsonl \\
    --out_dir {out_dir} \\
    --axis {trait} \\
    --alpha_max {alpha} \\
    --score_mode proj_rank \\
    --theta_lo {theta_lo} \\
    --theta_hi {theta_hi} \\
    --k_lo {k_lo} \\
    --k_hi {k_hi} \\
    --gating_mode {gating_mode} \\
    --num_prompts 10 \\
    --mask_bank vectors/soft_probe_masks.npz
"""

def main():
    print("Generating SLURM submit scripts for improved gating sweeps...")
    job_files = []
    
    for trait in TRAITS:
        for gating_mode in GATING_MODES:
            for alpha in ALPHAS:
                # Suffix and filename
                suffix = "max_norm" if gating_mode == "max_normalized" else "plateau"
                out_file = OUT_DIR / trait / f"masked_proj_rank_theta_{THETA_LO}_{THETA_HI}_k_{K_LO}_{K_HI}_{suffix}_Val{alpha}.jsonl"
                
                if out_file.exists():
                    continue
                
                # Write bash script
                script_content = TEMPLATE.format(
                    workspace=WORKSPACE,
                    log_dir=LOG_DIR,
                    out_dir=OUT_DIR,
                    trait=trait,
                    gating_mode=gating_mode,
                    alpha=alpha,
                    theta_lo=THETA_LO,
                    theta_hi=THETA_HI,
                    k_lo=K_LO,
                    k_hi=K_HI
                )
                
                script_path = JOBS_DIR / f"run_dlis_asym_{trait}_{gating_mode}_a{alpha}.sh"
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)
                job_files.append(script_path)

    print(f"Generated {len(job_files)} submit scripts.")
    if not job_files:
        print("All jobs already exist on disk.")
        return

    # Automatically submit
    for sf in job_files:
        res = subprocess.run(["sbatch", str(sf)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Submitted: {sf.name} -> {res.stdout.strip()}")
        else:
            print(f"[ERROR] Failed to submit {sf.name}: {res.stderr}")

if __name__ == "__main__":
    main()
