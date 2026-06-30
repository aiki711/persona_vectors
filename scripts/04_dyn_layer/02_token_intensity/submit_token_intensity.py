#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/submit_token_intensity.py
#
# Generates and submits SLURM jobs for token intensity surprisal gating sweep.
# Covers 5 traits, 6 configurations, and 2 methods (proj_rank and masked_proj_rank).
#

import os
import subprocess
from pathlib import Path

# Paths
WORKSPACE = Path("/home/s2550009/persona_vectors")
JOBS_DIR = WORKSPACE / "jobs/02_token_intensity"
OUT_DIR = WORKSPACE / "exp_token_intensity/results"
LOG_DIR = WORKSPACE / "log/02_token_intensity"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
CONFIGS = [
    # (id, name, theta_lo, theta_hi, k_lo, k_hi)
    ("conf1", "no_gating", 0.0, 99.0, 1.0, 1.0),
    ("conf2", "base_gating", 3.0, 7.0, 2.0, 2.0),
    ("conf3", "wider_gating", 1.0, 9.0, 2.0, 2.0),
    ("conf4", "narrower_gating", 4.0, 6.0, 2.0, 2.0),
    ("conf5", "sharp_gating", 3.0, 7.0, 8.0, 8.0),
    ("conf6", "gentle_gating", 3.0, 7.0, 0.5, 0.5),
]

METHODS = [
    ("proj_rank", ""), # (score_mode, mask_flag)
    ("masked_proj_rank", "--mask_bank vectors/soft_probe_masks.npz"),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dlis_{trait}_{conf_id}_{method}
#SBATCH --output={log_dir}/dlis_{trait}_{conf_id}_{method}.log
#SBATCH --error={log_dir}/dlis_{trait}_{conf_id}_{method}.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=0:30:00

cd {workspace}
source persona_steering/bin/activate

python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \\
    --config configs/mistral_7b.yaml \\
    --vector_bank vectors/vector_bank.npz \\
    --prompts inputs/alpaca_prompts_100.jsonl \\
    --out_dir {out_dir} \\
    --axis {trait} \\
    --alpha_max 5.0 \\
    --score_mode proj_rank \\
    --theta_lo {theta_lo} \\
    --theta_hi {theta_hi} \\
    --k_lo {k_lo} \\
    --k_hi {k_hi} \\
    --num_prompts 100 \\
    {mask_arg}
"""

def main():
    print("Generating SLURM submit scripts...")
    job_files = []
    
    for trait in TRAITS:
        for conf_id, name, theta_lo, theta_hi, k_lo, k_hi in CONFIGS:
            for m_name, mask_arg in METHODS:
                # Write bash script
                script_content = TEMPLATE.format(
                    workspace=WORKSPACE,
                    log_dir=LOG_DIR,
                    out_dir=OUT_DIR,
                    trait=trait,
                    conf_id=conf_id,
                    method=m_name,
                    theta_lo=theta_lo,
                    theta_hi=theta_hi,
                    k_lo=k_lo,
                    k_hi=k_hi,
                    mask_arg=mask_arg
                )
                
                script_path = JOBS_DIR / f"run_dlis_{trait}_{conf_id}_{m_name}.sh"
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)
                job_files.append(script_path)

    print(f"Generated {len(job_files)} submit scripts.")
    
    # Ask if we should submit
    ans = input("Do you want to submit all these jobs to SLURM now? (y/n): ")
    if ans.lower() == "y":
        print("Submitting jobs to SLURM...")
        for script_path in job_files:
            res = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"Submitted: {script_path.name} -> {res.stdout.strip()}")
            else:
                print(f"[ERROR] Failed to submit {script_path.name}: {res.stderr}")
    else:
        print("Jobs were generated but NOT submitted.")

if __name__ == "__main__":
    main()
