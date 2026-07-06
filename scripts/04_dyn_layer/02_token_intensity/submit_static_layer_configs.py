#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/submit_static_layer_configs.py
#

import os
import subprocess
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
JOBS_DIR = WORKSPACE / "jobs/static_layer"
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_static_layer/results"
LOG_DIR = WORKSPACE / "log/static_layer"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

# 3 Methods: (score_mode, use_mask, prefix)
METHODS = [
    ("proj_rank", False, "proj_rank"),
    ("proj_cosine", True, "masked_proj_cosine"),
    ("proj_rank", True, "masked_proj_rank")
]

# 6 Configs: (conf_id, theta_lo, theta_hi, k_lo, k_hi)
CONFIGS = [
    ("conf1", 0.0, 99.0, 1.0, 1.0),
    ("conf2", 3.0, 7.0, 2.0, 2.0),
    ("conf3", 1.0, 9.0, 2.0, 2.0),
    ("conf4", 4.0, 6.0, 2.0, 2.0),
    ("conf5", 3.0, 7.0, 8.0, 8.0),
    ("conf6", 3.0, 7.0, 0.5, 0.5)
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=sl_{prefix}_{conf_id}_{trait}
#SBATCH --output={log_dir}/sl_{prefix}_{conf_id}_{trait}.log
#SBATCH --error={log_dir}/sl_{prefix}_{conf_id}_{trait}.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=0:30:00

cd {workspace}
source persona_steering/bin/activate

# 1. Run generation with --static_layer
python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py \\
    --config configs/mistral_7b.yaml \\
    --vector_bank vectors/mean_diff_vectors.npz \\
    --prompts inputs/eval_prompts_10.jsonl \\
    --out_dir {out_dir} \\
    --axis {trait} \\
    --alpha_max 5.0 \\
    --score_mode {score_mode} \\
    --theta_lo {theta_lo} \\
    --theta_hi {theta_hi} \\
    --k_lo {k_lo} \\
    --k_hi {k_hi} \\
    --gating_mode standard \\
    --num_prompts 10 \\
    --static_layer {mask_bank_arg}

# 2. Run judge evaluation immediately
python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \\
    --file {out_dir}/{trait}/{out_filename} \\
    --axis {trait} \\
    --quant 4bit
"""

def main():
    print("Generating SLURM submit scripts for Static Layer configs...")
    job_files = []
    
    for trait in TRAITS:
        for (score_mode, use_mask, prefix) in METHODS:
            for (conf_id, theta_lo, theta_hi, k_lo, k_hi) in CONFIGS:
                
                # Check output filename
                out_filename = f"{prefix}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}_Val5.0.jsonl"
                out_file = OUT_DIR / trait / out_filename
                
                # Check if final score CSV already exists
                csv_filename = f"scores_{prefix}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}_Val5.0.csv"
                csv_file = OUT_DIR / trait / csv_filename
                if csv_file.exists():
                    continue
                
                mask_bank_arg = "--mask_bank vectors/soft_probe_masks.npz" if use_mask else ""
                
                script_content = TEMPLATE.format(
                    workspace=WORKSPACE,
                    log_dir=LOG_DIR,
                    out_dir=OUT_DIR,
                    trait=trait,
                    prefix=prefix,
                    conf_id=conf_id,
                    score_mode=score_mode,
                    theta_lo=theta_lo,
                    theta_hi=theta_hi,
                    k_lo=k_lo,
                    k_hi=k_hi,
                    mask_bank_arg=mask_bank_arg,
                    out_filename=out_filename
                )
                
                script_path = JOBS_DIR / f"run_sl_{prefix}_{conf_id}_{trait}.sh"
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)
                job_files.append(script_path)
                
    print(f"Generated {len(job_files)} submit scripts.")
    
    # Submit jobs in batches to avoid hitting partition array limits
    max_submit = 25
    submitted_count = 0
    
    for sf in job_files:
        if submitted_count >= max_submit:
            print(f"Limit of {max_submit} concurrent submissions reached. Please re-run later to submit remaining jobs.")
            break
            
        res = subprocess.run(["sbatch", str(sf)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Submitted: {sf.name} -> {res.stdout.strip()}")
            submitted_count += 1
        else:
            print(f"[ERROR] Failed to submit {sf.name}: {res.stderr}")
            
    print(f"Submission batch finished. Submitted {submitted_count} jobs.")

if __name__ == "__main__":
    main()
