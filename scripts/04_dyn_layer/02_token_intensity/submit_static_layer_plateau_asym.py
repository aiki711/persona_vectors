#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/submit_static_layer_plateau_asym.py
#

import os
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
JOBS_DIR = WORKSPACE / "jobs/static_layer_plateau_asym"
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_static_layer_plateau_asym/results"
LOG_DIR = WORKSPACE / "log/static_layer_plateau_asym"

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

# 9 Configs: (conf_id, theta_lo, theta_hi, k_lo, k_hi, gating_mode)
CONFIGS = [
    ("p_conf2", 3.0, 7.0, 2.0, 2.0, "plateau"),
    ("p_conf3", 1.0, 9.0, 2.0, 2.0, "plateau"),
    ("p_conf4", 4.0, 6.0, 2.0, 2.0, "plateau"),
    ("p_conf5", 3.0, 7.0, 8.0, 8.0, "plateau"),
    ("p_conf6", 3.0, 7.0, 0.5, 0.5, "plateau"),
    ("a_conf1", 3.0, 7.0, 0.5, 8.0, "max_normalized"),
    ("a_conf2", 3.0, 7.0, 8.0, 0.5, "max_normalized"),
    ("a_conf3", 1.0, 5.0, 1.0, 4.0, "max_normalized"),
    ("a_conf4", 5.0, 9.0, 4.0, 1.0, "max_normalized"),
]

TEMPLATE_HEADER = """#!/bin/bash
#SBATCH --job-name=sl_pa_{prefix}_{trait}
#SBATCH --output={log_dir}/sl_pa_{prefix}_{trait}.log
#SBATCH --error={log_dir}/sl_pa_{prefix}_{trait}.err
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=4:00:00

cd {workspace}
source persona_steering/bin/activate
"""

TEMPLATE_STEP = """
# Gating configuration: {conf_id} (theta: {theta_lo}-{theta_hi}, k: {k_lo}-{k_hi}, mode: {gating_mode})
echo "=========================================="
echo "Starting configuration {conf_id}..."
echo "=========================================="

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
    --gating_mode {gating_mode} \\
    --num_prompts 10 \\
    --static_layer {mask_bank_arg}

python scripts/04_dyn_layer/02_token_intensity/batch_eval.py \\
    --file {out_dir}/{trait}/{out_filename} \\
    --axis {trait} \\
    --quant 4bit
"""

def main():
    print("Generating Grouped Plateau & Asymmetric SLURM submit scripts (4 hours walltime)...")
    for trait in TRAITS:
        for (score_mode, use_mask, prefix) in METHODS:
            script_path = JOBS_DIR / f"run_sl_pa_{prefix}_{trait}.sh"
            
            # Start script with SBATCH header
            content = TEMPLATE_HEADER.format(
                prefix=prefix,
                trait=trait,
                log_dir=LOG_DIR,
                workspace=WORKSPACE
            )
            
            # Add steps for each of the 9 configs
            mask_bank_arg = "--mask_bank vectors/soft_probe_masks.npz" if use_mask else ""
            for (conf_id, theta_lo, theta_hi, k_lo, k_hi, gating_mode) in CONFIGS:
                suffix = ""
                if gating_mode == "max_normalized":
                    suffix = "_max_norm"
                elif gating_mode == "plateau":
                    suffix = "_plateau"
                out_filename = f"{prefix}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}{suffix}_Val5.0.jsonl"
                
                content += TEMPLATE_STEP.format(
                    conf_id=conf_id,
                    theta_lo=theta_lo,
                    theta_hi=theta_hi,
                    k_lo=k_lo,
                    k_hi=k_hi,
                    gating_mode=gating_mode,
                    out_dir=OUT_DIR,
                    trait=trait,
                    score_mode=score_mode,
                    mask_bank_arg=mask_bank_arg,
                    out_filename=out_filename
                )
                
            with open(script_path, "w") as f:
                f.write(content)
            
            # Make executable
            script_path.chmod(0o755)
            
    print(f"Generated 15 grouped SLURM submit scripts in {JOBS_DIR}")
    print("To submit a job, run: sbatch jobs/static_layer_plateau_asym/run_sl_pa_<prefix>_<trait>.sh")

if __name__ == "__main__":
    main()
