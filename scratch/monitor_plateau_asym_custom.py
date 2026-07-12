#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/monitor_plateau_asym_custom.py
#

import time
import subprocess
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
RESULTS_DIR = WORKSPACE / "exp_token_intensity/exp_static_layer_plateau_asym_custom/results"
PLOTTING_SCRIPT = WORKSPACE / "scripts/04_dyn_layer/02_token_intensity/plot_plateau_asym_custom.py"

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

METHODS = [
    "proj_rank",
    "masked_proj_cosine",
    "masked_proj_rank"
]

CONFIGS = [
    # (theta_lo, theta_hi, k_lo, k_hi, gating_mode)
    (2.0, 7.0, 1.0, 4.0, "max_normalized"),
    (1.0, 4.0, 0.5, 6.0, "max_normalized"),
    (2.0, 8.0, 0.8, 5.0, "max_normalized"),
    (2.0, 6.0, 0.5, 8.0, "plateau")
]

# Total target files = 5 traits * 3 methods * 4 configs = 60 files
TOTAL_EXPECTED_FILES = len(TRAITS) * len(METHODS) * len(CONFIGS)

def get_existing_files():
    count = 0
    existing = []
    
    for trait in TRAITS:
        for method in METHODS:
            for (theta_lo, theta_hi, k_lo, k_hi, gating_mode) in CONFIGS:
                suffix = ""
                if gating_mode == "max_normalized":
                    suffix = "_max_norm"
                elif gating_mode == "plateau":
                    suffix = "_plateau"
                    
                filename = f"scores_{method}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}{suffix}_Val5.0.csv"
                file_path = RESULTS_DIR / trait / filename
                if file_path.exists():
                    count += 1
                    existing.append(file_path)
    return count, existing

def main():
    print(f"Monitoring custom sweep results. Target files: {TOTAL_EXPECTED_FILES}")
    
    while True:
        # Check running jobs under SLURM with timeout to prevent NFS hang
        try:
            res = subprocess.run(["squeue", "-u", "s2550009"], capture_output=True, text=True, timeout=15)
            squeue_out = res.stdout
        except subprocess.TimeoutExpired:
            print("[Warning] squeue timed out. Continuing count check...")
            squeue_out = "NFS_FREEZE"
            
        count, _ = get_existing_files()
        
        # Check active jobs with custom prefix
        active_jobs = [line for line in squeue_out.splitlines() if "sl_pa_c" in line]
        
        print(f"[{time.strftime('%H:%M:%S')}] Detected files: {count}/{TOTAL_EXPECTED_FILES} | Running SLURM custom jobs: {len(active_jobs)}", flush=True)
        
        if count >= TOTAL_EXPECTED_FILES:
            print("All custom result files detected! Triggering plot generation...", flush=True)
            try:
                subprocess.run(["python", str(PLOTTING_SCRIPT)], check=True)
                print("Plot generation completed successfully!", flush=True)
            except Exception as e:
                print(f"Error running plotting script: {e}", flush=True)
            break
            
        if len(active_jobs) == 0 and squeue_out != "NFS_FREEZE":
            # Double check if any files were missed
            time.sleep(5)
            count, _ = get_existing_files()
            if count < TOTAL_EXPECTED_FILES:
                print(f"[Warning] No active custom jobs detected but only {count}/{TOTAL_EXPECTED_FILES} files exist.", flush=True)
                # We won't crash, we'll run the plot generation anyway for partial results
                print("Running plot script for available partial results...", flush=True)
                subprocess.run(["python", str(PLOTTING_SCRIPT)])
                break
            else:
                print("All files exist! Generating plots...", flush=True)
                subprocess.run(["python", str(PLOTTING_SCRIPT)])
                break
                
        time.sleep(30)

if __name__ == "__main__":
    main()
