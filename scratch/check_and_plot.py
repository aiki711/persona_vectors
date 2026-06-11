#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# scratch/check_and_plot.py
#
# Checks SLURM job statuses and automatically runs the plotter script
# when new results are generated.
#

import subprocess
import sys
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
RESULTS_DIR = Path("exp_steering_layer_midpoint_norm/results")
FIGURES_DIR = Path("exp_steering_layer_midpoint_norm/figures")

def get_active_jobs():
    try:
        res = subprocess.run(["squeue", "-u", "s2550009", "-h", "-o", "%i %j"], capture_output=True, text=True)
        jobs = []
        for line in res.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                job_id, job_name = parts[0], parts[1]
                if "midpoint_norm" in job_name:
                    jobs.append(job_id)
        return jobs
    except Exception as e:
        print(f"Error calling squeue: {e}")
        return []

def count_results():
    if not RESULTS_DIR.exists():
        return 0
    csv_files = list(RESULTS_DIR.glob("**/scores_layer_*.csv"))
    return len(csv_files)

def main():
    print("=== SLURM Job Monitor & Plotter ===")
    
    # 1. Check active jobs
    our_active_jobs = get_active_jobs()
    
    print(f"Our monitored jobs in queue: {our_active_jobs}")
    
    # 2. Count generated csv files
    csv_count = count_results()
    print(f"Number of generated scores files (CSV): {csv_count} / 1440")
    
    # 3. If we have new CSVs, let's run the plotting script
    if csv_count > 0:
        print("Running heatmap plotter script...")
        plotter_cmd = [
            "/home/s2550009/persona_vectors/persona_steering/bin/python",
            "scripts/04_dyn_layer/108_plot_online_norm_layer_heatmap.py",
            "--artifact_dir", "/home/s2550009/.gemini/antigravity-ide/brain/967cd169-1aa5-48db-a243-174e45692380/images"
        ]
        res = subprocess.run(plotter_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(f"Plotter errors/warnings:\n{res.stderr}")
            
    # 4. Determine status
    if len(our_active_jobs) == 0:
        if csv_count >= 1440:
            print("STATUS: ALL_DONE")
            sys.exit(0)
        elif csv_count > 0:
            print("STATUS: JOBS_FINISHED_PARTIAL_DATA")
            sys.exit(0)
        else:   
            print("STATUS: JOBS_NOT_STARTED_OR_FAILED")
            sys.exit(1)
    else:
        print("STATUS: RUNNING_OR_PENDING")
        sys.exit(2)

if __name__ == "__main__":
    main()
