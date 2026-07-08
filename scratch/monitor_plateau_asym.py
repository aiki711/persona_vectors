#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/monitor_plateau_asym.py
#

import subprocess
import time
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_static_layer_plateau_asym/results"

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
METHODS = ["proj_rank", "masked_proj_cosine", "masked_proj_rank"]

CONFIGS = [
    # (conf_id, theta_lo, theta_hi, k_lo, k_hi, suffix)
    ("p_conf2", 3.0, 7.0, 2.0, 2.0, "_plateau"),
    ("p_conf3", 1.0, 9.0, 2.0, 2.0, "_plateau"),
    ("p_conf4", 4.0, 6.0, 2.0, 2.0, "_plateau"),
    ("p_conf5", 3.0, 7.0, 8.0, 8.0, "_plateau"),
    ("p_conf6", 3.0, 7.0, 0.5, 0.5, "_plateau"),
    ("a_conf1", 3.0, 7.0, 0.5, 8.0, "_max_norm"),
    ("a_conf2", 3.0, 7.0, 8.0, 0.5, "_max_norm"),
    ("a_conf3", 1.0, 5.0, 1.0, 4.0, "_max_norm"),
    ("a_conf4", 5.0, 9.0, 4.0, 1.0, "_max_norm"),
]

def count_completed():
    count = 0
    for trait in TRAITS:
        for prefix in METHODS:
            for (conf_id, theta_lo, theta_hi, k_lo, k_hi, suffix) in CONFIGS:
                csv_name = f"scores_{prefix}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}{suffix}_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name
                if csv_path.exists():
                    count += 1
    return count

def get_queue_count():
    try:
        res = subprocess.run(["squeue", "-u", "s2550009"], capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            if len(lines) > 1:
                return len(lines) - 1
    except subprocess.TimeoutExpired:
        print("squeue command timed out after 15 seconds.", flush=True)
    except Exception as e:
        print(f"Error checking queue: {e}", flush=True)
    return 0

def main():
    print("Starting background queue manager for Plateau and Asymmetric Gating configurations...", flush=True)
    total_expected = 135
    
    while True:
        completed = count_completed()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed: {completed}/{total_expected}", flush=True)
        
        if completed >= total_expected:
            print("All 135 configurations finished and evaluated! Generating final bar charts...", flush=True)
            res = subprocess.run(["python", str(WORKSPACE / "scripts/04_dyn_layer/02_token_intensity/plot_plateau_asym.py")], capture_output=True, text=True)
            print(res.stdout, flush=True)
            if res.stderr:
                print("[ERROR] Plotting script error:", flush=True)
                print(res.stderr, flush=True)
            break
            
        queue_count = get_queue_count()
        print(f"Current jobs in SLURM queue: {queue_count}", flush=True)
        
        # Poll every 60 seconds
        time.sleep(60)

if __name__ == "__main__":
    main()
