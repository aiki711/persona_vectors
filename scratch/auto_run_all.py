#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/auto_run_all.py
#

import subprocess
import time
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_static_layer/results"

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

# 3 Methods
METHODS = ["proj_rank", "masked_proj_cosine", "masked_proj_rank"]

# 6 Configs: (conf_id, theta_lo, theta_hi, k_lo, k_hi)
CONFIGS = [
    ("conf1", 0.0, 99.0, 1.0, 1.0),
    ("conf2", 3.0, 7.0, 2.0, 2.0),
    ("conf3", 1.0, 9.0, 2.0, 2.0),
    ("conf4", 4.0, 6.0, 2.0, 2.0),
    ("conf5", 3.0, 7.0, 8.0, 8.0),
    ("conf6", 3.0, 7.0, 0.5, 0.5)
]

def count_completed():
    count = 0
    for trait in TRAITS:
        for prefix in METHODS:
            for (conf_id, theta_lo, theta_hi, k_lo, k_hi) in CONFIGS:
                csv_name = f"scores_{prefix}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name
                if csv_path.exists():
                    count += 1
    return count

def get_queue_count():
    try:
        res = subprocess.run(["squeue", "-u", "s2550009"], capture_output=True, text=True)
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            # subtract the header line if present
            if len(lines) > 1:
                return len(lines) - 1
    except Exception as e:
        print(f"Error checking queue: {e}")
    return 0

def main():
    print("Starting background queue manager for Static Layer configurations...")
    total_expected = 90
    
    while True:
        completed = count_completed()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed: {completed}/{total_expected}")
        
        if completed >= total_expected:
            print("All 90 configurations finished and evaluated! Generating final bar charts...")
            res = subprocess.run(["python", str(WORKSPACE / "scripts/04_dyn_layer/02_token_intensity/plot_static_layer_configs.py")], capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print("[ERROR] Plotting script error:")
                print(res.stderr)
            break
            
        queue_count = get_queue_count()
        print(f"Current jobs in SLURM queue: {queue_count}")
        
        if queue_count < 20:
            print("Queue has available slots. Submitting next batch...")
            res = subprocess.run(["python", str(WORKSPACE / "scripts/04_dyn_layer/02_token_intensity/submit_static_layer_configs.py")], capture_output=True, text=True)
            print(res.stdout)
            
        # Poll every 60 seconds
        time.sleep(60)

if __name__ == "__main__":
    main()
