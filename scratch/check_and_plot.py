#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/check_and_plot.py
#

import subprocess
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
RESULTS_DIR = WORKSPACE / "exp_token_intensity/exp_symmetric/results"

def main():
    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    alphas = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    missing = []
    found_count = 0
    for trait in traits:
        for alpha in alphas:
            csv_name = f"scores_masked_proj_rank_theta_0.0_99.0_k_1.0_1.0_Val{alpha}.csv"
            csv_path = RESULTS_DIR / trait / csv_name
            if not csv_path.exists():
                # Fallback to integer representation
                csv_name_int = f"scores_masked_proj_rank_theta_0.0_99.0_k_1.0_1.0_Val{int(alpha)}.csv"
                csv_path_int = RESULTS_DIR / trait / csv_name_int
                if not csv_path_int.exists():
                    missing.append((trait, alpha))
                else:
                    found_count += 1
            else:
                found_count += 1
                
    print(f"Status: Found {found_count}/35 evaluation CSV files. Missing {len(missing)} files.")
    
    # Run the plotting script
    print("Updating plots...")
    res = subprocess.run(["python", str(WORKSPACE / "scripts/04_dyn_layer/02_token_intensity/plot_high_intensity.py")], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[ERROR] Plotting script error output:")
        print(res.stderr)

if __name__ == "__main__":
    main()
