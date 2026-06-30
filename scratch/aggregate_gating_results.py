#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/aggregate_gating_results.py
#

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("exp_token_intensity/results")
TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

CONFIGS = [
    ("conf1", "No Gating (0-99)", "0.0", "99.0", "1.0", "1.0"),
    ("conf2", "Base Gating (3-7)", "3.0", "7.0", "2.0", "2.0"),
    ("conf3", "Wider (1-9)", "1.0", "9.0", "2.0", "2.0"),
    ("conf4", "Narrower (4-6)", "4.0", "6.0", "2.0", "2.0"),
    ("conf5", "Sharp (k=8)", "3.0", "7.0", "8.0", "8.0"),
    ("conf6", "Gentle (k=0.5)", "3.0", "7.0", "0.5", "0.5"),
]

METHODS = [
    ("proj_rank", "Proj Rank (Unmasked)"),
    ("masked_proj_rank", "PDF Proj Rank (Soft Masked)"),
]

def load_metrics(trait: str, score_mode: str, conf_params: tuple) -> tuple[float, float, float]:
    theta_lo, theta_hi, k_lo, k_hi = conf_params
    csv_name = f"scores_{score_mode}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}_Val5.0.csv"
    csv_path = RESULTS_DIR / trait / csv_name
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            score_col = "dyn_score"
            ppl_col = "dyn_ppl"
            reason_col = "dyn_reason"
            
            mean_score = df[score_col].mean()
            valid_ppl = df[ppl_col][np.isfinite(df[ppl_col])]
            mean_ppl = valid_ppl.mean() if not valid_ppl.empty else 999.0
            
            coherence_rate = df[reason_col].str.contains("Coherence: Yes", case=False, na=False).mean() if reason_col in df.columns else 1.0
            return mean_score, mean_ppl, coherence_rate
        except Exception as e:
            pass
    return 0.0, 999.0, 0.0

def main():
    print("# Surprisal Gating (DLIS) Aggregated Results\n")
    
    # We will build two tables: one for Proj Rank (Unmasked) and one for PDF Proj Rank (Soft Masked)
    for score_mode, method_name in METHODS:
        print(f"### {method_name}")
        print("| Configuration | Extraversion (Score/PPL/Coh) | Neuroticism | Openness | Conscientiousness | Agreeableness | Average |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        
        for conf_id, conf_name, theta_lo, theta_hi, k_lo, k_hi in CONFIGS:
            row_vals = []
            scores = []
            ppls = []
            cohs = []
            
            for trait in TRAITS:
                s, p, c = load_metrics(trait, score_mode, (theta_lo, theta_hi, k_lo, k_hi))
                row_vals.append(f"{s:.2f} / {p:.1f} / {c:.2f}")
                scores.append(s)
                ppls.append(p)
                cohs.append(c)
                
            avg_s = np.mean(scores)
            avg_p = np.mean(ppls)
            avg_c = np.mean(cohs)
            row_vals.append(f"**{avg_s:.2f} / {avg_p:.1f} / {avg_c:.2f}**")
            
            print(f"| **{conf_name}** | " + " | ".join(row_vals) + " |")
        print()

if __name__ == "__main__":
    main()
