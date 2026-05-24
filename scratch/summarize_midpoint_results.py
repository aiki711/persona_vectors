import pandas as pd
import numpy as np
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS   = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0]
MODES  = ["fixed", "sigmoid", "soft_plateau"]

results_dir = Path("exp_steering_dyn_ic_fusion_midpoint/results")

# Summarize scores and perplexities
summary_scores = {m: [] for m in MODES}
summary_ppls   = {m: [] for m in MODES}

for val in VALS:
    for m in MODES:
        scores = []
        ppls = []
        for trait in TRAITS:
            csv_path = results_dir / trait / f"scores_fusion_{m}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = results_dir / trait / f"scores_fusion_{m}_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    scores.append(df["dyn_score"].mean())
                    ppls.append(df["dyn_ppl"].mean())
                except:
                    pass
        avg_score = np.mean(scores) if scores else float('nan')
        avg_ppl = np.mean(ppls) if ppls else float('nan')
        
        summary_scores[m].append(avg_score)
        summary_ppls[m].append(avg_ppl)

# Write markdown tables
lines = []
lines.append("## Midpoint Normalized Sweeps Summary (All Traits Avg)")
lines.append("")
lines.append("### Average Personality Scores")
lines.append("| Steering Strength Ratio ($\\alpha_{max}$) | DLS Midpoint (Fixed) | Fusion Sigmoid | Fusion Soft-Plateau |")
lines.append("| :---: | :---: | :---: | :---: |")
for i, val in enumerate(VALS):
    s_fix = f"{summary_scores['fixed'][i]:.2f}" if not np.isnan(summary_scores['fixed'][i]) else "N/A"
    s_sig = f"{summary_scores['sigmoid'][i]:.2f}" if not np.isnan(summary_scores['sigmoid'][i]) else "N/A"
    s_plat = f"{summary_scores['soft_plateau'][i]:.2f}" if not np.isnan(summary_scores['soft_plateau'][i]) else "N/A"
    lines.append(f"| **{val}** | {s_fix} | {s_sig} | {s_plat} |")

lines.append("")
lines.append("### Average Perplexities")
lines.append("| Steering Strength Ratio ($\\alpha_{max}$) | DLS Midpoint (Fixed) | Fusion Sigmoid | Fusion Soft-Plateau |")
lines.append("| :---: | :---: | :---: | :---: |")
for i, val in enumerate(VALS):
    p_fix = f"{summary_ppls['fixed'][i]:.2f}" if not np.isnan(summary_ppls['fixed'][i]) else "N/A"
    p_sig = f"{summary_ppls['sigmoid'][i]:.2f}" if not np.isnan(summary_ppls['sigmoid'][i]) else "N/A"
    p_plat = f"{summary_ppls['soft_plateau'][i]:.2f}" if not np.isnan(summary_ppls['soft_plateau'][i]) else "N/A"
    
    # mark safe perplexity
    fix_str = f"{p_fix} (Safe)" if not np.isnan(summary_ppls['fixed'][i]) and summary_ppls['fixed'][i] <= 25.0 else p_fix
    sig_str = f"{p_sig} (Safe)" if not np.isnan(summary_ppls['sigmoid'][i]) and summary_ppls['sigmoid'][i] <= 25.0 else p_sig
    plat_str = f"{p_plat} (Safe)" if not np.isnan(summary_ppls['soft_plateau'][i]) and summary_ppls['soft_plateau'][i] <= 25.0 else p_plat
    
    lines.append(f"| **{val}** | {fix_str} | {sig_str} | {plat_str} |")

with open("scratch/midpoint_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("DONE")
