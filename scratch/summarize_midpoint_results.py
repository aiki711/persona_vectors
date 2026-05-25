import pandas as pd
import numpy as np
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

fusion_dir = Path("exp_steering_dyn_ic_fusion_midpoint/results")
baseline_dir = Path("exp_steering_dyn_layer_all_layers_midpoint/results")

methods = ["logit_diff", "anti_alignment", "fixed", "sigmoid", "soft_plateau"]

summary_scores = {m: [] for m in methods}
summary_ppls   = {m: [] for m in methods}

for val in VALS:
    for m in methods:
        scores = []
        ppls = []
        for trait in TRAITS:
            if m in ["logit_diff", "anti_alignment"]:
                csv_path = baseline_dir / trait / f"scores_{m}_Val{float(val)}.csv"
                if not csv_path.exists():
                    csv_path = baseline_dir / trait / f"scores_{m}_Val{val}.csv"
            else:
                csv_path = fusion_dir / trait / f"scores_fusion_{m}_Val{float(val)}.csv"
                if not csv_path.exists():
                    csv_path = fusion_dir / trait / f"scores_fusion_{m}_Val{val}.csv"
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    scores.append(df["dyn_score"].mean())
                    ppls.append(df["dyn_ppl"].mean())
                except Exception as e:
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
lines.append("| AlphaMax | Logit Diff | Anti-Alignment | DLS (Fixed) | Fusion Sigmoid | Fusion Soft-Plateau |")
lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
for i, val in enumerate(VALS):
    row = f"| **{val}**"
    for m in methods:
        val_score = summary_scores[m][i]
        val_str = f"{val_score:.2f}" if not np.isnan(val_score) else "N/A"
        row += f" | {val_str}"
    row += " |"
    lines.append(row)

lines.append("")
lines.append("### Average Perplexities")
lines.append("| AlphaMax | Logit Diff | Anti-Alignment | DLS (Fixed) | Fusion Sigmoid | Fusion Soft-Plateau |")
lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
for i, val in enumerate(VALS):
    row = f"| **{val}**"
    for m in methods:
        val_ppl = summary_ppls[m][i]
        if np.isnan(val_ppl):
            row += " | N/A"
        else:
            val_str = f"{val_ppl:.2f}"
            if val_ppl <= 25.0:
                row += f" | **{val_str} (Safe)**"
            else:
                row += f" | {val_str}"
    row += " |"
    lines.append(row)

markdown_content = "\n".join(lines) + "\n"
with open("scratch/midpoint_summary.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(markdown_content)
