#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/aggregate_step1.py
#

import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil

def main():
    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    out_base_dir = Path("exp_token_intensity/exp_dual_gating")
    
    t_lo = 1.2
    # Include all completed theta_hi values in step 1
    theta_hi_vals = [5.0, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.5]
    k_lo = 8.0
    k_hi = 8.0
    
    results = []
    
    for t_hi in theta_hi_vals:
        trait_scores = []
        trait_ppls = []
        for trait in traits:
            csv_path = out_base_dir / trait / f"scores_masked_proj_rank_theta_{t_lo}_{t_hi}_k_{k_lo}_{k_hi}_dual_Val5.0.csv"
            # Fallback if float formatting
            if not csv_path.exists():
                csv_path_alt = out_base_dir / trait / f"scores_masked_proj_rank_theta_{t_lo}_{float(t_hi)}_k_{k_lo}_{k_hi}_dual_Val5.0.csv"
                if csv_path_alt.exists():
                    csv_path = csv_path_alt
                    
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    trait_scores.append(df['dyn_score'].mean())
                    trait_ppls.append(df['dyn_ppl'].mean())
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
            else:
                pass
                
        if len(trait_scores) == 5:
            avg_score = np.mean(trait_scores)
            avg_ppl = np.mean(trait_ppls)
            results.append({
                "theta_hi": t_hi,
                "score": avg_score,
                "ppl": avg_ppl,
                "scores": trait_scores,
                "ppls": trait_ppls
            })
            
    # Write summary report
    report_path = out_base_dir / "coordinate_step1_summary.md"
    md_lines = [
        "# Coordinate Descent Optimization: Step 1 (Fixing theta_H = 1.2)",
        "\nThis report presents the fine-grained tuning of the surprisal threshold $\\theta_{IC}$ to optimize PPL while keeping $\\theta_H = 1.2$.\n",
        "## 1. Step 1 Sweep Results (theta_H = 1.2)\n",
        "| Surprisal Threshold (theta_IC) | Alignment Score (Target: >4.34) | Perplexity (PPL) (Target: <10.46) | Result |",
        "| :---: | :---: | :---: | :--- |"
    ]
    
    best_config = None
    best_score_above_baseline = -1
    best_ppl = 999.0
    
    # Sort results by theta_hi
    results_sorted = sorted(results, key=lambda x: x["theta_hi"])
    
    for r in results_sorted:
        status = ""
        if r["score"] >= 4.34 and r["ppl"] < 10.46:
            status = "🏆 Dominated No Gating on BOTH!"
            if r["score"] > best_score_above_baseline or (r["score"] == best_score_above_baseline and r["ppl"] < best_ppl):
                best_score_above_baseline = r["score"]
                best_ppl = r["ppl"]
                best_config = r
        elif r["score"] >= 4.20:
            status = "⭐ Highly Efficient"
            
        md_lines.append(f"| **{r['theta_hi']:.1f}** | **{r['score']:.3f}** | **{r['ppl']:.3f}** | {status} |")
        
    if best_config:
        md_lines.append(f"\n### Recommended Optimal theta_IC* from Step 1: **{best_config['theta_hi']:.1f}** (Score={best_config['score']:.3f}, PPL={best_config['ppl']:.3f})")
    else:
        # If no config dominated No Gating on both, choose the one with best trade-off
        # e.g. highest score above 4.30 with lowest PPL
        best_tradeoff = sorted(results, key=lambda x: (-x["score"], x["ppl"]))[0]
        md_lines.append(f"\n### Recommended Trade-off theta_IC* from Step 1: **{best_tradeoff['theta_hi']:.1f}** (Score={best_tradeoff['score']:.3f}, PPL={best_tradeoff['ppl']:.3f})")
    
    md_lines.append("\n## 2. Trait Breakdown Table (Scores / PPL)\n")
    md_lines.append("| Configuration | Extraversion | Neuroticism | Openness | Conscientiousness | Agreeableness |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results_sorted:
        scores = r["scores"]
        ppls = r["ppls"]
        md_lines.append(f"| **Dual-1.2-{r['theta_hi']:.1f}** | {scores[0]:.2f} / {ppls[0]:.1f} | {scores[1]:.2f} / {ppls[1]:.1f} | {scores[2]:.2f} / {ppls[2]:.1f} | {scores[3]:.2f} / {ppls[3]:.1f} | {scores[4]:.2f} / {ppls[4]:.1f} |")
        
    md_text = "\n".join(md_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Aggregated {len(results)} configurations.")
    print(f"Saved Step 1 report to: {report_path}")
    
    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    shutil.copy(report_path, artifact_dir / "coordinate_step1_summary.md")
    print("Copied report to artifacts.")
    
    print("\n" + md_text)

if __name__ == "__main__":
    main()
