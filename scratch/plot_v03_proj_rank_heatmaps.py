#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_v03_proj_rank_heatmaps.py
# Generate per-trait and summary heatmaps for proj_rank Rise & Fall sweeps
#

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/s2550009/persona_vectors/exp_token_intensity/exp_v03_proj_rank_sweeps")
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/3f7b9818-2c63-474f-b2e3-53654250dd23")

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

THETA_LO_LIST = [0.1, 0.2, 0.3, 0.4, 0.6]
K_LO_LIST = [2.0, 3.0, 4.0, 5.0, 6.0]

THETA_HI_LIST = [1.5, 2.0, 2.5, 3.0, 3.5]
K_HI_LIST = [0.05, 0.1, 0.2, 0.3, 0.4]

def parse_sweep_data(stage="rise"):
    records = []
    for trait in TRAITS:
        t_dir = BASE_DIR / trait
        if not t_dir.exists(): continue
        for csv_file in t_dir.glob("scores_masked_proj_rank_*.csv"):
            parts = csv_file.name.replace(".csv", "").split("_")
            try:
                t_lo = float(parts[5])
                t_hi = float(parts[6])
                k_lo = float(parts[8])
                k_hi = float(parts[9])
                
                df = pd.read_csv(csv_file)
                score = df["dyn_score"].mean()
                ppl = df["dyn_ppl"][pd.Series(df["dyn_ppl"]).notna()].mean()

                if stage == "rise" and t_hi == 99.0 and k_hi == 1.0 and t_lo in THETA_LO_LIST and k_lo in K_LO_LIST:
                    records.append({"trait": trait, "theta": t_lo, "k": k_lo, "score": score, "ppl": ppl})
                elif stage == "fall" and t_lo == 0.0 and k_lo == 1.0 and t_hi in THETA_HI_LIST and k_hi in K_HI_LIST:
                    records.append({"trait": trait, "theta": t_hi, "k": k_hi, "score": score, "ppl": ppl})
            except Exception as e:
                continue
    return pd.DataFrame(records)

def plot_grid(df, stage="rise", metric="score"):
    if df.empty:
        print(f"No data for {stage} ({metric})")
        return
    
    y_vals = THETA_LO_LIST if stage == "rise" else THETA_HI_LIST
    x_vals = K_LO_LIST if stage == "rise" else K_HI_LIST

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, trait in enumerate(TRAITS):
        ax = axes[idx]
        sub = df[df["trait"] == trait]
        if sub.empty: continue
        pivot = sub.pivot(index="theta", columns="k", values=metric).reindex(index=y_vals, columns=x_vals)
        
        fmt = ".2f"
        cmap = "YlGnBu" if metric == "score" else "YlOrRd"
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar=False)
        ax.set_title(f"{trait.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"$\theta_{\mathrm{lo}}$" if stage == "rise" else r"$\theta_{\mathrm{hi}}$")
        ax.set_xlabel(r"$k_{\mathrm{lo}}$" if stage == "rise" else r"$k_{\mathrm{hi}}$")

    # Summary Plot in 6th subplot
    ax_sum = axes[5]
    summary_pivot = df.groupby(["theta", "k"])[metric].mean().unstack().reindex(index=y_vals, columns=x_vals)
    fmt = ".2f"
    cmap = "YlGnBu" if metric == "score" else "YlOrRd"
    sns.heatmap(summary_pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax_sum, cbar=False)
    ax_sum.set_title(f"5-Trait Mean ({metric.capitalize()})", fontsize=12, fontweight="bold", color="darkred")
    ax_sum.set_ylabel(r"$\theta_{\mathrm{lo}}$" if stage == "rise" else r"$\theta_{\mathrm{hi}}$")
    ax_sum.set_xlabel(r"$k_{\mathrm{lo}}$" if stage == "rise" else r"$k_{\mathrm{hi}}$")

    title_str = f"proj_rank Axis {stage.capitalize()} Dynamic Steering - {metric.upper()}"
    plt.suptitle(title_str, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_png = ARTIFACTS_DIR / f"v03_proj_rank_{stage}_traits_{metric}_grid.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")

def main():
    df_rise = parse_sweep_data("rise")
    if not df_rise.empty:
        plot_grid(df_rise, "rise", "score")
        plot_grid(df_rise, "rise", "ppl")

    df_fall = parse_sweep_data("fall")
    if not df_fall.empty:
        plot_grid(df_fall, "fall", "score")
        plot_grid(df_fall, "fall", "ppl")

if __name__ == "__main__":
    main()
