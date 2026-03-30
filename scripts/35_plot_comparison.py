#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 35_plot_comparison.py
#

import argparse
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="exp_adaptive_steering/results")
    ap.add_argument("--out_dir", default="exp_adaptive_steering/figures")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = glob.glob(str(input_dir / "scores_adaptive_*.csv"))
    
    data = []
    for f in files:
        fname = Path(f).name
        # Match pattern like scores_adaptive_neuroticism_high_L15_svm_T1.0.csv
        # Pattern components:
        # 1. scores_adaptive_
        # 2. {trait}
        # 3. _high_L15_
        # 4. {method} (svm or mean_diff)
        # 5. _T{tau}
        # 6. .csv
        match = re.search(r"scores_adaptive_(.+?)_high_L15_(svm|mean_diff)_T([\d\.]+)\.csv", fname)
        if not match:
            continue
        
        trait = match.group(1)
        method = match.group(2)
        tau = float(match.group(3))
        
        df = pd.read_csv(f)
        # Assuming df has base_score, const_score, adapt_score, base_ppl, const_ppl, adapt_ppl
        res = {
            "trait": trait,
            "method": method,
            "tau": tau,
            "score": df["adapt_score"].mean(),
            "ppl": df["adapt_ppl"].mean(),
            "score_sem": df["adapt_score"].sem(),
            "ppl_sem": df["adapt_ppl"].sem()
        }
        data.append(res)
        
    if not data:
        print("No matching results found.")
        return

    df_all = pd.DataFrame(data)
    
    # Consistent color mapping
    COLOR_MAP = {"svm": "tab:blue", "mean_diff": "tab:red"}
    
    # 1. Plot overall trade-off (All traits averaged)
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Average across traits for each (method, tau)
    df_avg = df_all.groupby(["method", "tau"]).agg({
        "score": "mean",
        "ppl": "mean"
    }).reset_index()
    
    for method in sorted(df_avg["method"].unique()):
        subset = df_avg[df_avg["method"] == method].sort_values("tau")
        color = COLOR_MAP.get(method, None)
        plt.plot(subset["score"], subset["ppl"], marker="o", label=f"METHOD: {method.upper()}", color=color)
        for i, row in subset.iterrows():
            plt.annotate(f"T={row['tau']}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.xlabel("Personality Score (1-5)")
    plt.ylabel("Perplexity (Lower is better)")
    plt.title("Granular Sweep: Personality vs Coherence Trade-off (Average)")
    plt.legend()
    
    out_path = out_dir / "granular_sweep_comparison_all.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved comparison plot to {out_path}")
    
    # Save a summary CSV for reading the numbers
    summary_csv = out_dir / "granular_sweep_summary.csv"
    df_avg.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")

    # 2. Per-trait plots
    for trait in sorted(df_all["trait"].unique()):
        plt.figure(figsize=(10, 6))
        subset_trait = df_all[df_all["trait"] == trait]
        for method in sorted(subset_trait["method"].unique()):
            subset = subset_trait[subset_trait["method"] == method].sort_values("tau")
            color = COLOR_MAP.get(method, None)
            plt.errorbar(subset["score"], subset["ppl"], xerr=subset["score_sem"], yerr=subset["ppl_sem"], 
                         marker="o", label=f"METHOD: {method.upper()}", color=color)
            for i, row in subset.iterrows():
                plt.annotate(f"{row['tau']:.1f}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(0,5), fontsize=8)

        plt.xlabel("Personality Score (1-5)")
        plt.ylabel("Perplexity (Lower is better)")
        plt.title(f"Trade-off Sweep: {trait.capitalize()}")
        plt.legend()
        plt.savefig(out_dir / f"granular_sweep_{trait}.png")
        plt.close()

if __name__ == "__main__":
    main()
