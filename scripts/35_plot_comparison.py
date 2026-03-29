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

    # Pattern: scores_adaptive_{trait}_{method}_T{tau}_L15.csv
    files = glob.glob(str(input_dir / "scores_adaptive_*_L15.csv"))
    
    data = []
    for f in files:
        fname = Path(f).name
        # Match pattern like scores_adaptive_extraversion_svm_T3.0_L15.csv
        match = re.search(r"scores_adaptive_(.+?)_(svm|mean_diff)_T([\d\.]+)_L15\.csv", fname)
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
    
    # 1. Plot overall trade-off (All traits averaged)
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Average across traits for each (method, tau)
    df_avg = df_all.groupby(["method", "tau"]).agg({
        "score": "mean",
        "ppl": "mean"
    }).reset_index()
    
    for method in df_avg["method"].unique():
        subset = df_avg[df_avg["method"] == method].sort_values("tau")
        plt.plot(subset["score"], subset["ppl"], marker="o", label=f"Method: {method.upper()}")
        for i, row in subset.iterrows():
            plt.annotate(f"T={row['tau']}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel("Personality Score (1-5)")
    plt.ylabel("Perplexity (Lower is better)")
    plt.title("Granular Sweep: Personality vs Coherence Trade-off")
    plt.legend()
    
    out_path = out_dir / "granular_sweep_comparison_all.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved comparison plot to {out_path}")

    # 2. Per-trait plots
    for trait in df_all["trait"].unique():
        plt.figure(figsize=(10, 6))
        subset_trait = df_all[df_all["trait"] == trait]
        for method in subset_trait["method"].unique():
            subset = subset_trait[subset_trait["method"] == method].sort_values("tau")
            plt.errorbar(subset["score"], subset["ppl"], xerr=subset["score_sem"], yerr=subset["ppl_sem"], 
                         marker="o", label=f"{method.upper()}")
            for i, row in subset.iterrows():
                plt.annotate(f"{row['tau']:.1f}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(0,5), fontsize=8)

        plt.xlabel("Score")
        plt.ylabel("PPL")
        plt.title(f"Trade-off Sweep: {trait.capitalize()}")
        plt.legend()
        plt.savefig(out_dir / f"granular_sweep_{trait}.png")
        plt.close()

if __name__ == "__main__":
    main()
