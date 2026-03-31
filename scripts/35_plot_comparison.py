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
    
    patterns = [
        r"scores_adaptive_(.+?)_high_L15_stress_(svm|mean_diff)_T([\d\.]+)_A([\d\.]+)\.csv",
        r"scores_adaptive_(.+?)_high_L15_(svm|mean_diff)_T([\d\.]+)\.csv",
        r"scores_adaptive_(.+?)_T([\d\.]+)_A([\d\.]+)_L15\.csv"
    ]
    
    data = []
    for f in files:
        fname = Path(f).name
        matched = False
        trait, method, tau, alpha = None, None, None, None
        
        # 1. Stress Test Pattern
        m1 = re.search(r"scores_adaptive_(.+?)_high_L15_stress_(svm|mean_diff)_T([\d\.]+)_A([\d\.]+)\.csv", fname)
        if m1:
            trait, method, tau, alpha = m1.group(1), m1.group(2), float(m1.group(3)), float(m1.group(4))
            matched = True
            
        # 2. Standard Granular Sweep Pattern
        if not matched:
            m2 = re.search(r"scores_adaptive_(.+?)_high_L15_(svm|mean_diff)_T([\d\.]+)\.csv", fname)
            if m2:
                trait, method, tau, alpha = m2.group(1), m2.group(2), float(m2.group(3)), 15.0 # Default Alpha was usually 15 or 10
                matched = True
                
        # 3. Adaptive Param Sweep Pattern
        if not matched:
            m3 = re.search(r"scores_adaptive_(.+?)_T([\d\.]+)_A([\d\.]+)_L15\.csv", fname)
            if m3:
                trait, method, tau, alpha = m3.group(1), "svm", float(m3.group(2)), float(m3.group(3))
                matched = True
        
        if not matched:
            continue
        
        try:
            df = pd.read_csv(f)
            # Take means of adapt_score and adapt_ppl
            res = {
                "trait": trait,
                "method": method,
                "tau": tau,
                "alpha": alpha,
                "score": df["adapt_score"].mean(),
                "ppl": df["adapt_ppl"].mean(),
                "score_sem": df["adapt_score"].sem(),
                "ppl_sem": df["adapt_ppl"].sem()
            }
            data.append(res)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        
    if not data:
        print("No matching results found.")
        return

    df_all = pd.DataFrame(data)
    # Filter for tau <= 6.0 for better visualization of stable range
    df_all = df_all[df_all["tau"] <= 6.0]
    
    if df_all.empty:
        print("No results found with tau <= 6.0.")
        return
    COLOR_MAP = {"svm": "tab:blue", "mean_diff": "tab:red"}
    MARKERS = {15.0: "o", 30.0: "s", 50.0: "^", 10.0: "D"}
    
    # 1. Plot overall trade-off (Average across traits)
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Average across traits for each (method, alpha, tau)
    df_avg = df_all.groupby(["method", "alpha", "tau"]).agg({
        "score": "mean",
        "ppl": "mean"
    }).reset_index()
    
    for method in sorted(df_avg["method"].unique()):
        subset_m = df_avg[df_avg["method"] == method]
        color = COLOR_MAP.get(method, None)
        for alpha in sorted(subset_m["alpha"].unique()):
            subset = subset_m[subset_m["alpha"] == alpha].sort_values("tau")
            marker = MARKERS.get(alpha, "x")
            label = f"{method.upper()} (Alpha={alpha})"
            plt.plot(subset["score"], subset["ppl"], marker=marker, label=label, color=color, alpha=0.8)
            for i, row in subset.iterrows():
                plt.annotate(f"{row['tau']}", (row["score"], row["ppl"]), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.xlabel("Personality Score (1-5)")
    plt.ylabel("Perplexity (Lower is better)")
    plt.title("Stress Test: Consistency vs Coherence Trade-off (Average)")
    plt.legend()
    
    out_path = out_dir / "stress_test_comparison_all.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved comparison plot to {out_path}")
    
    # Save a summary CSV for reading the numbers
    summary_csv = out_dir / "stress_test_summary.csv"
    df_avg.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")

    # 2. Per-trait plots
    for trait in sorted(df_all["trait"].unique()):
        plt.figure(figsize=(10, 7))
        subset_trait = df_all[df_all["trait"] == trait]
        for method in sorted(subset_trait["method"].unique()):
            subset_m = subset_trait[subset_trait["method"] == method]
            color = COLOR_MAP.get(method, None)
            for alpha in sorted(subset_m["alpha"].unique()):
                subset = subset_m[subset_m["alpha"] == alpha].sort_values("tau")
                marker = MARKERS.get(alpha, "x")
                label = f"{method.upper()} (Alpha={alpha})"
                plt.errorbar(subset["score"], subset["ppl"], xerr=subset["score_sem"], yerr=subset["ppl_sem"], 
                             marker=marker, label=label, color=color, alpha=0.7)
                for i, row in subset.iterrows():
                    plt.annotate(f"{row['tau']:.1f}", (row["score"], row["ppl"]), 
                                 textcoords="offset points", xytext=(0,5), fontsize=8)

        plt.xlabel("Personality Score (1-5)")
        plt.ylabel("Perplexity (Lower is better)")
        plt.title(f"Stress Test: {trait.capitalize()}")
        plt.legend()
        plt.savefig(out_dir / f"stress_test_{trait}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
