#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 39_plot_full_layer_comparison.py
#
# Compare Constant vs Adaptive steering across a granular sweep [0.1, ..., 0.9]
# Plots Score (X) vs Perplexity (Y).

import argparse
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="exp_adaptive_steering/results/full_layer_granular")
    ap.add_argument("--out_dir", default="exp_adaptive_steering/figures/full_layer_granular")
    ap.add_argument("--axis", default="extraversion")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    # Check if we should use subdirectories or flat
    trait_dir = input_dir / args.axis
    if trait_dir.is_dir():
        search_path = trait_dir / f"scores_{args.axis}_Val*.csv"
    else:
        search_path = input_dir / f"scores_{args.axis}_Val*.csv"

    files = glob.glob(str(search_path))
    if not files:
        print(f"No files found at {search_path}")
        return

    data = []
    for f in files:
        # Extract VAL from scores_{axis}_Val{VAL}.csv
        match = re.search(r"Val([\d\.]+)\.csv", Path(f).name)
        if not match: continue
        val = float(match.group(1))
        
        try:
            df = pd.read_csv(f)
            # Base (stays same for all files, so we take mean)
            data.append({
                "val": val,
                "mode": "Constant",
                "score": df["const_score"].mean(),
                "ppl": df["const_ppl"].mean(),
                "score_sem": df["const_score"].sem(),
                "ppl_sem": df["const_ppl"].sem()
            })
            data.append({
                "val": val,
                "mode": "Adaptive",
                "score": df["adapt_score"].mean(),
                "ppl": df["adapt_ppl"].mean(),
                "score_sem": df["adapt_score"].sem(),
                "ppl_sem": df["adapt_ppl"].sem()
            })
            # Add base once
            if val == 0.1: # Just use one file for base
                data.append({
                    "val": 0.0,
                    "mode": "Baseline",
                    "score": df["base_score"].mean(),
                    "ppl": df["base_ppl"].mean(),
                    "score_sem": df["base_score"].sem(),
                    "ppl_sem": df["base_ppl"].sem()
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame(data)
    # Filter to val <= 0.15 as requested by user
    df_plot = df_plot[df_plot["val"] <= 0.15].sort_values(["mode", "val"])

    # --- Plotting ---
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    colors = {"Constant": "tab:red", "Adaptive": "tab:blue", "Baseline": "black"}
    markers = {"Constant": "x", "Adaptive": "s", "Baseline": "o"}
    
    # 1. Base Point
    base = df_plot[df_plot["mode"] == "Baseline"]
    plt.scatter(base["score"], base["ppl"], color=colors["Baseline"], marker=markers["Baseline"], s=100, label="Baseline", zorder=5)
    
    # 2. Constant Curve
    const = df_plot[df_plot["mode"] == "Constant"].sort_values("val")
    plt.plot(const["score"], const["ppl"], color=colors["Constant"], marker=markers["Constant"], label="Constant Steering", alpha=0.8, linewidth=2)
    for _, row in const.iterrows():
        plt.annotate(f"{row['val']}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color=colors["Constant"])

    # 3. Adaptive Curve
    adapt = df_plot[df_plot["mode"] == "Adaptive"].sort_values("val")
    plt.plot(adapt["score"], adapt["ppl"], color=colors["Adaptive"], marker=markers["Adaptive"], label="Adaptive Steering", alpha=0.8, linewidth=2)
    for _, row in adapt.iterrows():
        plt.annotate(f"{row['val']}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color=colors["Adaptive"])

    # 4. Fill between for visual clarity (Adaptive vs Constant)
    # This is tricky because X coords differ. Let's skip or use a simple alpha.
    
    plt.xlabel(f"Personality Score ({args.axis.capitalize()})", fontsize=12)
    plt.ylabel("Perplexity (Log Scale recommended for high PPL)", fontsize=12)
    plt.yscale("log") # PPL can explode
    plt.title(f"Comparison: Constant vs Adaptive Steering (Full-Layer)\nAxis: {args.axis.capitalize()}", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save
    out_file = out_dir / f"full_layer_comparison_{args.axis}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved comparison plot to {out_file}")
    
    # Save summary stats
    summary_file = out_dir / f"summary_stats_{args.axis}.csv"
    df_plot.to_csv(summary_file, index=False)
    print(f"Saved summary stats to {summary_file}")

if __name__ == "__main__":
    main()
