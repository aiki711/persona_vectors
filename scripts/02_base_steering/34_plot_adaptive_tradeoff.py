#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 34_plot_adaptive_tradeoff.py
#
# Visualizes the tradeoff between personality score and perplexity
# for Baseline, Constant, and Adaptive steering.

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to scored CSV (e.g., scores_adaptive_extraversion_L15.csv)")
    parser.add_argument("--output", required=True, help="Path to output plot PNG")
    parser.add_argument("--axis", required=True, help="Personality axis being evaluated")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Calculate means and std err
    methods = [("Base", "base_score", "base_ppl"), 
               ("Constant", "const_score", "const_ppl"), 
               ("Adaptive", "adapt_score", "adapt_ppl")]

    plot_data = []

    for name, score_col, ppl_col in methods:
        avg_score = df[score_col].mean()
        avg_ppl = df[ppl_col].mean()
        
        # Calculate standard error of the mean for error bars
        sem_score = df[score_col].sem()
        sem_ppl = df[ppl_col].sem()
        
        plot_data.append({
            "Method": name,
            "Personality Score": avg_score,
            "Perplexity": avg_ppl,
            "Score_SEM": sem_score,
            "PPL_SEM": sem_ppl
        })

    plot_df = pd.DataFrame(plot_data)

    # Print summary
    print(f"--- Trade-off Summary ({args.axis}) ---")
    print(plot_df.to_string(index=False))

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with error bars
    colors = {"Base": "gray", "Constant": "red", "Adaptive": "blue"}
    markers = {"Base": "o", "Constant": "x", "Adaptive": "s"}

    for _, row in plot_df.iterrows():
        method = row["Method"]
        
        ax.errorbar(
            x=row["Personality Score"], 
            y=row["Perplexity"],
            xerr=row["Score_SEM"],
            yerr=row["PPL_SEM"],
            fmt=markers[method],
            color=colors[method],
            markersize=10,
            capsize=5,
            label=method
        )
        
        # Add labels near points
        ax.annotate(
            method, 
            (row["Personality Score"], row["Perplexity"]), 
            textcoords="offset points", 
            xytext=(10,-10), 
            ha='left'
        )

    # Improve labels and titles
    ax.set_xlabel(f"Personality Score ({args.axis.capitalize()})\n<-- Lower | Higher -->", fontsize=12)
    ax.set_ylabel("Perplexity (Lower is Better/More Fluent)", fontsize=12)
    ax.set_title(f"Trade-off: Personality vs Perplexity ({args.axis})", fontsize=14, pad=15)
    
    # Legend
    ax.legend(title="Steering Method", loc="upper left")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved trade-off plot to {out_path}")

if __name__ == "__main__":
    main()
