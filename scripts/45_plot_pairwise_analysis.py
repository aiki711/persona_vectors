#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 45_plot_pairwise_analysis.py
#
# Visualize Pairwise Comparison results as heatmaps.
# Score +3 (Strong increase) to -3 (Strong decrease).
#

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.5, 1, 2, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40]

def highlight_safe_cells(ax, p_ppl, threshold=25.0):
    """Draw a black thin border around cells where PPL <= threshold."""
    rows, cols = p_ppl.shape
    for r in range(rows):
        for c in range(cols):
            val = p_ppl.iloc[r, c]
            if val <= threshold:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", lw=2))

def load_pairwise_summary(input_dir: Path, trait: str) -> pd.DataFrame:
    """Load all pairwise CSVs for a trait and return a mean summary."""
    trait_dir = input_dir / trait
    if not trait_dir.exists():
        return pd.DataFrame()
        
    records = []
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"layer_{layer}_Val{val}_pairwise.csv"
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
                
            records.append({
                "layer": layer,
                "val": val,
                "const_shift": df["const_shift"].mean(),
                "adapt_shift": df["adapt_shift"].mean(),
                "const_ppl": df["const_ppl"].mean(),
                "adapt_ppl": df["adapt_ppl"].mean()
            })
    
    return pd.DataFrame(records)

def plot_pairwise_heatmaps(df: pd.DataFrame, trait: str, out_dir: Path):
    """Plot heatmaps for constant and adaptive shifts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    p_const = df.pivot(index="val", columns="layer", values="const_shift")
    p_adapt = df.pivot(index="val", columns="layer", values="adapt_shift")
    p_c_ppl = df.pivot(index="val", columns="layer", values="const_ppl")
    p_a_ppl = df.pivot(index="val", columns="layer", values="adapt_ppl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    cmap = "RdBu_r"
    vmin, vmax = -3.0, 3.0
    
    sns.heatmap(p_const, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, center=0, ax=ax1)
    highlight_safe_cells(ax1, p_c_ppl, threshold=25.0)
    ax1.set_title(f"Constant - Relative Shift [{trait.capitalize()}] (Border: PPL<=25)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Val")
    
    sns.heatmap(p_adapt, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, center=0, ax=ax2)
    highlight_safe_cells(ax2, p_a_ppl, threshold=25.0)
    ax2.set_title(f"Adaptive - Relative Shift [{trait.capitalize()}] (Border: PPL<=25)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Val")
    
    plt.suptitle(f"Pairwise Comparison Analysis: {trait.capitalize()}", fontsize=14)
    plt.tight_layout()
    
    out_path = out_dir / f"heatmap_{trait}_pairwise.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved pairwise heatmap: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="exp_steering_layer_analysis/pairwise_results")
    parser.add_argument("--out_dir", default="exp_steering_layer_analysis/figures_pairwise")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    all_traits_data = []
    
    for trait in TRAITS:
        print(f"Processing Pairwise for {trait}...")
        df = load_pairwise_summary(in_dir, trait)
        if not df.empty:
            plot_pairwise_heatmaps(df, trait, out_dir / trait)
            df["trait"] = trait
            all_traits_data.append(df)
            
    if all_traits_data:
        print("Generating Global Summary...")
        full = pd.concat(all_traits_data)
        avg = full.groupby(["layer", "val"]).mean(numeric_only=True).reset_index()
        
        p_const = avg.pivot(index="val", columns="layer", values="const_shift")
        p_adapt = avg.pivot(index="val", columns="layer", values="adapt_shift")
        p_c_ppl = avg.pivot(index="val", columns="layer", values="const_ppl")
        p_a_ppl = avg.pivot(index="val", columns="layer", values="adapt_ppl")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(p_const, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-3, vmax=3, center=0, ax=ax1)
        highlight_safe_cells(ax1, p_c_ppl, threshold=25.0)
        ax1.set_title("Constant - Avg Relative Shift (All Traits) (Border: PPL<=25)")
        
        sns.heatmap(p_adapt, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-3, vmax=3, center=0, ax=ax2)
        highlight_safe_cells(ax2, p_a_ppl, threshold=25.0)
        ax2.set_title("Adaptive - Avg Relative Shift (All Traits) (Border: PPL<=25)")
        
        plt.tight_layout()
        plt.savefig(out_dir / "summary_all_traits_pairwise.png", dpi=200)
        plt.close()
        print(f"  Saved Global Pairwise summary: {out_dir / 'summary_all_traits_pairwise.png'}")

if __name__ == "__main__":
    main()
