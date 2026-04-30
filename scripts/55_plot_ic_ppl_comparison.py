#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]

def highlight_safe_cells(ax, p_ppl, threshold=24.0):
    """Draw a black thin border around cells where PPL <= threshold."""
    rows, cols = p_ppl.shape
    for r in range(rows):
        for c in range(cols):
            val = p_ppl.iloc[r, c]
            if val <= threshold:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", lw=2))

def load_ppl_delta(input_dir: Path, trait: str) -> pd.DataFrame:
    """Load results and calculate PPL difference per condition."""
    trait_dir = input_dir / trait
    if not trait_dir.exists():
        return pd.DataFrame()
        
    records = []
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"ic_adapt_layer{layer}_Tau{val}_S1.5_pairwise.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"ic_adapt_layer{layer}_Tau{float(val)}_S1.5_pairwise.csv"
                if not csv_path.exists():
                    continue
            
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                continue

            if df.empty:
                continue
                
            c_ppl = df["const_ppl"].mean()
            ic_ppl = df["ic_adapt_ppl"].mean()
            
            # Improvement Rate (%)
            # const_ppl is usually > 0.
            # positive rate means IC is better (lower PPL) than Constant.
            rate = 100.0 * (c_ppl - ic_ppl) / c_ppl if c_ppl > 0 else 0
            
            records.append({
                "layer": layer,
                "val": val,
                "ppl_rate": rate,
                "const_ppl": c_ppl,
                "ic_adapt_ppl": ic_ppl
            })
    
    return pd.DataFrame(records)

def plot_ppl_comparison(df: pd.DataFrame, trait: str, out_dir: Path):
    """Plot PPL comparison heatmap (IC-Adaptive vs Constant)."""
    trait_out_dir = out_dir / trait
    trait_out_dir.mkdir(parents=True, exist_ok=True)
    
    p_rate = df.pivot(index="val", columns="layer", values="ppl_rate")
    p_ic_ppl = df.pivot(index="val", columns="layer", values="ic_adapt_ppl")
    
    plt.figure(figsize=(8, 6))
    
    # Red for positive (IC is better than Constant), Blue for negative
    sns.heatmap(p_rate, annot=True, fmt=".1f", cmap="RdBu_r", vmin=-80, vmax=80, center=0, 
                linewidths=0.5, linecolor="gray", cbar_kws={'label': 'PPL Improvement Rate vs Constant (%)'})
    
    highlight_safe_cells(plt.gca(), p_ic_ppl, threshold=24.0)
    
    plt.title(f"PPL Improvement: IC-Adaptive vs Constant [{trait.capitalize()}]\n(Red = IC Better, Border: IC PPL <= 24)", fontsize=12, fontweight="bold")
    plt.xlabel("Layer")
    plt.ylabel("Val (Tau)")
    
    out_path = trait_out_dir / f"heatmap_{trait}_ppl_comparison_vs_const.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved PPL comparison heatmap: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="exp_steering_ic_adaptive/pairwise_vs_const_results")
    parser.add_argument("--out_dir", default="exp_steering_ic_adaptive/figures")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    all_data = []
    
    for trait in TRAITS:
        print(f"Processing PPL delta for {trait}...")
        df = load_ppl_delta(in_dir, trait)
        if not df.empty:
            plot_ppl_comparison(df, trait, out_dir)
            df["trait"] = trait
            all_data.append(df)
            
    if all_data:
        print("Generating Global Summary...")
        full = pd.concat(all_data)
        avg = full.groupby(["layer", "val"]).mean(numeric_only=True).reset_index()
        
        p_rate = avg.pivot(index="val", columns="layer", values="ppl_rate")
        p_ic_ppl = avg.pivot(index="val", columns="layer", values="ic_adapt_ppl")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_rate, annot=True, fmt=".1f", cmap="RdBu_r", vmin=-80, vmax=80, center=0,
                    linewidths=0.5, linecolor="gray", cbar_kws={'label': 'Avg PPL Improvement Rate (%)'})
        highlight_safe_cells(plt.gca(), p_ic_ppl, threshold=24.0)
        plt.title("Avg PPL Improvement: IC-Adaptive vs Constant (All Traits)\n(Border: IC PPL <= 24)", fontsize=12, fontweight="bold")
        plt.xlabel("Layer")
        plt.ylabel("Val (Tau)")
        
        plt.tight_layout()
        plt.savefig(out_dir / "summary_all_traits_ppl_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Global PPL comparison summary: {out_dir / 'summary_all_traits_ppl_comparison.png'}")

if __name__ == "__main__":
    main()
