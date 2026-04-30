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

def load_pairwise_summary(input_dir: Path, trait: str) -> pd.DataFrame:
    """Load all pairwise CSVs for a trait and return a mean summary."""
    trait_dir = input_dir / trait
    if not trait_dir.exists():
        return pd.DataFrame()
        
    records = []
    for layer in LAYERS:
        for val in VALS:
            # Filename: ic_adapt_layerX_TauY_S1.5_pairwise.csv
            csv_path = trait_dir / f"ic_adapt_layer{layer}_Tau{val}_S1.5_pairwise.csv"
            if not csv_path.exists():
                # fallback for float formatting differences
                csv_path = trait_dir / f"ic_adapt_layer{layer}_Tau{float(val)}_S1.5_pairwise.csv"
                if not csv_path.exists():
                    continue
            
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                continue

            if df.empty:
                continue
                
            records.append({
                "layer": layer,
                "val": val,
                "pairwise_score": df["pairwise_score"].mean(),
                "ic_adapt_ppl": df["ic_adapt_ppl"].mean()
            })
    
    return pd.DataFrame(records)

def plot_pairwise_heatmap(df: pd.DataFrame, trait: str, out_dir: Path):
    """Plot heatmap for IC-Adaptive pairwise shifts vs Constant."""
    trait_out_dir = out_dir / trait
    trait_out_dir.mkdir(parents=True, exist_ok=True)
    
    p_score = df.pivot(index="val", columns="layer", values="pairwise_score")
    p_ppl = df.pivot(index="val", columns="layer", values="ic_adapt_ppl")
    
    plt.figure(figsize=(8, 6))
    
    cmap = "RdBu_r"
    vmin, vmax = -3.0, 3.0
    
    sns.heatmap(p_score, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, center=0)
    highlight_safe_cells(plt.gca(), p_ppl, threshold=24.0)
    
    plt.title(f"IC-Adaptive Relative Shift [{trait.capitalize()}]\n(Border: PPL<=24)", fontsize=12)
    plt.xlabel("Layer")
    plt.ylabel("Val (Tau)")
    
    plt.tight_layout()
    
    out_path = trait_out_dir / f"heatmap_{trait}_pairwise_vs_const.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved pairwise heatmap: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="exp_steering_ic_adaptive/pairwise_vs_const_results")
    parser.add_argument("--out_dir", default="exp_steering_ic_adaptive/figures")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    all_traits_data = []
    
    for trait in TRAITS:
        print(f"Processing Pairwise for {trait}...")
        df = load_pairwise_summary(in_dir, trait)
        if not df.empty:
            plot_pairwise_heatmap(df, trait, out_dir)
            df["trait"] = trait
            all_traits_data.append(df)
            
    if all_traits_data:
        print("Generating Global Summary...")
        full = pd.concat(all_traits_data)
        avg = full.groupby(["layer", "val"]).mean(numeric_only=True).reset_index()
        
        p_score = avg.pivot(index="val", columns="layer", values="pairwise_score")
        p_ppl = avg.pivot(index="val", columns="layer", values="ic_adapt_ppl")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_score, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-3, vmax=3, center=0)
        highlight_safe_cells(plt.gca(), p_ppl, threshold=24.0)
        plt.title("Avg Relative Shift (All Traits) - IC-Adaptive\n(Border: PPL<=24)", fontsize=12)
        plt.xlabel("Layer")
        plt.ylabel("Val (Tau)")
        
        plt.tight_layout()
        plt.savefig(out_dir / "summary_all_traits_pairwise.png", dpi=200)
        plt.close()
        print(f"  Saved Global Pairwise summary: {out_dir / 'summary_all_traits_pairwise.png'}")

if __name__ == "__main__":
    main()
