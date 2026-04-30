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

def load_tradeoff_data(input_dir: Path, trait: str) -> pd.DataFrame:
    """Load Pairwise CSVs and extract Pairwise Score and PPLs."""
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
                
            p_score = df["pairwise_score"].mean()
            c_ppl = df["const_ppl"].mean()
            ic_ppl = df["ic_adapt_ppl"].mean()
            
            records.append({
                "layer": layer,
                "val": val,
                "pairwise_score": p_score,
                "const_ppl": c_ppl,
                "ic_adapt_ppl": ic_ppl,
                "ppl_diff": c_ppl - ic_ppl
            })
    
    return pd.DataFrame(records)

def plot_tradeoff_scatter(df: pd.DataFrame, trait: str, out_dir: Path):
    """Plot Scatter of Pairwise Score vs PPL Difference."""
    trait_out_dir = out_dir / trait
    trait_out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot, color by layer
    scatter = plt.scatter(df["pairwise_score"], df["ppl_diff"], 
                          c=df["layer"], cmap="viridis", alpha=0.7, s=50, edgecolors="k")
    
    plt.axhline(0, color="red", linestyle="--", alpha=0.5, label="No PPL Change")
    plt.axvline(0, color="blue", linestyle="--", alpha=0.5, label="No Score Change")
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Layer")
    
    plt.title(f"Trade-off: Pairwise Score vs PPL Improvement [{trait.capitalize()}]\n(IC-Adaptive vs Constant)", fontsize=12)
    plt.xlabel("Pairwise Score (Positive = IC is better)")
    plt.ylabel("PPL Improvement (Constant PPL - IC PPL)")
    plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    out_path = trait_out_dir / f"scatter_{trait}_tradeoff_vs_const.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved trade-off scatter: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="exp_steering_ic_adaptive/pairwise_vs_const_results")
    parser.add_argument("--out_dir", default="exp_steering_ic_adaptive/figures")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    all_data = []
    
    for trait in TRAITS:
        print(f"Processing trade-off for {trait}...")
        df = load_tradeoff_data(in_dir, trait)
        if not df.empty:
            plot_tradeoff_scatter(df, trait, out_dir)
            df["trait"] = trait
            all_data.append(df)
            
    if all_data:
        print("Generating Global Trade-off Summary...")
        full = pd.concat(all_data)
        
        plt.figure(figsize=(10, 8))
        # Color by trait
        palette = sns.color_palette("Set1", len(TRAITS))
        sns.scatterplot(data=full, x="pairwise_score", y="ppl_diff", hue="trait", alpha=0.7, s=50, edgecolor="k")
        
        plt.axhline(0, color="red", linestyle="--", alpha=0.5)
        plt.axvline(0, color="blue", linestyle="--", alpha=0.5)
        
        plt.title("Global Trade-off: Pairwise Score vs PPL Improvement\n(IC-Adaptive vs Constant)", fontsize=14, fontweight="bold")
        plt.xlabel("Pairwise Score (Positive = IC is more expressive)")
        plt.ylabel("PPL Improvement (Constant PPL - IC PPL)")
        plt.legend(title="Trait")
        plt.grid(True, alpha=0.3)
        
        out_path = out_dir / "summary_all_traits_tradeoff_vs_const.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved Global Trade-off summary: {out_path}")

if __name__ == "__main__":
    main()
