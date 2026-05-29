#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def load_combined_data(trait: str) -> pd.DataFrame:
    """Load Constant and IC-Adaptive Pairwise (vs Base) data."""
    const_dir = Path("exp_steering_layer_analysis/pairwise_results") / trait
    ic_dir = Path("exp_steering_ic_adaptive/pairwise_results") / trait
    
    records = []
    for layer in LAYERS:
        for val in VALS:
            # 1. Load Constant
            c_score = 0.0
            c_ppl = 1000.0 # Default high PPL
            
            c_csv_path = const_dir / f"layer_{layer}_Val{val}_pairwise.csv"
            # In layer_analysis, val might be formatted as integer if it's .0
            if not c_csv_path.exists():
                c_csv_path = const_dir / f"layer_{layer}_Val{int(val) if float(val).is_integer() else float(val)}_pairwise.csv"
            
            if c_csv_path.exists():
                try:
                    df_c = pd.read_csv(c_csv_path)
                    if not df_c.empty:
                        c_score = df_c["const_shift"].mean()
                        c_ppl = df_c["const_ppl"].mean()
                except:
                    pass
            
            # 2. Load IC-Adaptive (vs Base)
            ic_score = 0.0
            ic_ppl = 1000.0
            
            ic_csv_path = ic_dir / f"ic_adapt_layer{layer}_Tau{val}_S1.5_pairwise.csv"
            if not ic_csv_path.exists():
                ic_csv_path = ic_dir / f"ic_adapt_layer{layer}_Tau{float(val)}_S1.5_pairwise.csv"
                
            if ic_csv_path.exists():
                try:
                    df_ic = pd.read_csv(ic_csv_path)
                    if not df_ic.empty:
                        ic_score = df_ic["pairwise_score"].mean()
                        ic_ppl = df_ic["ic_adapt_ppl"].mean()
                except:
                    pass
                    
            records.append({
                "layer": layer,
                "val": val,
                "const_score": c_score,
                "const_ppl": c_ppl,
                "ic_score": ic_score,
                "ic_ppl": ic_ppl
            })
            
    return pd.DataFrame(records)

def plot_side_by_side(df: pd.DataFrame, trait: str, out_dir: Path):
    trait_out_dir = out_dir / trait
    trait_out_dir.mkdir(parents=True, exist_ok=True)
    
    p_c_score = df.pivot(index="val", columns="layer", values="const_score")
    p_ic_score = df.pivot(index="val", columns="layer", values="ic_score")
    p_c_ppl = df.pivot(index="val", columns="layer", values="const_ppl")
    p_ic_ppl = df.pivot(index="val", columns="layer", values="ic_ppl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Range for Delta score (Divergent)
    d_vmin, d_vmax = -3.0, 3.0
    cmap_score = "RdBu_r"
    cmap_ppl = "RdYlGn_r"
    
    configs = [
        (axes[0, 0], p_c_score, p_c_ppl, "Constant: Pairwise Score (vs Base)", cmap_score, d_vmin, d_vmax, ".2f"),
        (axes[0, 1], p_ic_score, p_ic_ppl, "IC-Adaptive: Pairwise Score (vs Base)", cmap_score, d_vmin, d_vmax, ".2f"),
        (axes[1, 0], p_c_ppl, p_c_ppl, "Constant: PPL", cmap_ppl, 1, 100, ".1f"),
        (axes[1, 1], p_ic_ppl, p_ic_ppl, "IC-Adaptive: PPL", cmap_ppl, 1, 100, ".1f")
    ]
    
    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax, center=0 if "Score" in title else None,
                    linewidths=0.4, linecolor="gray", ax=ax_obj)
        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=24.0)
        ax_obj.set_title(f"{title}\n(Border: PPL<=24)", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Val")
        
    plt.suptitle(f"Side-by-Side Comparison: Constant vs IC-Adaptive [{trait.capitalize()}]", fontsize=16)
    plt.tight_layout()
    
    out_path = trait_out_dir / f"heatmap_{trait}_side_by_side.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved side-by-side plot: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="exp_steering_ic_adaptive/figures")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    
    all_data = []
    
    for trait in TRAITS:
        print(f"Processing side-by-side plot for {trait}...")
        df = load_combined_data(trait)
        if not df.empty:
            plot_side_by_side(df, trait, out_dir)
            df["trait"] = trait
            all_data.append(df)
            
    if all_data:
        print("Generating Global Side-by-Side Summary...")
        full = pd.concat(all_data)
        avg = full.groupby(["layer", "val"]).mean(numeric_only=True).reset_index()
        plot_side_by_side(avg, "all_traits_average", out_dir)
        
if __name__ == "__main__":
    main()
