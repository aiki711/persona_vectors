#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 43_plot_delta_analysis.py
#
# Calculate Delta = (Steered Score) - (Baseline Score)
# and visualize as heatmaps.
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

def load_delta_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    """Load CSVs and calculate average delta scores per condition."""
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            
            # Calculate mean scores and PPLs
            base_s = df["base_score"].mean()
            const_s = df["const_score"].mean()
            adapt_s = df["adapt_score"].mean()
            
            base_p = df["base_ppl"].mean()
            const_p = df["const_ppl"].mean()
            adapt_p = df["adapt_ppl"].mean()

            records.append({
                "layer": layer,
                "val":   val,
                "base_score": base_s,
                "const_delta": const_s - base_s,
                "adapt_delta": adapt_s - base_s,
                "const_ppl": const_p,
                "adapt_ppl": adapt_p
            })
    return pd.DataFrame(records)

def highlight_safe_cells(ax, p_ppl, threshold=25.0):
    """Draw a black thin border around cells where PPL <= threshold."""
    rows, cols = p_ppl.shape
    for r in range(rows):
        for c in range(cols):
            val = p_ppl.iloc[r, c]
            if val <= threshold:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", lw=2))

def plot_axis_delta(df: pd.DataFrame, axis: str, out_dir: Path):
    """Generate 2x2 Delta heatmap for the given axis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pivot maps
    p_c_delta = df.pivot(index="val", columns="layer", values="const_delta")
    p_a_delta = df.pivot(index="val", columns="layer", values="adapt_delta")
    p_c_ppl   = df.pivot(index="val", columns="layer", values="const_ppl")
    p_a_ppl   = df.pivot(index="val", columns="layer", values="adapt_ppl")

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # Layout:
    # [0,0]: Const Delta  [0,1]: Adapt Delta
    # [1,0]: Const PPL    [1,1]: Adapt PPL
    
    # Range for Delta score (Divergent)
    d_vmin, d_vmax = -2.0, 2.0
    cmap_delta = "RdBu_r" # Red: Positive change, Blue: Negative change
    cmap_ppl   = "RdYlGn_r"

    configs = [
        (axes[0, 0], p_c_delta, p_c_ppl, "Constant — Score Delta", cmap_delta, d_vmin, d_vmax, ".2f"),
        (axes[0, 1], p_a_delta, p_a_ppl, "Adaptive — Score Delta", cmap_delta, d_vmin, d_vmax, ".2f"),
        (axes[1, 0], p_c_ppl,   p_c_ppl, "Constant — PPL",         cmap_ppl,   1, 100, ".1f"),
        (axes[1, 1], p_a_ppl,   p_a_ppl, "Adaptive — PPL",         cmap_ppl,   1, 100, ".1f"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax, center=0 if "Delta" in title else None,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 9})
        
        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        
        ax_obj.set_title(f"{title} [{axis.capitalize()}] (Border: PPL<=25)", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Val")

    plt.suptitle(f"Layer-Sweep Delta Analysis: {axis.capitalize()}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_path = out_dir / f"heatmap_{axis}_delta.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved Delta heatmap: {out_path}")

def make_summary_delta(input_dir: Path, out_dir: Path):
    """Aggregate all traits and plot average delta."""
    all_dfs = []
    for trait in TRAITS:
        df = load_delta_summary(input_dir, trait)
        if not df.empty:
            df["trait"] = trait
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data found for summary.")
        return
    
    full = pd.concat(all_dfs)
    avg = full.groupby(["layer", "val"]).mean(numeric_only=True).reset_index()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    configs = [
        ("const_delta", "Constant — Delta Score (All traits avg)", "RdBu_r", -2.0, 2.0, axes[0, 0]),
        ("adapt_delta", "Adaptive — Delta Score (All traits avg)",  "RdBu_r", -2.0, 2.0, axes[0, 1]),
        ("const_ppl",   "Constant — PPL (All traits avg)",          "RdYlGn_r", 1, 100, axes[1, 0]),
        ("adapt_ppl",   "Adaptive — PPL (All traits avg)",          "RdYlGn_r", 1, 100, axes[1, 1]),
    ]
    
    for col, title, cmap, vmin, vmax, ax_obj in configs:
        p = avg.pivot(index="val", columns="layer", values=col)
        fmt = ".2f" if "delta" in col else ".1f"
        
        # Reference PPL for border
        ppl_col = col.replace("delta", "ppl") if "delta" in col else col
        p_ppl = avg.pivot(index="val", columns="layer", values=ppl_col)

        sns.heatmap(p, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax, center=0 if "Delta" in title else None,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 9})
        
        highlight_safe_cells(ax_obj, p_ppl, threshold=25.0)
        
        ax_obj.set_title(title + " (Border: PPL<=25)", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Val")

    plt.suptitle("Summary: Layer-Sweep Delta Analysis (Global Average)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    summary_path = out_dir / "summary_all_traits_delta.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved Global Delta summary: {summary_path}")
    
    # Save CSV
    avg.to_csv(out_dir / "summary_delta_all_traits.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    fig_dir = out_dir / "figures"
    
    for trait in TRAITS:
        print(f"Processing Delta for [{trait}]...")
        df = load_delta_summary(in_dir, trait)
        if df.empty:
            print(f"  No data for {trait}")
            continue
        plot_axis_delta(df, trait, fig_dir / trait)

    print("Generating Global Summary...")
    make_summary_delta(in_dir, out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
