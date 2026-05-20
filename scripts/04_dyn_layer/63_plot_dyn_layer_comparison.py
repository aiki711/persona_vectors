#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 63_plot_dyn_layer_comparison.py
#
# Creates a Pareto tradeoff plot comparing:
# 1. Base Layer Sweeps (Constant & Adaptive)
# 2. Bhandari et al. DLS (logit_diff)
# 3. Proposed DLS (anti_alignment)
# 4. Constrained Bhandari et al. DLS
# 5. Constrained Proposed DLS

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def load_base_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            # Check float first, fallback to original
            csv_path = trait_dir / f"scores_layer_{layer}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv"
                if not csv_path.exists():
                    continue
            df = pd.read_csv(csv_path)
            records.append({
                "layer": layer, "val": val,
                "base_score": df["base_score"].mean(), "base_ppl": df["base_ppl"].mean(),
                "const_score": df["const_score"].mean(), "const_ppl": df["const_ppl"].mean(),
                "adapt_score": df["adapt_score"].mean(), "adapt_ppl": df["adapt_ppl"].mean(),
            })
    return pd.DataFrame(records)

def load_dyn_summary(input_dir: Path, axis: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = input_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            if not csv_path.exists():
                continue
        df = pd.read_csv(csv_path)
        records.append({
            "val": val,
            "score": df["dyn_score"].mean(),
            "ppl": df["dyn_ppl"].mean(),
        })
    return pd.DataFrame(records)

def make_tradeoff_plot(base_df: pd.DataFrame, logit_df: pd.DataFrame, anti_df: pd.DataFrame, 
                       logit_cns_df: pd.DataFrame, anti_cns_df: pd.DataFrame,
                       logit_zsc_df: pd.DataFrame, anti_zsc_df: pd.DataFrame,
                       logit_czs_df: pd.DataFrame, anti_czs_df: pd.DataFrame,
                       axis: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if base_df.empty:
        print(f"[{axis}] No base data to plot.")
        return

    layer_vals = sorted(base_df["layer"].unique())
    cmap = plt.get_cmap("tab20", len(layer_vals))
    color_map = {l: cmap(i) for i, l in enumerate(layer_vals)}

    for mode, score_col, ppl_col, ax in [
        ("Constant Background", "const_score", "const_ppl", axes[0]),
        ("Adaptive Background", "adapt_score", "adapt_ppl", axes[1]),
    ]:
        # Plot Base Background Lines
        for layer in layer_vals:
            sub = base_df[base_df["layer"] == layer].sort_values("val")
            ax.plot(sub[score_col], sub[ppl_col], "-", color=color_map[layer], alpha=0.3, linewidth=1.0)
            ax.scatter(sub[score_col], sub[ppl_col], color=color_map[layer], s=15, alpha=0.3)
        
        # Baseline point
        b_score, b_ppl = base_df["base_score"].mean(), base_df["base_ppl"].mean()
        ax.scatter([b_score], [b_ppl], color="black", marker="*", s=250, zorder=5, label="Original Base")

        # Plot Bhandari DLS (Logit Diff)
        if not logit_df.empty:
            logit_df = logit_df.sort_values("val")
            ax.plot(logit_df["score"], logit_df["ppl"], "-o", color="blue", linewidth=2.5, markersize=8, zorder=6, label="Bhandari (Unconstrained)")
            for _, row in logit_df.iterrows():
                ax.annotate(f"{row['val']:g}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="blue", weight='bold')

        # Plot Proposed DLS (Anti Alignment)
        if not anti_df.empty:
            anti_df = anti_df.sort_values("val")
            ax.plot(anti_df["score"], anti_df["ppl"], "-s", color="red", linewidth=2.5, markersize=8, zorder=7, label="Proposed (Unconstrained)")
            for _, row in anti_df.iterrows():
                ax.annotate(f"{row['val']:g}", (row["score"], row["ppl"]), textcoords="offset points", xytext=(4, -12), fontsize=8, color="red", weight='bold')

        # Plot Constrained Bhandari DLS
        if not logit_cns_df.empty:
            logit_cns_df = logit_cns_df.sort_values("val")
            ax.plot(logit_cns_df["score"], logit_cns_df["ppl"], "--o", color="dodgerblue", linewidth=3.0, markersize=9, zorder=8, label="Bhandari (Constrained)")
            
        # Plot Constrained Proposed DLS
        if not anti_cns_df.empty:
            anti_cns_df = anti_cns_df.sort_values("val")
            ax.plot(anti_cns_df["score"], anti_cns_df["ppl"], "--s", color="salmon", linewidth=3.0, markersize=9, zorder=9, label="Proposed (Constrained)")

        # Plot Z-score Normalized Bhandari DLS
        if not logit_zsc_df.empty:
            logit_zsc_df = logit_zsc_df.sort_values("val")
            ax.plot(logit_zsc_df["score"], logit_zsc_df["ppl"], "-.^", color="darkblue", linewidth=3.0, markersize=9, zorder=10, label="Bhandari (Z-score)")
            
        # Plot Z-score Normalized Proposed DLS
        if not anti_zsc_df.empty:
            anti_zsc_df = anti_zsc_df.sort_values("val")
            ax.plot(anti_zsc_df["score"], anti_zsc_df["ppl"], "-.v", color="darkred", linewidth=3.0, markersize=9, zorder=11, label="Proposed (Z-score)")

        # Plot Constrained Z-score Normalized Bhandari DLS
        if not logit_czs_df.empty:
            logit_czs_df = logit_czs_df.sort_values("val")
            ax.plot(logit_czs_df["score"], logit_czs_df["ppl"], ":d", color="teal", linewidth=3.0, markersize=9, zorder=12, label="Bhandari (Constrained Z-score)")

        # Plot Constrained Z-score Normalized Proposed DLS
        if not anti_czs_df.empty:
            anti_czs_df = anti_czs_df.sort_values("val")
            ax.plot(anti_czs_df["score"], anti_czs_df["ppl"], ":D", color="purple", linewidth=3.0, markersize=9, zorder=13, label="Proposed (Constrained Z-score)")

        ax.set_title(f"Pareto Front Comparison — {axis.capitalize()}\n(Background: {mode})", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Personality Score ({axis.capitalize()})", fontsize=11)
        ax.set_ylabel("Perplexity (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved plot: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="exp_steering_layer_analysis/results")
    ap.add_argument("--dyn_dir", default="exp_steering_dyn_layer/results")
    ap.add_argument("--cns_dir", default="exp_steering_dyn_layer_constrained/results")
    ap.add_argument("--zsc_dir", default="exp_steering_dyn_layer_zscore/results")
    ap.add_argument("--czs_dir", default="exp_steering_dyn_layer_CnsZsc/results")
    ap.add_argument("--out_dir", default="exp_steering_dyn_layer_CnsZsc/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for axis in TRAITS:
        print(f"\nProcessing {axis}...")
        base_df = load_base_summary(Path(args.base_dir), axis)
        logit_df = load_dyn_summary(Path(args.dyn_dir), axis, "logit_diff")
        anti_df = load_dyn_summary(Path(args.dyn_dir), axis, "anti_alignment")
        logit_cns_df = load_dyn_summary(Path(args.cns_dir), axis, "logit_diff")
        anti_cns_df = load_dyn_summary(Path(args.cns_dir), axis, "anti_alignment")
        logit_zsc_df = load_dyn_summary(Path(args.zsc_dir), axis, "logit_diff")
        anti_zsc_df = load_dyn_summary(Path(args.zsc_dir), axis, "anti_alignment")
        logit_czs_df = load_dyn_summary(Path(args.czs_dir), axis, "logit_diff")
        anti_czs_df = load_dyn_summary(Path(args.czs_dir), axis, "anti_alignment")
        
        out_path = out_dir / f"tradeoff_comparison_{axis}.png"
        make_tradeoff_plot(base_df, logit_df, anti_df, logit_cns_df, anti_cns_df, logit_zsc_df, anti_zsc_df, logit_czs_df, anti_czs_df, axis, out_path)

if __name__ == "__main__":
    main()
