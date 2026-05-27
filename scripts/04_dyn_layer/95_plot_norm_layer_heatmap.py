#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 95_plot_norm_layer_heatmap.py
#
# Generates heatmaps (Score and PPL) for the norm-scaled single-layer sweep.
# Reads from exp_steering_layer_norm/results/{axis}/scores_layer_{L}_Val{alpha}.csv
#
# Outputs:
#   exp_steering_layer_norm/figures/{axis}/heatmap_{axis}.png   (per-trait)
#   exp_steering_layer_norm/figures/summary_norm_layer.png       (all traits)
#

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = list(range(32))
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]


def load_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{layer}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv"
                if not csv_path.exists():
                    continue
            try:
                df = pd.read_csv(csv_path)
                records.append({
                    "layer": layer,
                    "val":   val,
                    "const_score": df["const_score"].mean(),
                    "const_ppl":   df["const_ppl"].mean(),
                })
            except Exception:
                pass
    return pd.DataFrame(records)


def highlight_safe_cells(ax, p_ppl, threshold=25.0):
    if p_ppl is None or p_ppl.empty:
        return
    for i in range(len(p_ppl.index)):
        for j in range(len(p_ppl.columns)):
            val = p_ppl.iloc[i, j]
            if not np.isnan(val) and val <= threshold:
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2.5, clip_on=False)
                ax.add_patch(rect)


def make_empty_pivot(vals, layers):
    return pd.DataFrame(np.nan, index=layers, columns=vals)


def plot_trait_heatmap(axis: str, input_dir: Path, out_dir: Path):
    """Plot Score & PPL heatmaps (layers × alpha) for a single trait."""
    df = load_summary(input_dir, axis)
    fig_dir = out_dir / "figures" / axis
    fig_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"  [SKIP] No data for {axis}")
        return

    # Build pivots: rows = layers (0-31), cols = alpha values
    layers_present = sorted(df["layer"].unique())
    vals_present   = sorted(df["val"].unique())

    p_score = make_empty_pivot(vals_present, layers_present)
    p_ppl   = make_empty_pivot(vals_present, layers_present)

    for _, row in df.iterrows():
        if row["layer"] in p_score.index and row["val"] in p_score.columns:
            p_score.at[row["layer"], row["val"]] = row["const_score"]
            p_ppl.at[row["layer"], row["val"]]   = row["const_ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle(f"Norm-Scaled Single-Layer Steering — {axis.capitalize()}", fontsize=16, fontweight="bold")

    # Score heatmap
    ax = axes[0]
    sns.heatmap(
        p_score, ax=ax, cmap="YlOrRd", vmin=0, vmax=5,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Personality Score (0-5)"},
    )
    ax.set_title("Personality Score", fontsize=13)
    ax.set_xlabel("Alpha (α)", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    highlight_safe_cells(ax, p_ppl, threshold=25.0)

    # PPL heatmap
    ax = axes[1]
    sns.heatmap(
        p_ppl, ax=ax, cmap="RdYlGn_r", vmin=10, vmax=60,
        annot=True, fmt=".1f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Perplexity (PPL)"},
    )
    ax.set_title("Perplexity (PPL)", fontsize=13)
    ax.set_xlabel("Alpha (α)", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    highlight_safe_cells(ax, p_ppl, threshold=25.0)

    plt.tight_layout()
    out_path = fig_dir / f"heatmap_{axis}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_summary_heatmap(input_dir: Path, out_dir: Path):
    """Plot a summary heatmap with the best safe score per layer for each trait."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_trait_data = {}
    for axis in TRAITS:
        df = load_summary(input_dir, axis)
        if df.empty:
            continue
        # Best safe score (PPL <= 25) per layer
        best_per_layer = {}
        for layer in LAYERS:
            ld = df[df["layer"] == layer]
            safe = ld[ld["const_ppl"] <= 25.0]
            if not safe.empty:
                best_per_layer[layer] = safe["const_score"].max()
            else:
                best_per_layer[layer] = np.nan
        all_trait_data[axis] = best_per_layer

    if not all_trait_data:
        print("[SKIP] No data for summary heatmap.")
        return

    summary_df = pd.DataFrame(all_trait_data).T  # rows=traits, cols=layers
    summary_df.columns = [f"L{c}" for c in summary_df.columns]

    fig, ax = plt.subplots(figsize=(28, 5))
    sns.heatmap(
        summary_df, ax=ax, cmap="YlOrRd", vmin=0, vmax=5,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Best Safe Personality Score"},
    )
    ax.set_title("Norm-Scaled Single-Layer Sweep — Best Safe Score per Layer & Trait", fontsize=14, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Trait", fontsize=12)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)

    plt.tight_layout()
    out_path = fig_dir / "summary_norm_layer.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="exp_steering_layer_norm/results",
                    help="Path to norm-scaled sweep results")
    ap.add_argument("--out_dir", default="exp_steering_layer_norm",
                    help="Output base directory for figures")
    ap.add_argument("--traits", nargs="*", default=TRAITS,
                    help="Traits to plot")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)

    print("=== Norm-Scaled Single-Layer Sweep Heatmaps ===")
    for axis in args.traits:
        print(f"\n[{axis}]")
        plot_trait_heatmap(axis, input_dir, out_dir)

    print("\n[Summary]")
    plot_summary_heatmap(input_dir, out_dir)
    print("\nDONE.")


if __name__ == "__main__":
    main()
