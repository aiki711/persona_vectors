#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 77_plot_fusion_midpoint_comparison.py
#
# 中点正規化モード（--norm_mode midpoint）での実験結果を集計し、
#   - Midpoint DLS (Fixed)
#   - Midpoint Fusion Sigmoid
#   - Midpoint Fusion Soft-Plateau
# の3手法を比較するヒートマップを作成するスクリプト。
#
# Outputs:
#   - exp_steering_dyn_ic_fusion_midpoint/figures/{trait}/heatmap_{trait}_fusion_midpoint.png
#   - exp_steering_dyn_ic_fusion_midpoint/figures/summary_fusion_midpoint_all_traits.png
#

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS   = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0]
MODES  = ["fixed", "sigmoid", "soft_plateau"]

METHOD_NAMES = {
    "fixed": "DLS_Midpoint",
    "sigmoid": "Fusion_Sigmoid",
    "soft_plateau": "Fusion_Plateau"
}

def load_summary(results_dir: Path, axis: str, mode: str) -> pd.DataFrame:
    records = []
    trait_dir = results_dir / axis
    for val in VALS:
        # Check both float representation and raw string in filename
        csv_path = trait_dir / f"scores_fusion_{mode}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_fusion_{mode}_Val{val}.csv"
            if not csv_path.exists():
                continue
        try:
            df = pd.read_csv(csv_path)
            records.append({
                "val": val,
                "score": df["dyn_score"].mean(),
                "ppl":   df["dyn_ppl"].mean(),
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
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2, clip_on=False)
                ax.add_patch(rect)

def plot_axis(axis: str, results_dir: Path, out_dir: Path):
    print(f"[{axis}] Processing midpoint comparison heatmap...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data for each mode
    dfs = {}
    for m in MODES:
        dfs[m] = load_summary(results_dir, axis, m)

    # Build comparison tables
    score_data = {}
    ppl_data = {}

    for m in MODES:
        name = METHOD_NAMES[m]
        df = dfs[m]
        if not df.empty:
            idx_df = df.set_index("val")
            score_data[name] = idx_df["score"]
            ppl_data[name] = idx_df["ppl"]

    if not score_data:
        print(f"  [WARNING] No data found for {axis}")
        return

    p_score = pd.DataFrame(score_data)
    p_ppl   = pd.DataFrame(ppl_data)

    # Reindex to ensure sorted index of sweep values
    p_score = p_score.reindex(VALS)
    p_ppl   = p_ppl.reindex(VALS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Score Heatmap
    sns.heatmap(p_score, annot=True, fmt=".2f", cmap="YlGn",
                vmin=1.0, vmax=5.0, linewidths=0.4, linecolor="gray",
                ax=axes[0], annot_kws={"size": 9})
    axes[0].set_title(f"Score Comparison [{axis.capitalize()}]\n(Target score: ~4.0-5.0)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Evaluation Method")
    axes[0].set_ylabel("Val (Relative Steering Strength)")

    # PPL Heatmap
    sns.heatmap(p_ppl, annot=True, fmt=".1f", cmap="RdYlGn_r",
                vmin=1.0, vmax=50.0, linewidths=0.4, linecolor="gray",
                ax=axes[1], annot_kws={"size": 9})
    axes[1].set_title(f"PPL Comparison [{axis.capitalize()}]\n(Border: PPL <= 25.0 is Safe)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Evaluation Method")
    axes[1].set_ylabel("Val (Relative Steering Strength)")

    highlight_safe_cells(axes[0], p_ppl, threshold=25.0)
    highlight_safe_cells(axes[1], p_ppl, threshold=25.0)

    plt.suptitle(f"Midpoint-Normalized DLS vs IC Fusion Comparison: {axis.capitalize()}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = out_dir / f"heatmap_{axis}_fusion_midpoint.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap to: {out_path}")


def make_summary_heatmap(results_dir: Path, out_dir: Path):
    print("\n[Summary] Generating average summary heatmap across all traits...")
    out_dir.mkdir(parents=True, exist_ok=True)

    trait_scores = {METHOD_NAMES[m]: [] for m in MODES}
    trait_ppls   = {METHOD_NAMES[m]: [] for m in MODES}

    for axis in TRAITS:
        for m in MODES:
            name = METHOD_NAMES[m]
            df = load_summary(results_dir, axis, m)
            if not df.empty:
                idx_df = df.set_index("val").reindex(VALS)
                trait_scores[name].append(idx_df["score"])
                trait_ppls[name].append(idx_df["ppl"])

    p_score_list = []
    p_ppl_list = []

    for name in METHOD_NAMES.values():
        if trait_scores[name]:
            p_score_list.append(pd.concat(trait_scores[name], axis=1).mean(axis=1).rename(name))
        if trait_ppls[name]:
            p_ppl_list.append(pd.concat(trait_ppls[name], axis=1).mean(axis=1).rename(name))

    if not p_score_list:
        print("  [WARNING] No summary data found.")
        return

    p_score = pd.concat(p_score_list, axis=1)
    p_ppl   = pd.concat(p_ppl_list, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Score Heatmap
    sns.heatmap(p_score, annot=True, fmt=".2f", cmap="YlGn",
                vmin=1.0, vmax=5.0, linewidths=0.4, linecolor="gray",
                ax=axes[0], annot_kws={"size": 9})
    axes[0].set_title("Score Summary (All Traits Avg)\n(Target score: ~4.0-5.0)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Evaluation Method")
    axes[0].set_ylabel("Val (Relative Steering Strength)")

    # PPL Heatmap
    sns.heatmap(p_ppl, annot=True, fmt=".1f", cmap="RdYlGn_r",
                vmin=1.0, vmax=50.0, linewidths=0.4, linecolor="gray",
                ax=axes[1], annot_kws={"size": 9})
    axes[1].set_title("PPL Summary (All Traits Avg)\n(Border: PPL <= 25.0 is Safe)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Evaluation Method")
    axes[1].set_ylabel("Val (Relative Steering Strength)")

    highlight_safe_cells(axes[0], p_ppl, threshold=25.0)
    highlight_safe_cells(axes[1], p_ppl, threshold=25.0)

    plt.suptitle("Midpoint-Normalized DLS vs IC Fusion Summary — All Traits Average",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    summary_path = out_dir / "summary_fusion_midpoint_all_traits.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary comparison to: {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="exp_steering_dyn_ic_fusion_midpoint/results")
    ap.add_argument("--out_dir",     default="exp_steering_dyn_ic_fusion_midpoint/figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)

    for trait in TRAITS:
        plot_axis(trait, results_dir, out_dir / trait)

    make_summary_heatmap(results_dir, out_dir)
    print("\nMidpoint normalized plotting complete.")


if __name__ == "__main__":
    main()
