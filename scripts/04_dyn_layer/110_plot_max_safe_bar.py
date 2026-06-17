#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/110_plot_max_safe_bar.py
#
# Generates a premium grouped bar chart comparing the maximum safe steering scores
# (PPL ≤ 25.0) of Unsteered, Cos-Only, Rank-Only, and Logit-Diff across all traits and the average.
#

import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_LABELS = {
    "extraversion":      "Extraversion",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness":     "Agreeableness",
}
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def get_max_safe_score(results_dir: Path, trait: str, method: str) -> tuple[float, float, float]:
    """
    Finds the maximum safe steering score for a given trait and method.
    Returns (max_score, corresponding_alpha, corresponding_ppl) or (0.0, np.nan, np.nan).
    """
    best_score = 0.0
    best_alpha = np.nan
    best_ppl = np.nan
    
    trait_dir = results_dir / trait
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "dyn_score" in df.columns:
                    df["dyn_score"] = df["dyn_score"].replace(0, np.nan)
                mean_score = df["dyn_score"].mean()
                mean_ppl = df["dyn_ppl"].mean()
                
                # Check safety threshold
                if mean_ppl <= 25.0:
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = val
                        best_ppl = mean_ppl
            except Exception:
                pass
    return best_score, best_alpha, best_ppl


def get_unsteered_baseline_score(results_dir: Path, trait: str) -> float:
    """
    Returns the average unsteered baseline score (base_score) for a given trait.
    """
    trait_dir = results_dir / trait
    for method in ["cos_only", "rank_only", "logit_diff"]:
        for val in VALS:
            csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "base_score" in df.columns:
                        df["base_score"] = df["base_score"].replace(0, np.nan)
                        val_mean = df["base_score"].mean()
                        if not np.isnan(val_mean):
                            return val_mean
                except Exception:
                    pass
    return 3.0  # Fallback


def main():
    ap = argparse.ArgumentParser(description="Plot grouped bar chart comparing maximum safe DLS scores.")
    ap.add_argument("--results_dir", required=True, help="Path to results directory")
    ap.add_argument("--out_dir", required=True, help="Output folder for figures")
    ap.add_argument("--artifact_dir", default=None, help="Folder to copy results for conversation viewing")
    ap.add_argument("--title_prefix", required=True, help="Title prefix (e.g., Norm or Raw)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    title_prefix = args.title_prefix

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data for all traits
    data = []
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(results_dir, trait)
        cos_score, cos_alpha, cos_ppl = get_max_safe_score(results_dir, trait, "cos_only")
        rank_score, rank_alpha, rank_ppl = get_max_safe_score(results_dir, trait, "rank_only")
        ld_score, ld_alpha, ld_ppl = get_max_safe_score(results_dir, trait, "logit_diff")
        
        data.append({
            "trait": TRAIT_LABELS[trait],
            "unsteered": (ub_score, np.nan, np.nan),
            "cos_only": (cos_score, cos_alpha, cos_ppl),
            "rank_only": (rank_score, rank_alpha, rank_ppl),
            "logit_diff": (ld_score, ld_alpha, ld_ppl),
        })

    # Calculate averages (filter out 0.0 values)
    def clean_mean(vals):
        valid = [v for v in vals if v > 0.0]
        return np.mean(valid) if valid else 0.0

    avg_ub = clean_mean([d["unsteered"][0] for d in data])
    avg_cos = clean_mean([d["cos_only"][0] for d in data])
    avg_rank = clean_mean([d["rank_only"][0] for d in data])
    avg_ld = clean_mean([d["logit_diff"][0] for d in data])

    data.append({
        "trait": "Average",
        "unsteered": (avg_ub, np.nan, np.nan),
        "cos_only": (avg_cos, np.nan, np.nan),
        "rank_only": (avg_rank, np.nan, np.nan),
        "logit_diff": (avg_ld, np.nan, np.nan),
    })

    # Plotting setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    width = 0.20  # Width for 4 bars
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Premium colors matching each method
    color_unsteered = "#7f8c8d" # Muted Grey
    color_cos        = "#e67e22" # Premium Orange/Coral (matching cos_only)
    color_rank       = "#1abc9c" # Premium Teal/Green (matching rank_only)
    color_logit      = "#1f4e79" # Premium Deep Steel Blue (matching logit_diff)
    
    ub_vals   = [d["unsteered"][0] for d in data]
    cos_vals  = [d["cos_only"][0] for d in data]
    rank_vals = [d["rank_only"][0] for d in data]
    ld_vals   = [d["logit_diff"][0] for d in data]
    
    rects1 = ax.bar(x - 1.5 * width, ub_vals,   width, label="Unsteered Baseline", color=color_unsteered, zorder=3)
    rects2 = ax.bar(x - 0.5 * width, cos_vals,  width, label="DLS (Cos-Only)", color=color_cos, zorder=3)
    rects3 = ax.bar(x + 0.5 * width, rank_vals, width, label="DLS (Rank-Only)", color=color_rank, zorder=3)
    rects4 = ax.bar(x + 1.5 * width, ld_vals,   width, label="DLS (Logit-Diff)", color=color_logit, zorder=3)
    
    # Dashed baseline at 3.0 (unsteered neutral score)
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    
    # Title and styling
    ax.set_title(f"{title_prefix} DLS — Maximum Safe Steering Score Comparison (PPL ≤ 25.0)", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)

    # Helper function to attach labels on top of the bars
    def autolabel(rects, data_key):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height == 0.0 or np.isnan(height):
                continue
            
            # Label text
            label_text = f"{height:.2f}"
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                        color="#333333")
            
            # Show alpha value inside the bar (if applicable)
            info = data[i][data_key]
            alpha_val = info[1]
            if not np.isnan(alpha_val):
                alpha_text = f"α={alpha_val}"
                ax.annotate(alpha_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -14),
                            textcoords="offset points",
                            ha="center", va="top",
                            fontsize=7.5, color="white", fontweight="semibold", rotation=90)

    autolabel(rects1, "unsteered")
    autolabel(rects2, "cos_only")
    autolabel(rects3, "rank_only")
    autolabel(rects4, "logit_diff")

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9.5)
    
    # Save figure
    file_name = "max_safe_score_compare.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {out_path}")
    
    # Copy to artifacts
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / f"{title_prefix.lower()}_{file_name}"
        shutil.copy(out_path, dest_path)
        print(f"Copied comparison plot to artifacts: {dest_path}")

if __name__ == "__main__":
    main()
