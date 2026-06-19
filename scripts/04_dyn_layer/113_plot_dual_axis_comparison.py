#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/113_plot_dual_axis_comparison.py
#
# Generates a premium dual-axis chart showing:
#   - Personality Score (Bars, Left Y-axis, higher is better)
#   - Perplexity / PPL (Line/Markers, Right Y-axis, lower is better)
# comparing Logit-Diff, PDF Cos-Only, and PDF Proj Cos-Only.
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

def get_best_safe_score(results_dir: Path, trait: str, method: str) -> tuple[float, float, float]:
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


def main():
    ap = argparse.ArgumentParser(description="Plot dual-axis comparison of best safe DLS configurations.")
    ap.add_argument("--baseline_dir", default="exp_steering_dyn_layer_proj_prior/results_test_unseen")
    ap.add_argument("--pdf_dir", default="exp_steering_dyn_layer_pdf/results")
    ap.add_argument("--out_dir", default="exp_steering_dyn_layer_pdf/figures")
    ap.add_argument("--artifact_dir", default=None)
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data for all traits
    data = []
    for trait in TRAITS:
        ld_score, ld_alpha, ld_ppl = get_best_safe_score(baseline_dir, trait, "logit_diff")
        cos_score, cos_alpha, cos_ppl = get_best_safe_score(pdf_dir, trait, "masked_cos_only")
        proj_score, proj_alpha, proj_ppl = get_best_safe_score(pdf_dir, trait, "masked_proj_cos_only")
        
        data.append({
            "trait": TRAIT_LABELS[trait],
            "logit_diff": (ld_score, ld_alpha, ld_ppl),
            "cos_only": (cos_score, cos_alpha, cos_ppl),
            "proj_cos": (proj_score, proj_alpha, proj_ppl),
        })

    # Calculate averages
    def clean_mean(vals):
        valid = [v for v in vals if v > 0.0]
        return np.mean(valid) if valid else 0.0

    avg_ld_score = clean_mean([d["logit_diff"][0] for d in data])
    avg_ld_ppl = clean_mean([d["logit_diff"][2] for d in data])
    avg_cos_score = clean_mean([d["cos_only"][0] for d in data])
    avg_cos_ppl = clean_mean([d["cos_only"][2] for d in data])
    avg_proj_score = clean_mean([d["proj_cos"][0] for d in data])
    avg_proj_ppl = clean_mean([d["proj_cos"][2] for d in data])

    data.append({
        "trait": "Average",
        "logit_diff": (avg_ld_score, np.nan, avg_ld_ppl),
        "cos_only": (avg_cos_score, np.nan, avg_cos_ppl),
        "proj_cos": (avg_proj_score, np.nan, avg_proj_ppl),
    })

    # Plot setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    width = 0.25  # Width for 3 bars
    
    fig, ax1 = plt.subplots(figsize=(15, 8.5))
    
    # Left Y-axis: Personality Score (Bars)
    # Colors matching the previous plots
    color_logit = "#1f4e79" # Steel Blue
    color_cos   = "#e67e22" # Orange/Coral
    color_proj  = "#2ecc71" # Green
    
    ld_scores   = [d["logit_diff"][0] for d in data]
    cos_scores  = [d["cos_only"][0] for d in data]
    proj_scores = [d["proj_cos"][0] for d in data]
    
    rects1 = ax1.bar(x - width, ld_scores,   width, label="DLS (Logit-Diff Baseline) [Score]", color=color_logit, alpha=0.85, zorder=3)
    rects2 = ax1.bar(x,         cos_scores,  width, label="PDF Cos-Only (Masked) [Score]", color=color_cos, alpha=0.85, zorder=3)
    rects3 = ax1.bar(x + width, proj_scores, width, label="PDF Proj Cos-Only (Masked) [Score]", color=color_proj, alpha=0.85, zorder=3)
    
    ax1.set_ylabel("Personality Score (Left Axis, 1.0 to 5.0, Higher is Better)", fontsize=11, fontweight="bold", color="#333333")
    ax1.set_ylim(0.8, 5.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5, color="#bbbbbb", zorder=0)
    
    # Right Y-axis: Perplexity / PPL (Line/Markers)
    ax2 = ax1.twinx()
    
    ld_ppls   = [d["logit_diff"][2] for d in data]
    cos_ppls  = [d["cos_only"][2] for d in data]
    proj_ppls = [d["proj_cos"][2] for d in data]
    
    # Plot PPL as lines with markers
    ax2.plot(x - width, ld_ppls,   marker="o", markersize=8, color="#112d47", linestyle="--", linewidth=1.5, label="Logit-Diff [PPL]", zorder=5)
    ax2.plot(x,         cos_ppls,  marker="s", markersize=8, color="#a0510c", linestyle="--", linewidth=1.5, label="PDF Cos-Only [PPL]", zorder=5)
    ax2.plot(x + width, proj_ppls, marker="^", markersize=8, color="#1e824c", linestyle="--", linewidth=1.5, label="PDF Proj Cos [PPL]", zorder=5)
    
    # Draw safety threshold on the PPL axis
    ax2.axhline(y=25.0, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.8, zorder=2)
    ax2.annotate("PPL Safety Limit (25.0)", xy=(5.2, 25.2), color="#e74c3c", fontsize=9, fontweight="bold")
    
    ax2.set_ylabel("Perplexity (Right Axis, PPL, Lower is Better)", fontsize=11, fontweight="bold", color="#333333")
    ax2.set_ylim(0.0, 30.0)  # PPL safety range
    
    # Annotate values
    def autolabel_bars(rects, data_key):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height == 0.0 or np.isnan(height):
                continue
            ax1.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color="#222222")
            
            # Show alpha value
            info = data[i][data_key]
            alpha_val = info[1]
            if not np.isnan(alpha_val):
                ax1.annotate(f"α={alpha_val}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -14), textcoords="offset points",
                            ha="center", va="top",
                            fontsize=7.5, color="white", fontweight="bold", rotation=90)

    autolabel_bars(rects1, "logit_diff")
    autolabel_bars(rects2, "cos_only")
    autolabel_bars(rects3, "proj_cos")

    # Annotate PPL values on markers
    for i in range(len(categories)):
        # Logit-Diff PPL
        if not np.isnan(ld_ppls[i]):
            ax2.annotate(f"{ld_ppls[i]:.1f}", xy=(i - width, ld_ppls[i]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#112d47")
        # PDF Cos-Only PPL
        if not np.isnan(cos_ppls[i]):
            ax2.annotate(f"{cos_ppls[i]:.1f}", xy=(i, cos_ppls[i]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#a0510c")
        # PDF Proj Cos PPL
        if not np.isnan(proj_ppls[i]):
            ax2.annotate(f"{proj_ppls[i]:.1f}", xy=(i + width, proj_ppls[i]), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#1e824c")

    # Title and Legending
    plt.title("DLS PDF Sweep — Best Safe Configuration: Score (Bars) vs. Perplexity (Lines)", fontsize=13, fontweight="bold", pad=20)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9.5)
    
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    # Save figure
    file_name = "pdf_dual_axis_comparison.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved dual-axis comparison plot to: {out_path}")
    
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / file_name
        shutil.copy(out_path, dest_path)
        print(f"Copied dual-axis comparison plot to artifacts: {dest_path}")

if __name__ == "__main__":
    main()
