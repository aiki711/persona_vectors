#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/112_plot_pareto_tradeoff.py
#
# Plots a 2D Pareto trade-off chart (Personality Score vs. Perplexity)
# to visualize the score-fluency trade-offs of different DLS methods.
#

import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def load_metrics(results_dir: Path, method: str):
    """
    Computes the average Score and PPL across all traits for each alpha value.
    """
    alpha_scores = []
    alpha_ppls = []
    alphas_present = []
    
    for val in VALS:
        scores = []
        ppls = []
        for trait in TRAITS:
            csv_path = results_dir / trait / f"scores_{method}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = results_dir / trait / f"scores_{method}_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "dyn_score" in df.columns:
                        df["dyn_score"] = df["dyn_score"].replace(0, np.nan)
                    scores.append(df["dyn_score"].mean())
                    ppls.append(df["dyn_ppl"].mean())
                except Exception:
                    pass
        if len(scores) == len(TRAITS):  # Only include if all traits have data for this alpha
            alpha_scores.append(np.mean(scores))
            alpha_ppls.append(np.mean(ppls))
            alphas_present.append(val)
            
    return pd.DataFrame({
        "alpha": alphas_present,
        "score": alpha_scores,
        "ppl": alpha_ppls
    })

def main():
    ap = argparse.ArgumentParser(description="Plot 2D Pareto trade-off curves for DLS methods.")
    ap.add_argument("--baseline_dir", default="exp_steering_dyn_layer_proj_prior/results_test_unseen", help="Baseline results directory")
    ap.add_argument("--pdf_dir", default="exp_steering_dyn_layer_pdf/results", help="PDF results directory")
    ap.add_argument("--out_dir", default="exp_steering_dyn_layer_pdf/figures", help="Output folder for figures")
    ap.add_argument("--artifact_dir", default=None, help="Folder to copy results to")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data for each method
    df_ld = load_metrics(baseline_dir, "logit_diff")
    df_cos = load_metrics(pdf_dir, "masked_cos_only")
    df_proj = load_metrics(pdf_dir, "masked_proj_cos_only")

    # Plot setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Pareto curves definitions
    methods_data = [
        ("Logit-Diff Baseline (Unmasked)", df_ld,   "#1f4e79", "o", "-"),
        ("PDF Cos-Only (Masked)",          df_cos,  "#e67e22", "s", "-"),
        ("PDF Proj Cos-Only (Masked)",     df_proj, "#2ecc71", "^", "-"),
    ]

    # Draw a shaded region for the "Safe Perplexity Zone" (PPL <= 25.0)
    ax.axhspan(0, 25.0, color="#2ecc71", alpha=0.08, label="Safe Generation Zone (PPL ≤ 25.0)", zorder=1)
    ax.axhline(y=25.0, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, label="Safety Threshold (PPL = 25.0)", zorder=2)
    
    for label, df, color, marker, linestyle in methods_data:
        if df.empty:
            continue
        
        # Split into safe and unsafe points
        safe_df = df[df["ppl"] <= 25.0]
        unsafe_df = df[df["ppl"] > 25.0]
        
        # Plot safe curve
        ax.plot(df["score"], df["ppl"], linestyle=linestyle, color=color, alpha=0.5, zorder=2)
        
        # Plot safe points (opaque)
        ax.scatter(safe_df["score"], safe_df["ppl"], label=label, color=color, marker=marker, s=80, edgecolors="black", linewidths=1.0, zorder=4)
        
        # Plot unsafe points (translucent with red border or 'x' marker to indicate breakdown)
        if not unsafe_df.empty:
            ax.scatter(unsafe_df["score"], unsafe_df["ppl"], color=color, marker=marker, s=60, alpha=0.25, edgecolors="#e74c3c", linewidths=1.2, zorder=3)

        # Annotate alpha values for key points
        for _, row in df.iterrows():
            alpha = row["alpha"]
            # Label only select alphas to avoid clutter (e.g. 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
            if alpha in [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
                alpha_text = f"α={alpha}"
                offset = (5, 5)
                # Adjust labels for specific points to avoid overlaps
                if label.startswith("PDF Cos-Only") and alpha == 5.0:
                    offset = (-30, 8)
                elif label.startswith("PDF Proj") and alpha == 10.0:
                    offset = (5, -12)
                
                ax.annotate(alpha_text,
                            xy=(row["score"], row["ppl"]),
                            xytext=offset, textcoords="offset points",
                            fontsize=8.5, fontweight="semibold", color="#333333", alpha=0.8)

    # Ideal zone direction indicator
    ax.annotate("Ideal Direction\n(High Score, Low PPL)",
                xy=(4.95, 6.0), xytext=(4.3, 10.0),
                arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=2, headwidth=8),
                fontsize=10, fontweight="bold", color="#27ae60", ha="center")

    # Chart decorations
    ax.set_title("DLS Methods Pareto Frontier: Personality Score vs. Perplexity (All-Trait Average)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Personality Score (Higher is Better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perplexity (PPL, Lower is Better)", fontsize=11, fontweight="bold")
    
    ax.set_xlim(3.0, 5.15)
    ax.set_ylim(4.5, 45.0)  # Focus on key perplexity range
    
    ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9.5)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Save figure
    file_name = "dls_pareto_tradeoff.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto trade-off plot to: {out_path}")
    
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / file_name
        shutil.copy(out_path, dest_path)
        print(f"Copied Pareto trade-off plot to artifacts: {dest_path}")

if __name__ == "__main__":
    main()
