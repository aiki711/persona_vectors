#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/111_plot_pdf_max_safe_bar.py
#
# Generates a grouped bar chart comparing the maximum safe steering scores (PPL ≤ 25.0)
# for the PDF sweep (unsteered, logit_diff baseline, and all 6 PDF methods).
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

# Methods to plot: (display_name, loader_key)
METHODS = [
    ("PDF Cos-Only",         "masked_cos_only"),
    ("PDF Rank-Only",        "masked_rank_only"),
    ("PDF Proj Cos-Only",    "masked_proj_cos_only"),
    ("PDF Proj Rank-Only",   "masked_proj_rank_only"),
]

def get_max_safe_score(results_dir: Path, trait: str, method: str) -> tuple[float, float, float]:
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
                
                if mean_ppl <= 25.0:
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = val
                        best_ppl = mean_ppl
            except Exception:
                pass
    return best_score, best_alpha, best_ppl


def get_unsteered_baseline_score(results_dir: Path, trait: str) -> float:
    trait_dir = results_dir / trait
    for method in ["logit_diff", "masked_cos_only", "masked_rank_only"]:
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
    return 3.0


def main():
    ap = argparse.ArgumentParser(description="Plot grouped bar chart comparing maximum safe PDF DLS scores.")
    ap.add_argument("--baseline_dir", default="exp_steering_dyn_layer_proj_prior/results", help="Baseline logit_diff results directory")
    ap.add_argument("--pdf_dir", default="exp_steering_dyn_layer_pdf/results", help="PDF results directory")
    ap.add_argument("--out_dir", default="exp_steering_dyn_layer_pdf/figures", help="Output directory for figures")
    ap.add_argument("--artifact_dir", default=None, help="Folder to copy results to")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = []
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(pdf_dir, trait)
        ld_score, ld_alpha, ld_ppl = get_max_safe_score(baseline_dir, trait, "logit_diff")
        
        # Load PDF methods
        pdf_methods_results = {}
        for display_name, loader_key in METHODS:
            score, alpha, ppl = get_max_safe_score(pdf_dir, trait, loader_key)
            pdf_methods_results[display_name] = (score, alpha, ppl)
            
        data.append({
            "trait": TRAIT_LABELS[trait],
            "unsteered": (ub_score, np.nan, np.nan),
            "logit_diff": (ld_score, ld_alpha, ld_ppl),
            **pdf_methods_results
        })

    # Calculate averages
    def clean_mean(vals):
        valid = [v for v in vals if v > 0.0]
        return np.mean(valid) if valid else 0.0

    avg_ub = clean_mean([d["unsteered"][0] for d in data])
    avg_ld = clean_mean([d["logit_diff"][0] for d in data])
    
    avg_pdf_results = {}
    for display_name, _ in METHODS:
        avg_pdf_results[display_name] = (clean_mean([d[display_name][0] for d in data]), np.nan, np.nan)

    data.append({
        "trait": "Average",
        "unsteered": (avg_ub, np.nan, np.nan),
        "logit_diff": (avg_ld, np.nan, np.nan),
        **avg_pdf_results
    })

    # Plot setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    
    # 8 bars in total (Unsteered, Logit-Diff, 6 PDF methods)
    num_bars = 2 + len(METHODS)
    width = 0.10
    
    fig, ax = plt.subplots(figsize=(20, 9))
    
    # Premium color palette
    colors = {
        "unsteered": "#7f8c8d",  # Grey
        "logit_diff": "#1f4e79",  # Dark Blue
        "PDF Cos-Only": "#f39c12",
        "PDF Rank-Only": "#9b59b6",
        "PDF Proj Cos-Only": "#e74c3c",
        "PDF Proj Rank-Only": "#1abc9c",
        "PDF Proj Cos-Prior": "#9b59b6",
        "PDF Proj Rank-Prior": "#2ecc71",
    }
    
    # Plot bars
    offset_start = - (num_bars - 1) / 2.0
    
    rects_list = []
    labels_list = []
    
    # 1. Unsteered
    rects_list.append(ax.bar(x + (offset_start * width), [d["unsteered"][0] for d in data], width, label="Unsteered Baseline", color=colors["unsteered"], zorder=3))
    labels_list.append("unsteered")
    
    # 2. Logit-Diff
    rects_list.append(ax.bar(x + ((offset_start + 1) * width), [d["logit_diff"][0] for d in data], width, label="DLS (Logit-Diff Baseline)", color=colors["logit_diff"], zorder=3))
    labels_list.append("logit_diff")
    
    # 3. PDF Methods
    for i, (display_name, _) in enumerate(METHODS):
        rects_list.append(ax.bar(x + ((offset_start + 2 + i) * width), [d[display_name][0] for d in data], width, label=display_name, color=colors.get(display_name, "#34495e"), zorder=3))
        labels_list.append(display_name)
        
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    
    # Title and styling
    ax.set_title("PDF DLS — Maximum Safe Steering Score Comparison (PPL ≤ 25.0)", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)

    # Attach labels
    for r_idx, rects in enumerate(rects_list):
        data_key = labels_list[r_idx]
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height == 0.0 or np.isnan(height):
                continue
            
            label_text = f"{height:.2f}"
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
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
                            fontsize=6.5, color="white", fontweight="semibold", rotation=90)

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9)
    
    # Save figure
    file_name = "max_safe_score_compare.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {out_path}")
    
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / f"pdf_{file_name}"
        shutil.copy(out_path, dest_path)
        print(f"Copied comparison plot to artifacts: {dest_path}")

if __name__ == "__main__":
    main()
