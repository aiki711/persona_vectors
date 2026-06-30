#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/124_plot_fixed_alpha_comparison.py
#
# Generates grouped bar charts comparing the steering scores of all 9 fixed-layer DLS methods
# at specific fixed values of alpha (e.g., alpha = 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0).
# If a method fails the safety criteria (Mean PPL <= 25.0, Max PPL <= 35.0, Coherence >= 80%)
# at that alpha, its bar is drawn with diagonal hatches and lower opacity.
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
VALS = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]  # Key alphas to compare

# 9 highly distinct and premium colors
METHODS = [
    ("DLS Logit-Diff",        "logit_diff",             "#1abc9c"),  # Teal
    ("DLS Cos-Only",          "cos_only",               "#e67e22"),  # Orange
    ("DLS Rank-Only",         "rank_only",              "#2c3e50"),  # Midnight Navy
    ("DLS Proj Cos-Only",     "proj_cos_only",          "#3498db"),  # Light Blue
    ("DLS Proj Rank-Only",    "proj_rank_only",         "#2ecc71"),  # Emerald Green
    ("PDF Cos-Only",          "masked_cos_only",        "#f1c40f"),  # Yellow
    ("PDF Rank-Only",         "masked_rank_only",       "#9b59b6"),  # Purple
    ("PDF Proj Cos-Only",     "masked_proj_cos_only",   "#e74c3c"),  # Red
    ("PDF Proj Rank-Only",    "masked_proj_rank_only",  "#e84393"),  # Pink / Magenta
]

def load_score_and_safety(results_dir: Path, trait: str, method: str, alpha: float) -> tuple[float, bool]:
    trait_dir = results_dir / trait
    csv_path = trait_dir / f"scores_{method}_Val{float(alpha)}.csv"
    if not csv_path.exists():
        csv_path = trait_dir / f"scores_{method}_Val{val}.csv" if 'val' in locals() else trait_dir / f"scores_{method}_Val{alpha}.csv"
        
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Find score column
            if "dyn_score" in df.columns:
                score_col = "dyn_score"
            elif "fusion_score" in df.columns:
                score_col = "fusion_score"
            else:
                score_col = df.columns[2]
            
            df[score_col] = df[score_col].replace(0, np.nan)
            mean_score = df[score_col].mean()
            
            # Safety check
            ppl_col = "dyn_ppl" if "dyn_ppl" in df.columns else "fusion_ppl"
            mean_ppl = df[ppl_col].mean()
            max_ppl = df[ppl_col].max()
            
            reason_col = "dyn_reason" if "dyn_reason" in df.columns else "fusion_reason"
            if reason_col in df.columns:
                coherence_rate = df[reason_col].str.contains("Coherence: Yes", case=False, na=False).mean()
            else:
                coherence_rate = 1.0
                
            is_safe = (
                mean_ppl <= 25.0 and
                coherence_rate >= 0.8 and
                max_ppl <= 35.0
            )
            return mean_score, is_safe
        except Exception:
            pass
    return 0.0, False

def get_unsteered_baseline_score(results_dir: Path, trait: str) -> float:
    trait_dir = results_dir / trait
    for display_name, loader_key, _ in METHODS:
        for val in [1.0, 2.0, 4.0]:
            csv_path = trait_dir / f"scores_{loader_key}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_{loader_key}_Val{val}.csv"
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

def plot_for_alpha(results_dir: Path, out_dir: Path, artifact_dir: Path, alpha: float):
    print(f"Generating comparison chart for alpha = {alpha}...")
    data = []
    
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(results_dir, trait)
        method_results = {}
        for display_name, loader_key, _ in METHODS:
            score, is_safe = load_score_and_safety(results_dir, trait, loader_key, alpha)
            method_results[display_name] = (score, is_safe)
            
        data.append({
            "trait": TRAIT_LABELS[trait],
            "Unsteered Baseline": (ub_score, True),
            **method_results
        })
        
    # Calculate average
    avg_ub = np.mean([d["Unsteered Baseline"][0] for d in data])
    avg_results = {}
    for display_name, loader_key, _ in METHODS:
        scores = [d[display_name][0] for d in data if d[display_name][0] > 0.0]
        safeties = [d[display_name][1] for d in data if d[display_name][0] > 0.0]
        avg_score = np.mean(scores) if scores else 0.0
        # Average is considered "safe" only if all traits are safe at this alpha
        is_all_safe = all(safeties) if safeties else False
        avg_results[display_name] = (avg_score, is_all_safe)
        
    data.append({
        "trait": "Average",
        "Unsteered Baseline": (avg_ub, True),
        **avg_results
    })
    
    # Plotting
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    
    num_bars = 1 + len(METHODS)
    width = 0.08
    
    fig, ax = plt.subplots(figsize=(24, 10))
    
    colors = {
        "Unsteered Baseline": "#7f8c8d"  # Grey
    }
    for display_name, _, color in METHODS:
        colors[display_name] = color
        
    offset_start = - (num_bars - 1) / 2.0
    
    # Plot Unsteered
    ax.bar(x + (offset_start * width), [d["Unsteered Baseline"][0] for d in data], width, 
           label="Unsteered Baseline", color=colors["Unsteered Baseline"], zorder=3)
           
    # Plot methods
    for bar_idx, (display_name, _, color) in enumerate(METHODS):
        scores = []
        opacities = []
        hatches = []
        for d in data:
            score, is_safe = d[display_name]
            scores.append(score)
            if is_safe:
                opacities.append(1.0)
                hatches.append("")
            else:
                opacities.append(0.4)
                hatches.append("//")
                
        # Draw bars individually to handle distinct safety shading/hatches
        x_offsets = x + ((offset_start + 1 + bar_idx) * width)
        for idx in range(len(data)):
            bar_obj = ax.bar(x_offsets[idx], scores[idx], width, 
                             color=color, alpha=opacities[idx], hatch=hatches[idx], edgecolor="black", 
                             linewidth=0.5 if hatches[idx] else 0.0, zorder=3,
                             label=display_name if idx == 0 else "") # Avoid duplicate labels in legend
            
            # Annotate score value
            if scores[idx] > 0.0:
                ax.annotate(f"{scores[idx]:.2f}",
                     xy=(x_offsets[idx], scores[idx]),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha="center", va="bottom",
                     fontsize=7, fontweight="bold",
                     color="#333333")
                
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    
    title_text = f"Fixed-Layer DLS Steering Score Comparison (Alpha = {alpha}) [N-gram-Free Safety]\n(Hatched Bars with Opacity indicate unsafe configurations at this alpha)"
    ax.set_title(title_text, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)
    
    # Place legend
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9, ncol=2)
    
    # Save figure
    file_name = f"score_compare_alpha_{alpha}.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / file_name
        shutil.copy(out_path, dest_path)
        print(f"Copied to artifact: {dest_path}")

def main():
    ap = argparse.ArgumentParser(description="Plot DLS score comparisons at fixed alpha values.")
    ap.add_argument("--results_dir", default="exp_steering_dyn_layer_raw/results")
    ap.add_argument("--out_dir", default="exp_steering_dyn_layer_raw/figures/fixed_alpha")
    ap.add_argument("--artifact_dir", default="/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    args = ap.parse_args()
    
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for alpha in VALS:
        plot_for_alpha(results_dir, out_dir, artifact_dir, alpha)
        
    print("\nAll fixed alpha plots generated successfully.")

if __name__ == "__main__":
    main()
