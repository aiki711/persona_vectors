#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 88_plot_max_safe_score.py
#
# Generates a premium grouped bar chart comparing the maximum safe steering scores
# of Logit-Diff, Proj-Only (ablation), and Proj-Prior across all traits (and the average).
# Loads results dynamically from evaluation CSVs.
#
# Output: exp_steering_dyn_layer_proj_prior/figures/max_safe_score_compare.png
#

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
        # Check both float format and normal format
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "dyn_score" in df.columns:
                    df["dyn_score"] = df["dyn_score"].replace(0, 1)
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

def get_unsteered_baseline_score(results_dir: Path, trait: str, method: str) -> float:
    """
    Returns the average unsteered baseline score (base_score) for a given trait and method.
    """
    trait_dir = results_dir / trait
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "base_score" in df.columns:
                    df["base_score"] = df["base_score"].replace(0, 1)
                    return df["base_score"].mean()
            except Exception:
                pass
    return 3.0  # Fallback

def main():
    logit_diff_dir = Path("exp_steering_dyn_layer_all_layers_midpoint/results")
    proj_prior_dir = Path("exp_steering_dyn_layer_proj_prior/results")

    # Load data for all traits
    data = []
    for trait in TRAITS:
        # 1. Logit-Diff
        ld_score, ld_alpha, ld_ppl = get_max_safe_score(logit_diff_dir, trait, "logit_diff")
        # 2. Cos-Prior
        cp_score, cp_alpha, cp_ppl = get_max_safe_score(proj_prior_dir, trait, "cos_prior")
        # 3. Unsteered Baseline
        ub_score = get_unsteered_baseline_score(proj_prior_dir, trait, "cos_prior")
        
        data.append({
            "trait": TRAIT_LABELS[trait],
            "logit_diff": (ld_score, ld_alpha, ld_ppl),
            "cos_prior":  (cp_score, cp_alpha, cp_ppl),
            "unsteered":  (ub_score, np.nan, np.nan)
        })
        
        print(f"[{TRAIT_LABELS[trait]}]")
        print(f"  Unsteered  : Score={ub_score:.2f}")
        print(f"  Logit-Diff : Score={ld_score:.2f} (alpha={ld_alpha}, ppl={ld_ppl:.2f})")
        print(f"  Cos-Prior  : Score={cp_score:.2f} (alpha={cp_alpha}, ppl={cp_ppl:.2f})")

    # Calculate averages
    ld_scores_all = [d["logit_diff"][0] for d in data]
    cp_scores_all = [d["cos_prior"][0] for d in data]
    ub_scores_all = [d["unsteered"][0] for d in data]
    
    avg_ld = np.mean(ld_scores_all)
    avg_cp = np.mean(cp_scores_all)
    avg_ub = np.mean(ub_scores_all)
    
    data.append({
        "trait": "Average",
        "logit_diff": (avg_ld, np.nan, np.nan),
        "cos_prior":  (avg_cp, np.nan, np.nan),
        "unsteered":  (avg_ub, np.nan, np.nan)
    })
    
    print("\n[Averages]")
    print(f"  Unsteered  : {avg_ub:.2f}")
    print(f"  Logit-Diff : {avg_ld:.2f}")
    print(f"  Cos-Prior  : {avg_cp:.2f}")

    # Plot setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    width = 0.24  # Width for 3 bars

    fig, ax = plt.subplots(figsize=(15, 7.5))
    
    # Custom premium colors
    color_unsteered = "#7f8c8d" # Sleek Premium Muted Grey
    color_logit = "#1f4e79"   # Premium Deep Steel Blue
    color_cos   = "#d95f02"   # Premium Coral/Orange
    
    # Extract score values
    ub_vals = [d["unsteered"][0] for d in data]
    ld_vals = [d["logit_diff"][0] for d in data]
    cp_vals = [d["cos_prior"][0] for d in data]
    
    # Plot bars
    rects1 = ax.bar(x - width, ub_vals, width, label="Unsteered Baseline", color=color_unsteered, zorder=3)
    rects2 = ax.bar(x,         ld_vals, width, label="Logit-Diff (Baseline)", color=color_logit, zorder=3)
    rects3 = ax.bar(x + width, cp_vals, width, label="Cos-Sim (Proposed)", color=color_cos, zorder=3)

    # Dashed baseline at 3.0 (unsteered neutral score)
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    
    # Title and styling
    ax.set_title("Maximum Safe Steering Score Comparison (PPL ≤ 25.0)", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0.8, 5.25)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)

    # Helper function to attach values on top of the bars
    def autolabel(rects, data_key, is_prior=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            
            # Format text
            label_text = f"{height:.2f}"
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                        color="#333333")
            
            # Annotate alpha and improvement
            info = data[i][data_key]
            alpha_val = info[1]
            
            if not np.isnan(alpha_val) and is_prior:
                alpha_text = f"α={alpha_val}"
                # Alpha label inside the bar
                ax.annotate(alpha_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -14),  # inside the bar
                            textcoords="offset points",
                            ha="center", va="top",
                            fontsize=8, color="white", fontweight="semibold")
                
                # Improvement relative to Logit-Diff
                ld_score = data[i]["logit_diff"][0]
                diff = height - ld_score
                diff_text = f"{diff:+.2f}"
                
                imp_color = "#27ae60" if diff >= 0 else "#c0392b"
                ax.annotate(f"({diff_text})",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 16),  # higher than the score label
                            textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=8, color=imp_color, fontweight="bold")
                            
            # Add average improvement label for the Average category
            if categories[i] == "Average" and is_prior:
                ld_score = data[i]["logit_diff"][0]
                diff = height - ld_score
                diff_text = f"{diff:+.2f}"
                imp_color = "#27ae60" if diff >= 0 else "#c0392b"
                ax.annotate(f"({diff_text})",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 16),
                            textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=8, color=imp_color, fontweight="bold")

    autolabel(rects1, "unsteered")
    autolabel(rects2, "logit_diff")
    autolabel(rects3, "cos_prior", is_prior=True)

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9.5)
    
    # Save figure
    out_dir = Path("exp_steering_dyn_layer_proj_prior/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "max_safe_score_compare.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {out_path}")
    
    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/42af965e-7b98-48aa-bc1b-ea07d6f49983/images")
    if artifact_dir.exists():
        dest_path = artifact_dir / "max_safe_score_compare.png"
        shutil.copy(out_path, dest_path)
        print(f"Copied comparison plot to artifacts: {dest_path}")

if __name__ == "__main__":
    main()
