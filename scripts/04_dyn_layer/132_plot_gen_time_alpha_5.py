#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/132_plot_gen_time_alpha_5.py
#
# Generates a comparison bar chart of steering scores at alpha = 5.0:
#   - Unsteered Baseline
#   - Prompt-level DLS Logit-Diff (alpha = 5.0)
#   - 8 generation-time DLS methods (alpha = 5.0)
#
# Outputs: exp_steering_dyn_gen_time_raw/figures/gen_time_alpha_5_comparison.png
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

# The 8 gen-time methods to load
GEN_TIME_METHODS = [
    ("Gen-Time Cos-Only",          "cos_only",               "#e67e22"),
    ("Gen-Time Rank-Only",         "rank_only",              "#2c3e50"),
    ("Gen-Time Proj Cos-Only",     "proj_cos_only",          "#3498db"),
    ("Gen-Time Proj Rank-Only",    "proj_rank_only",         "#2ecc71"),
    ("Gen-Time PDF Cos-Only",      "masked_cos_only",        "#f1c40f"),
    ("Gen-Time PDF Rank-Only",     "masked_rank_only",       "#9b59b6"),
    ("Gen-Time PDF Proj Cos-Only",  "masked_proj_cos_only",   "#e74c3c"),
    ("Gen-Time PDF Proj Rank-Only", "masked_proj_rank_only",  "#e84393"),
]

def load_score_for_val(results_dir: Path, trait: str, method: str, val: float) -> float:
    csv_path = results_dir / trait / f"scores_{method}_Val{float(val)}.csv"
    if not csv_path.exists():
        csv_path = results_dir / trait / f"scores_{method}_Val{val}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            score_col = "dyn_score" if "dyn_score" in df.columns else df.columns[2]
            df[score_col] = df[score_col].replace(0, np.nan)
            mean_score = df[score_col].mean()
            if not np.isnan(mean_score):
                return mean_score
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
    return np.nan

def get_unsteered_baseline_score(gen_results_dir: Path, trait: str) -> float:
    # Attempt to load base_score from one of the gen-time result files
    trait_dir = gen_results_dir / trait
    for _, method, _ in GEN_TIME_METHODS:
        csv_path = trait_dir / f"scores_{method}_Val5.0.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "base_score" in df.columns:
                    df["base_score"] = df["base_score"].replace(0, np.nan)
                    mean_score = df["base_score"].mean()
                    if not np.isnan(mean_score):
                        return mean_score
            except Exception:
                pass
    return 3.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_results_dir", default="exp_steering_dyn_gen_time_raw/results")
    ap.add_argument("--prompt_results_dir", default="exp_steering_dyn_layer_raw/results")
    ap.add_argument("--out_dir", default="exp_steering_dyn_gen_time_raw/figures")
    ap.add_argument("--artifact_dir", default=None)
    args = ap.parse_args()

    gen_results_dir = Path(args.gen_results_dir)
    prompt_results_dir = Path(args.prompt_results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling data for alpha = 5.0 comparison...")

    # Data structure for plotting
    data = []
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(gen_results_dir, trait)
        
        # Load prompt-level logit diff score
        prompt_ld = load_score_for_val(prompt_results_dir, trait, "logit_diff", 5.0)
        
        # Load all 8 gen-time methods
        gen_results = {}
        for display_name, method_key, _ in GEN_TIME_METHODS:
            score = load_score_for_val(gen_results_dir, trait, method_key, 5.0)
            gen_results[display_name] = score

        data.append({
            "trait": TRAIT_LABELS[trait],
            "Unsteered Baseline": ub_score,
            "Prompt DLS Logit-Diff": prompt_ld,
            **gen_results
        })

    # Compute Average across all traits
    avg_row = {
        "trait": "Average",
        "Unsteered Baseline": np.mean([d["Unsteered Baseline"] for d in data]),
        "Prompt DLS Logit-Diff": np.mean([d["Prompt DLS Logit-Diff"] for d in data if not np.isnan(d["Prompt DLS Logit-Diff"])])
    }
    for display_name, _, _ in GEN_TIME_METHODS:
        avg_row[display_name] = np.mean([d[display_name] for d in data if not np.isnan(d[display_name])])
    data.append(avg_row)

    # Plot Setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    
    # 10 bars per category: Unsteered + Prompt Logit-Diff + 8 Gen-time DLS
    num_bars = 10
    width = 0.08
    
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # Define colors
    colors = {
        "Unsteered Baseline": "#7f8c8d",
        "Prompt DLS Logit-Diff": "#95a5a6"
    }
    for display_name, _, color in GEN_TIME_METHODS:
        colors[display_name] = color

    offset_start = - (num_bars - 1) / 2.0
    rects_list = []
    labels_list = []

    # 1. Unsteered
    rects_list.append(ax.bar(x + (offset_start * width), [d["Unsteered Baseline"] for d in data], width, label="Unsteered Baseline", color=colors["Unsteered Baseline"], zorder=3))
    labels_list.append("Unsteered Baseline")

    # 2. Prompt Logit-Diff
    rects_list.append(ax.bar(x + ((offset_start + 1) * width), [d["Prompt DLS Logit-Diff"] for d in data], width, label="Prompt DLS Logit-Diff", color=colors["Prompt DLS Logit-Diff"], zorder=3))
    labels_list.append("Prompt DLS Logit-Diff")

    # 3. 8 Gen-Time methods
    for i, (display_name, _, _) in enumerate(GEN_TIME_METHODS):
        rects_list.append(ax.bar(x + ((offset_start + 2 + i) * width), [d[display_name] for d in data], width, label=display_name, color=colors[display_name], zorder=3))
        labels_list.append(display_name)

    # Styling
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_title("DLS Comparison at Alpha = 5.0 (Autoregressive Gen-Time DLS vs Prompt DLS)", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)

    # Attach values on top of the bars
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

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9, ncol=2)
    
    out_path = out_dir / "gen_time_alpha_5_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    if args.artifact_dir:
        art_dir = Path(args.artifact_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        dest_path = art_dir / "gen_time_alpha_5_comparison.png"
        shutil.copy(out_path, dest_path)
        print(f"Copied to artifact: {dest_path}")

if __name__ == "__main__":
    main()
