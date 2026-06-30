#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/plot_token_intensity.py
#
# Aggregates evaluation results across 6 configurations for DLIS (surprisal gating)
# and generates comparison bar charts.
#

import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
RESULTS_DIR = Path("exp_token_intensity/results")
OUT_DIR = Path("exp_token_intensity/figures")
ARTIFACT_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_LABELS = {
    "extraversion":      "Extraversion",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness":     "Agreeableness",
}

CONFIGS = [
    ("conf1", "No Gating"),
    ("conf2", "Base Gating (3-7)"),
    ("conf3", "Wider (1-9)"),
    ("conf4", "Narrower (4-6)"),
    ("conf5", "Sharp (k=8)"),
    ("conf6", "Gentle (k=0.5)"),
]

METHODS = [
    ("proj_rank", "Proj Rank (Unmasked)"),
    ("masked_proj_rank", "PDF Proj Rank (Soft Masked)"),
]

# Config params map
CONFIG_PARAMS = {
    "conf1": ("0.0", "99.0", "1.0", "1.0"),
    "conf2": ("3.0", "7.0", "2.0", "2.0"),
    "conf3": ("1.0", "9.0", "2.0", "2.0"),
    "conf4": ("4.0", "6.0", "2.0", "2.0"),
    "conf5": ("3.0", "7.0", "8.0", "8.0"),
    "conf6": ("3.0", "7.0", "0.5", "0.5"),
}

# Colors
METHOD_COLORS = {
    "proj_rank": "#2ecc71",        # Emerald Green
    "masked_proj_rank": "#e84393", # Pink / Magenta
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

def load_metrics(trait: str, score_mode: str, conf_id: str) -> tuple[float, float, float]:
    # Construct filename based on config parameters
    theta_lo, theta_hi, k_lo, k_hi = CONFIG_PARAMS[conf_id]
    
    csv_name = f"scores_{score_mode}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}_Val5.0.csv"
    csv_path = RESULTS_DIR / trait / csv_name
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            score_col = "dyn_score" if "dyn_score" in df.columns else df.columns[2]
            ppl_col = "dyn_ppl" if "dyn_ppl" in df.columns else "fusion_ppl"
            reason_col = "dyn_reason" if "dyn_reason" in df.columns else "fusion_reason"
            
            mean_score = df[score_col].mean()
            # Handle possible inf/nan in perplexity
            valid_ppl = df[ppl_col][np.isfinite(df[ppl_col])]
            mean_ppl = valid_ppl.mean() if not valid_ppl.empty else 999.0
            
            coherence_rate = df[reason_col].str.contains("Coherence: Yes", case=False, na=False).mean() if reason_col in df.columns else 1.0
            return mean_score, mean_ppl, coherence_rate
        except Exception as e:
            print(f"Error reading {csv_path.name}: {e}")
    return 0.0, 999.0, 0.0

def get_unsteered_baseline_ppl(trait: str) -> float:
    # Estimate baseline PPL from any scores file
    trait_dir = RESULTS_DIR / trait
    for f in trait_dir.glob("scores_*.csv"):
        try:
            df = pd.read_csv(f)
            if "base_ppl" in df.columns:
                valid_ppl = df["base_ppl"][np.isfinite(df["base_ppl"])]
                if not valid_ppl.empty:
                    return valid_ppl.mean()
        except:
            pass
    return 10.0

def plot_bar_chart(metric_type: str, title: str, ylabel: str, y_lim=None, baseline_val=None):
    # Collect data for all configurations
    plot_data = []
    
    for trait in TRAITS:
        for conf_id, conf_name in CONFIGS:
            for score_mode, method_name in METHODS:
                score, ppl, coherence = load_metrics(trait, score_mode, conf_id)
                val = score if metric_type == "score" else (ppl if metric_type == "ppl" else coherence)
                
                plot_data.append({
                    "Trait": TRAIT_LABELS[trait],
                    "Configuration": conf_name,
                    "Method": method_name,
                    "Value": val,
                    "conf_id": conf_id,
                    "score_mode": score_mode
                })
                
    # Calculate Averages
    for conf_id, conf_name in CONFIGS:
        for score_mode, method_name in METHODS:
            vals = [d["Value"] for d in plot_data if d["conf_id"] == conf_id and d["score_mode"] == score_mode and d["Trait"] != "Average"]
            # Exclude 0.0 or 999.0 if they are failures
            valid_vals = [v for v in vals if (metric_type != "ppl" and v > 0.0) or (metric_type == "ppl" and v < 500.0)]
            avg_val = np.mean(valid_vals) if valid_vals else (0.0 if metric_type != "ppl" else 999.0)
            
            plot_data.append({
                "Trait": "Average",
                "Configuration": conf_name,
                "Method": method_name,
                "Value": avg_val,
                "conf_id": conf_id,
                "score_mode": score_mode
            })
            
    df_plot = pd.DataFrame(plot_data)
    
    # Generate overall comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(20, 16), sharex=False, sharey=False)
    axes = axes.flatten()
    
    categories = ["Average"] + [TRAIT_LABELS[t] for t in TRAITS]
    
    for idx, cat in enumerate(categories):
        ax = axes[idx]
        df_sub = df_plot[df_plot["Trait"] == cat]
        
        # Plot bars
        sns.barplot(
            data=df_sub, x="Configuration", y="Value", hue="Method",
            palette=[METHOD_COLORS["proj_rank"], METHOD_COLORS["masked_proj_rank"]],
            ax=ax, edgecolor="black", linewidth=0.5
        )
        
        # Annotate values
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8, padding=3, fontweight="bold")
            
        # Baseline line if specified
        if baseline_val is not None:
            b_val = baseline_val.get(cat.lower(), baseline_val) if isinstance(baseline_val, dict) else baseline_val
            ax.axhline(y=b_val, color="#7f8c8d", linestyle="--", linewidth=1.2, label=f"Baseline ({b_val:.2f})")
            
        ax.set_title(f"{cat}", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        if y_lim:
            ax.set_ylim(y_lim)
            
        # Only show legend for first plot
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9)
        else:
            ax.get_legend().remove()
            
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    file_name = f"gating_{metric_type}_comparison.png"
    plt.savefig(OUT_DIR / file_name, dpi=200, bbox_inches="tight")
    shutil.copy(OUT_DIR / file_name, ARTIFACT_DIR / f"gating_{metric_type}_comparison.png")
    plt.close()
    print(f"Generated gating {metric_type} comparison chart.")

def main():
    print("Aggregating metrics for Surprisal Gating (DLIS) configurations...")
    
    # Calculate baseline values
    baseline_scores = {t.lower(): 3.0 for t in TRAITS}
    baseline_scores["average"] = 3.0
    
    baseline_ppls = {}
    for trait in TRAITS:
        baseline_ppls[trait] = get_unsteered_baseline_ppl(trait)
    baseline_ppls["average"] = np.mean(list(baseline_ppls.values())) if baseline_ppls else 10.0
    
    baseline_coherence = {t.lower(): 1.0 for t in TRAITS}
    baseline_coherence["average"] = 1.0

    # Plot metrics
    plot_bar_chart("score", "DLIS Steering Score Comparison across Gating Configurations (Alpha_max = 5.0)", "Personality Steering Score", y_lim=(1.0, 5.2), baseline_val=baseline_scores)
    plot_bar_chart("ppl", "DLIS Perplexity (PPL) Comparison across Gating Configurations (Alpha_max = 5.0)", "Text Perplexity (PPL)", y_lim=(5.0, 35.0), baseline_val=baseline_ppls)
    plot_bar_chart("coherence", "DLIS Coherence Rate Comparison across Gating Configurations (Alpha_max = 5.0)", "Coherence Rate (0.0 to 1.0)", y_lim=(0.0, 1.05), baseline_val=baseline_coherence)
    
    print("All plotting for Surprisal Gating finished successfully!")

if __name__ == "__main__":
    main()
