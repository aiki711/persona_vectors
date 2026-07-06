#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/plot_static_layer_configs.py
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
WORKSPACE = Path("/home/s2550009/persona_vectors")
RESULTS_DIR = WORKSPACE / "exp_token_intensity/exp_static_layer/results"
FIGURES_DIR = WORKSPACE / "exp_token_intensity/exp_static_layer/figures"
BASELINE_DIR = WORKSPACE / "exp_layer_selection/exp_steering_dyn_layer_raw/results"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

METHODS = [
    ("proj_rank", "DLS Proj Rank-Only"),
    ("masked_proj_cosine", "PDF Proj Cos-Only"),
    ("masked_proj_rank", "PDF Proj Rank-Only")
]

CONFIG_PARAMS = {
    "conf1": ("0.0", "99.0", "1.0", "1.0"),
    "conf2": ("3.0", "7.0", "2.0", "2.0"),
    "conf3": ("1.0", "9.0", "2.0", "2.0"),
    "conf4": ("4.0", "6.0", "2.0", "2.0"),
    "conf5": ("3.0", "7.0", "8.0", "8.0"),
    "conf6": ("3.0", "7.0", "0.5", "0.5"),
}

CONFIG_LABELS = {
    "conf1": "Conf 1\n(No Gating)",
    "conf2": "Conf 2\n(Base: 3-7)",
    "conf3": "Conf 3\n(Wider: 1-9)",
    "conf4": "Conf 4\n(Narrow: 4-6)",
    "conf5": "Conf 5\n(Sharp: k=8)",
    "conf6": "Conf 6\n(Gentle: k=0.5)",
}

METHOD_COLORS = {
    "proj_rank": "#3498db",           # Sleek Blue
    "masked_proj_cosine": "#e67e22",  # Warm Orange
    "masked_proj_rank": "#9b59b6"     # Amethyst Purple
}

def load_method_metrics(score_mode_prefix: str, conf_id: str) -> tuple[float, float]:
    """Loads metrics for a specific method and configuration, averaged over all traits."""
    scores, ppls = [], []
    theta_lo, theta_hi, k_lo, k_hi = CONFIG_PARAMS[conf_id]
    
    for trait in TRAITS:
        csv_name = f"scores_{score_mode_prefix}_theta_{theta_lo}_{theta_hi}_k_{k_lo}_{k_hi}_Val5.0.csv"
        csv_path = RESULTS_DIR / trait / csv_name
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                score_col = "dyn_score" if "dyn_score" in df.columns else df.columns[2]
                ppl_col = "dyn_ppl" if "dyn_ppl" in df.columns else "fusion_ppl"
                
                scores.append(df[score_col].mean())
                valid_ppl = df[ppl_col][np.isfinite(df[ppl_col])]
                if not valid_ppl.empty:
                    ppls.append(valid_ppl.mean())
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                
    mean_score = np.mean(scores) if scores else 0.0
    mean_ppl = np.mean(ppls) if ppls else 999.0
    return mean_score, mean_ppl

def load_baselines() -> tuple[float, float, float, float]:
    """Loads unsteered baseline and logit-diff steering baseline, averaged over all traits."""
    base_scores, base_ppls = [], []
    ld_scores, ld_ppls = [], []
    
    for trait in TRAITS:
        csv_path = BASELINE_DIR / trait / "scores_logit_diff_Val5.0.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                base_scores.append(df["base_score"].mean())
                base_ppls.append(df["base_ppl"].mean())
                ld_scores.append(df["dyn_score"].mean())
                ld_ppls.append(df["dyn_ppl"].mean())
            except Exception as e:
                print(f"Error loading baseline {csv_path}: {e}")
                
    return np.mean(base_scores), np.mean(base_ppls), np.mean(ld_scores), np.mean(ld_ppls)

def main():
    print("Aggregating metrics for Static Layer configurations...")
    
    # Load baselines
    unsteered_score, unsteered_ppl, logit_diff_score, logit_diff_ppl = load_baselines()
    print(f"Baselines:")
    print(f"  Unsteered -> Score: {unsteered_score:.3f}, PPL: {unsteered_ppl:.3f}")
    print(f"  Logit-diff -> Score: {logit_diff_score:.3f}, PPL: {logit_diff_ppl:.3f}")
    
    # Collect data for bar charts
    # shape: [num_methods, num_configs]
    score_matrix = []
    ppl_matrix = []
    
    for (prefix, name) in METHODS:
        method_scores = []
        method_ppls = []
        for conf in CONFIG_PARAMS.keys():
            s, p = load_method_metrics(prefix, conf)
            method_scores.append(s)
            method_ppls.append(p)
        score_matrix.append(method_scores)
        ppl_matrix.append(method_ppls)
        
    x_labels = [CONFIG_LABELS[c] for c in CONFIG_PARAMS.keys()]
    num_configs = len(x_labels)
    num_methods = len(METHODS)
    
    # Plot parameters
    x = np.arange(num_configs)
    width = 0.25  # width of each bar
    
    # ------------------ Plot 1: Steering Alignment Score Bar Chart ------------------
    plt.figure(figsize=(10, 6))
    for i, (prefix, name) in enumerate(METHODS):
        offset = (i - (num_methods - 1) / 2) * width
        plt.bar(x + offset, score_matrix[i], width, label=name, color=METHOD_COLORS[prefix], edgecolor='black', alpha=0.9)
        
    # Reference lines for baselines
    plt.axhline(unsteered_score, color='#7f8c8d', linestyle='--', linewidth=1.8, label=f"Unsteered Baseline ({unsteered_score:.2f})")
    plt.axhline(logit_diff_score, color='#e74c3c', linestyle='-.', linewidth=1.8, label=f"Logit-diff Baseline ({logit_diff_score:.2f})")
    
    plt.ylabel('Steering Alignment Score (1.0 - 5.0)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Steering Alignment Score by Gating Config (Static Layer Selection)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, x_labels, fontsize=10)
    plt.ylim(1.0, 5.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    
    score_fig_path = FIGURES_DIR / "static_layer_score_comparison.png"
    plt.savefig(score_fig_path, dpi=300)
    print(f"Saved: {score_fig_path}")
    if ARTIFACTS_DIR.exists():
        import shutil
        shutil.copy(score_fig_path, ARTIFACTS_DIR / score_fig_path.name)
        print(f"Copied to artifacts: {ARTIFACTS_DIR / score_fig_path.name}")
        
    # ------------------ Plot 2: Perplexity (PPL) Bar Chart ------------------
    plt.figure(figsize=(10, 6))
    for i, (prefix, name) in enumerate(METHODS):
        offset = (i - (num_methods - 1) / 2) * width
        # Clip perplexity to 30.0 for visualization safety
        clipped_ppls = [min(p, 30.0) for p in ppl_matrix[i]]
        plt.bar(x + offset, clipped_ppls, width, label=name, color=METHOD_COLORS[prefix], edgecolor='black', alpha=0.9)
        
    # Reference lines for baselines
    plt.axhline(unsteered_ppl, color='#7f8c8d', linestyle='--', linewidth=1.8, label=f"Unsteered Baseline ({unsteered_ppl:.2f})")
    plt.axhline(logit_diff_ppl, color='#e74c3c', linestyle='-.', linewidth=1.8, label=f"Logit-diff Baseline ({logit_diff_ppl:.2f})")
    
    plt.ylabel('Text Perplexity (PPL)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Text Perplexity by Gating Config (Static Layer Selection)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, x_labels, fontsize=10)
    plt.ylim(0, 20.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    ppl_fig_path = FIGURES_DIR / "static_layer_ppl_comparison.png"
    plt.savefig(ppl_fig_path, dpi=300)
    print(f"Saved: {ppl_fig_path}")
    if ARTIFACTS_DIR.exists():
        import shutil
        shutil.copy(ppl_fig_path, ARTIFACTS_DIR / ppl_fig_path.name)
        print(f"Copied to artifacts: {ARTIFACTS_DIR / ppl_fig_path.name}")

if __name__ == "__main__":
    main()
