#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/plot_high_intensity.py
#
# Plots Steering Score, PPL, and Coherence Rate as a function of alpha_max.
#

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

WORKSPACE = Path("/home/s2550009/persona_vectors")
RESULTS_DIR = WORKSPACE / "exp_token_intensity/results"
FIGURES_DIR = WORKSPACE / "exp_token_intensity/figures"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHAS = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]

THETA_LO = 3.0
THETA_HI = 7.0
K_LO = 0.5
K_HI = 0.5

def get_metrics_for_alpha(alpha):
    scores = []
    ppls = []
    cohs = []
    
    for trait in TRAITS:
        csv_name = f"scores_masked_proj_rank_theta_{THETA_LO}_{THETA_HI}_k_{K_LO}_{K_HI}_Val{alpha}.csv"
        csv_path = RESULTS_DIR / trait / csv_name
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Score
                scores.append(df["dyn_score"].mean())
                # PPL (filter out inf/nan)
                valid_ppl = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])]
                if not valid_ppl.empty:
                    ppls.append(valid_ppl.mean())
                else:
                    ppls.append(999.0)
                # Coherence
                if "dyn_reason" in df.columns:
                    coh_rate = df["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean()
                    cohs.append(coh_rate)
                else:
                    cohs.append(1.0)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            print(f"Warning: {csv_path} does not exist yet.")
            
    if scores:
        return np.mean(scores), np.mean(ppls), np.mean(cohs)
    return None

def main():
    print("Aggregating metrics for high-intensity sweep...")
    plot_data = []
    for alpha in ALPHAS:
        res = get_metrics_for_alpha(alpha)
        if res is not None:
            mean_score, mean_ppl, mean_coh = res
            plot_data.append({
                'alpha': alpha,
                'score': mean_score,
                'ppl': mean_ppl,
                'coherence': mean_coh
            })
            print(f"Alpha={alpha:.1f} -> Score: {mean_score:.2f}, PPL: {mean_ppl:.2f}, Coherence: {mean_coh:.2%}")

    if not plot_data:
        print("No data available to plot.")
        return

    df_plot = pd.DataFrame(plot_data)

    # Plot 1: Score & PPL on dual-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = '#1f77b4'
    ax1.set_xlabel('Maximum Steering Intensity (alpha_max)')
    ax1.set_ylabel('Steering Alignment Score (1.0 - 5.0)', color=color)
    line1 = ax1.plot(df_plot['alpha'], df_plot['score'], color=color, marker='o', linewidth=2.5, label='Steering Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(1.0, 5.0)

    ax2 = ax1.twinx()  
    color = '#d62728'
    ax2.set_ylabel('Text Perplexity (PPL)', color=color)
    line2 = ax2.plot(df_plot['alpha'], df_plot['ppl'], color=color, marker='s', linestyle='--', linewidth=2, label='Perplexity (PPL)')
    ax2.tick_params(axis='y', labelcolor=color)
    # Baseline PPL reference line
    ax2.axhline(8.5, color='gray', linestyle=':', alpha=0.7, label='Baseline PPL (~8.5)')

    # Add legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('DLIS Gentle Gating: Steering Score and PPL vs. Alpha Max', pad=20)
    fig.tight_layout()
    
    fig_path = FIGURES_DIR / "high_intensity_score_ppl.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")
    if ARTIFACTS_DIR.exists():
        import shutil
        shutil.copy(fig_path, ARTIFACTS_DIR / "high_intensity_score_ppl.png")
        print("Copied to artifacts folder.")

    plt.close()

    # Plot 2: Coherence Rate
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot['alpha'], df_plot['coherence'] * 100, color='#2ca02c', marker='D', linewidth=2.5, label='Coherence Rate')
    plt.xlabel('Maximum Steering Intensity (alpha_max)')
    plt.ylabel('Coherence Rate (%)')
    plt.title('DLIS Gentle Gating: Coherence Rate vs. Alpha Max', pad=20)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()

    fig_path_coh = FIGURES_DIR / "high_intensity_coherence.png"
    plt.savefig(fig_path_coh, dpi=300)
    print(f"Saved: {fig_path_coh}")
    if ARTIFACTS_DIR.exists():
        import shutil
        shutil.copy(fig_path_coh, ARTIFACTS_DIR / "high_intensity_coherence.png")
        print("Copied to artifacts folder.")

    plt.close()
    print("All plotting for high intensity sweep finished successfully!")

if __name__ == "__main__":
    main()
