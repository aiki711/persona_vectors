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
RESULTS_DIR = WORKSPACE / "exp_token_intensity/exp_symmetric/results"
FIGURES_DIR = WORKSPACE / "exp_token_intensity/exp_symmetric/figures"
NO_GATING_DIR = WORKSPACE / "exp_layer_selection/exp_steering_dyn_layer_raw/results"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
ALPHAS = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

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
                scores.append(df["dyn_score"].mean())
                valid_ppl = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])]
                if not valid_ppl.empty:
                    ppls.append(valid_ppl.mean())
                else:
                    ppls.append(999.0)
                if "dyn_reason" in df.columns:
                    coh_rate = df["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean()
                    cohs.append(coh_rate)
                else:
                    cohs.append(1.0)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            pass
            
    if scores:
        return np.mean(scores), np.mean(ppls), np.mean(cohs)
    return None

def get_no_gating_metrics_for_alpha(alpha):
    scores = []
    ppls = []
    cohs = []
    
    for trait in TRAITS:
        csv_name = f"scores_masked_proj_rank_only_Val{alpha}.csv"
        csv_path = NO_GATING_DIR / trait / csv_name
        
        if not csv_path.exists():
            csv_name = f"scores_masked_proj_rank_only_Val{int(alpha)}.csv"
            csv_path = NO_GATING_DIR / trait / csv_name
            
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                scores.append(df["dyn_score"].mean())
                valid_ppl = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])]
                if not valid_ppl.empty:
                    ppls.append(valid_ppl.mean())
                else:
                    ppls.append(999.0)
                if "dyn_reason" in df.columns:
                    coh_rate = df["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean()
                    cohs.append(coh_rate)
                else:
                    cohs.append(1.0)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
            
    if scores:
        return np.mean(scores), np.mean(ppls), np.mean(cohs)
    return None

def get_dyn_no_gating_metrics_for_alpha(alpha):
    scores = []
    ppls = []
    cohs = []
    
    for trait in TRAITS:
        # Dynamic No Gating has theta_lo=0.0, theta_hi=99.0, k_lo=1.0, k_hi=1.0
        csv_name = f"scores_masked_proj_rank_theta_0.0_99.0_k_1.0_1.0_Val{alpha}.csv"
        csv_path = RESULTS_DIR / trait / csv_name
        
        if not csv_path.exists():
            csv_name = f"scores_masked_proj_rank_theta_0.0_99.0_k_1.0_1.0_Val{int(alpha)}.csv"
            csv_path = RESULTS_DIR / trait / csv_name
            
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                scores.append(df["dyn_score"].mean())
                valid_ppl = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])]
                if not valid_ppl.empty:
                    ppls.append(valid_ppl.mean())
                else:
                    ppls.append(999.0)
                if "dyn_reason" in df.columns:
                    coh_rate = df["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean()
                    cohs.append(coh_rate)
                else:
                    cohs.append(1.0)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
            
    if scores:
        return np.mean(scores), np.mean(ppls), np.mean(cohs)
    return None

def main():
    print("Aggregating metrics for Gentle Gating (Symmetric)...")
    gentle_data = []
    for alpha in ALPHAS:
        res = get_metrics_for_alpha(alpha)
        if res is not None:
            mean_score, mean_ppl, mean_coh = res
            gentle_data.append({
                'alpha': alpha,
                'score': mean_score,
                'ppl': mean_ppl,
                'coherence': mean_coh
            })
            print(f"Gentle Gating Alpha={alpha:.1f} -> Score: {mean_score:.2f}, PPL: {mean_ppl:.2f}, Coherence: {mean_coh:.2%}")

    print("Aggregating metrics for Dynamic No Gating...")
    dyn_no_gating_data = []
    NO_GATING_ALPHAS = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    for alpha in NO_GATING_ALPHAS:
        res = get_dyn_no_gating_metrics_for_alpha(alpha)
        if res is not None:
            mean_score, mean_ppl, mean_coh = res
            dyn_no_gating_data.append({
                'alpha': alpha,
                'score': mean_score,
                'ppl': mean_ppl,
                'coherence': mean_coh
            })
            print(f"Dyn No Gating Alpha={alpha:.1f} -> Score: {mean_score:.2f}, PPL: {mean_ppl:.2f}, Coherence: {mean_coh:.2%}")

    print("Aggregating metrics for Static No Gating...")
    static_no_gating_data = []
    for alpha in NO_GATING_ALPHAS:
        res = get_no_gating_metrics_for_alpha(alpha)
        if res is not None:
            mean_score, mean_ppl, mean_coh = res
            static_no_gating_data.append({
                'alpha': alpha,
                'score': mean_score,
                'ppl': mean_ppl,
                'coherence': mean_coh
            })
            print(f"Static No Gating Alpha={alpha:.1f} -> Score: {mean_score:.2f}, PPL: {mean_ppl:.2f}, Coherence: {mean_coh:.2%}")

    if not gentle_data:
        print("No Gentle Gating data available to plot.")
        return

    df_gentle = pd.DataFrame(gentle_data)
    df_dyn_no_gating = pd.DataFrame(dyn_no_gating_data) if dyn_no_gating_data else None
    df_static_no_gating = pd.DataFrame(static_no_gating_data) if static_no_gating_data else None

    # Plot 1: Score & PPL on dual-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_gentle_score = '#1f77b4'       # Dark blue
    color_dyn_no_gating_score = '#17becf' # Cyan / Teal
    color_static_no_gating_score = '#aec7e8' # Light blue
    
    ax1.set_xlabel('Maximum Steering Intensity (alpha_max)')
    ax1.set_ylabel('Steering Alignment Score (1.0 - 5.0)', color='#1f77b4')
    
    line1 = ax1.plot(df_gentle['alpha'], df_gentle['score'], color=color_gentle_score, marker='o', linewidth=2.5, label='Dynamic Gentle Gating Score')
    lines = line1
    
    if df_dyn_no_gating is not None:
        line2 = ax1.plot(df_dyn_no_gating['alpha'], df_dyn_no_gating['score'], color=color_dyn_no_gating_score, marker='d', linestyle='--', linewidth=2, label='Dynamic No Gating Score')
        lines += line2
        
    if df_static_no_gating is not None:
        line3 = ax1.plot(df_static_no_gating['alpha'], df_static_no_gating['score'], color=color_static_no_gating_score, marker='s', linestyle=':', linewidth=2, label='Static No Gating Score')
        lines += line3
        
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(1.0, 5.0)

    ax2 = ax1.twinx()  
    color_gentle_ppl = '#d62728'          # Dark red
    color_dyn_no_gating_ppl = '#ff7f0e'   # Orange
    color_static_no_gating_ppl = '#ff9896' # Light red/pink
    
    ax2.set_ylabel('Text Perplexity (PPL)', color='#d62728')
    
    line4 = ax2.plot(df_gentle['alpha'], df_gentle['ppl'], color=color_gentle_ppl, marker='o', linestyle='-', linewidth=2.5, label='Dynamic Gentle Gating PPL')
    lines += line4
    
    if df_dyn_no_gating is not None:
        line5 = ax2.plot(df_dyn_no_gating['alpha'], df_dyn_no_gating['ppl'], color=color_dyn_no_gating_ppl, marker='d', linestyle='--', linewidth=2, label='Dynamic No Gating PPL')
        lines += line5
        
    if df_static_no_gating is not None:
        line6 = ax2.plot(df_static_no_gating['alpha'], df_static_no_gating['ppl'], color=color_static_no_gating_ppl, marker='s', linestyle=':', linewidth=2, label='Static No Gating PPL')
        lines += line6
        
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0.0, 35.0)
    
    # Baseline PPL reference line and legend
    baseline_line = ax2.axhline(8.5, color='gray', linestyle=':', alpha=0.7, label='Baseline PPL (~8.5)')
    lines = lines + [baseline_line]
    labels = [l.get_label() for l in lines]
    
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    plt.title('DLIS Gating Comparison: Steering Score and PPL vs. Alpha Max', pad=20)
    
    fig_path = FIGURES_DIR / "high_intensity_score_ppl.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    if ARTIFACTS_DIR.exists():
        import shutil
        shutil.copy(fig_path, ARTIFACTS_DIR / "high_intensity_score_ppl.png")
        print("Copied to artifacts folder.")

    plt.close()

    # Plot 2: Coherence Rate
    plt.figure(figsize=(10, 5))
    plt.plot(df_gentle['alpha'], df_gentle['coherence'] * 100, color='#2ca02c', marker='o', linewidth=2.5, label='Dynamic Gentle Gating Coherence')
    
    if df_dyn_no_gating is not None:
        plt.plot(df_dyn_no_gating['alpha'], df_dyn_no_gating['coherence'] * 100, color='#bcbd22', marker='d', linestyle='--', linewidth=2, label='Dynamic No Gating Coherence')
        
    if df_static_no_gating is not None:
        plt.plot(df_static_no_gating['alpha'], df_static_no_gating['coherence'] * 100, color='#a1d99b', marker='s', linestyle=':', linewidth=2, label='Static No Gating Coherence')
        
    plt.xlabel('Maximum Steering Intensity (alpha_max)')
    plt.ylabel('Coherence Rate (%)')
    plt.title('DLIS Gating Comparison: Coherence Rate vs. Alpha Max', pad=20)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True)

    fig_path_coh = FIGURES_DIR / "high_intensity_coherence.png"
    plt.savefig(fig_path_coh, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path_coh}")
    if ARTIFACTS_DIR.exists():
        import shutil
        shutil.copy(fig_path_coh, ARTIFACTS_DIR / "high_intensity_coherence.png")
        print("Copied to artifacts folder.")

    plt.close()
    print("All plotting for high intensity sweep finished successfully!")

if __name__ == "__main__":
    main()
