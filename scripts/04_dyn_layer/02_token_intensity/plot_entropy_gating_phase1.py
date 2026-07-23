#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/plot_entropy_gating_phase1.py
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# Paths
WORKSPACE = Path("/home/s2550009/persona_vectors")
RESULTS_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
THETA_LIST = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8]
K_LIST = [1.5, 4.0, 8.0]

def load_metrics(theta_lo: float, k_lo: float, trait: str) -> tuple[float, float]:
    csv_name = f"scores_masked_proj_rank_theta_{theta_lo:.1f}_7.0_k_{k_lo:.1f}_2.0_entropy_Val5.0.csv"
    # Special check for format difference if any
    csv_path = RESULTS_DIR / trait / csv_name
    if not csv_path.exists():
        # Retry with potential string formats
        csv_name = f"scores_masked_proj_rank_theta_{theta_lo}_7.0_k_{k_lo}_2.0_entropy_Val5.0.csv"
        csv_path = RESULTS_DIR / trait / csv_name
        
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            score = df["dyn_score"].mean()
            ppl = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean()
            return score, ppl
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    return 0.0, 999.0

def plot_heatmap(data, title, xlabel, ylabel, xticks, yticks, filename, cmap, fmt=".3f", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Show all ticks
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    # Label them with the respective list entries
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_yticklabels(yticks, fontsize=10)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            val = data[i, j]
            # Simple threshold for text color contrast
            norm_val = im.norm(val)
            text_color = "white" if norm_val > 0.5 else "black"
            if cmap == "OrRd" and norm_val > 0.6:  # For dark red background
                text_color = "white"
            ax.text(j, i, f"{val:{fmt[1:]}}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)
            
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    fig.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
    fig.tight_layout()
    
    # Save fig
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Copy to artifacts
    shutil.copy(out_path, ARTIFACTS_DIR / filename)
    print(f"Saved and copied heatmap: {filename}")

def main():
    plt.close("all")
    
    # 1. Collect all data
    # Dimensions: [Trait, Theta, K]
    score_data = np.zeros((len(TRAITS), len(THETA_LIST), len(K_LIST)))
    ppl_data = np.zeros((len(TRAITS), len(THETA_LIST), len(K_LIST)))
    
    for t_idx, trait in enumerate(TRAITS):
        for th_idx, theta in enumerate(THETA_LIST):
            for k_idx, k in enumerate(K_LIST):
                score, ppl = load_metrics(theta, k, trait)
                score_data[t_idx, th_idx, k_idx] = score
                ppl_data[t_idx, th_idx, k_idx] = ppl
                
    # 2. Summary (Average across Traits)
    avg_score = score_data.mean(axis=0)
    avg_ppl = ppl_data.mean(axis=0)
    
    # 3. Plot Summary Heatmaps
    plot_heatmap(
        avg_score, 
        "Summary Alignment Score Heatmap (Phase 1 Sweep)", 
        "Slope (k_lo)", 
        "Entropy Threshold (theta_lo)", 
        K_LIST, 
        THETA_LIST, 
        "entropy_gating_phase1_summary_score.png", 
        "YlGnBu",
        fmt=".3f"
    )
    
    plot_heatmap(
        avg_ppl, 
        "Summary Text Perplexity (PPL) Heatmap (Phase 1 Sweep)", 
        "Slope (k_lo)", 
        "Entropy Threshold (theta_lo)", 
        K_LIST, 
        THETA_LIST, 
        "entropy_gating_phase1_summary_ppl.png", 
        "OrRd",
        fmt=".3f"
    )
    
    # 4. Plot Trait-specific Heatmaps (subplots for easy visualization)
    fig_scores, axes_scores = plt.subplots(2, 3, figsize=(18, 11))
    fig_ppls, axes_ppls = plt.subplots(2, 3, figsize=(18, 11))
    
    axes_scores = axes_scores.flatten()
    axes_ppls = axes_ppls.flatten()
    
    for idx, trait in enumerate(TRAITS):
        # Score Trait
        ax_s = axes_scores[idx]
        im_s = ax_s.imshow(score_data[idx], cmap="YlGnBu", aspect='auto')
        ax_s.set_xticks(np.arange(len(K_LIST)))
        ax_s.set_yticks(np.arange(len(THETA_LIST)))
        ax_s.set_xticklabels(K_LIST)
        ax_s.set_yticklabels(THETA_LIST)
        ax_s.set_title(f"{trait.capitalize()} Alignment Score", fontweight='bold', fontsize=11)
        # annotations
        for i in range(len(THETA_LIST)):
            for j in range(len(K_LIST)):
                val = score_data[idx, i, j]
                norm_val = im_s.norm(val)
                txt_col = "white" if norm_val > 0.5 else "black"
                ax_s.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_col, fontweight='bold', fontsize=9)
                
        # PPL Trait
        ax_p = axes_ppls[idx]
        im_p = ax_p.imshow(ppl_data[idx], cmap="OrRd", aspect='auto')
        ax_p.set_xticks(np.arange(len(K_LIST)))
        ax_p.set_yticks(np.arange(len(THETA_LIST)))
        ax_p.set_xticklabels(K_LIST)
        ax_p.set_yticklabels(THETA_LIST)
        ax_p.set_title(f"{trait.capitalize()} Perplexity (PPL)", fontweight='bold', fontsize=11)
        # annotations
        for i in range(len(THETA_LIST)):
            for j in range(len(K_LIST)):
                val = ppl_data[idx, i, j]
                norm_val = im_p.norm(val)
                txt_col = "white" if norm_val > 0.6 else "black"
                ax_p.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_col, fontweight='bold', fontsize=9)
                
    # Leave the 6th subplot empty or plot the average there
    # Let's plot the average on the 6th subplot!
    ax_s_avg = axes_scores[5]
    im_s_avg = ax_s_avg.imshow(avg_score, cmap="YlGnBu", aspect='auto')
    ax_s_avg.set_xticks(np.arange(len(K_LIST)))
    ax_s_avg.set_yticks(np.arange(len(THETA_LIST)))
    ax_s_avg.set_xticklabels(K_LIST)
    ax_s_avg.set_yticklabels(THETA_LIST)
    ax_s_avg.set_title("AVERAGE (Summary Score)", fontweight='bold', fontsize=11, color='blue')
    for i in range(len(THETA_LIST)):
        for j in range(len(K_LIST)):
            val = avg_score[i, j]
            norm_val = im_s_avg.norm(val)
            txt_col = "white" if norm_val > 0.5 else "black"
            ax_s_avg.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_col, fontweight='bold', fontsize=9)
            
    ax_p_avg = axes_ppls[5]
    im_p_avg = ax_p_avg.imshow(avg_ppl, cmap="OrRd", aspect='auto')
    ax_p_avg.set_xticks(np.arange(len(K_LIST)))
    ax_p_avg.set_yticks(np.arange(len(THETA_LIST)))
    ax_p_avg.set_xticklabels(K_LIST)
    ax_p_avg.set_yticklabels(THETA_LIST)
    ax_p_avg.set_title("AVERAGE (Summary PPL)", fontweight='bold', fontsize=11, color='red')
    for i in range(len(THETA_LIST)):
        for j in range(len(K_LIST)):
            val = avg_ppl[i, j]
            norm_val = im_p_avg.norm(val)
            txt_col = "white" if norm_val > 0.6 else "black"
            ax_p_avg.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_col, fontweight='bold', fontsize=9)
            
    # Set labels for all subplots
    for ax in list(axes_scores) + list(axes_ppls):
        ax.set_xlabel("Slope (k_lo)", fontsize=9)
        ax.set_ylabel("Threshold (theta_lo)", fontsize=9)
        
    fig_scores.suptitle("Alignment Score Heatmaps by Character Trait (Phase 1 Sweep)", fontsize=16, fontweight='bold', y=0.98)
    fig_scores.tight_layout()
    fig_scores.savefig(FIGURES_DIR / "entropy_gating_phase1_traits_score.png", dpi=200, bbox_inches='tight')
    shutil.copy(FIGURES_DIR / "entropy_gating_phase1_traits_score.png", ARTIFACTS_DIR / "entropy_gating_phase1_traits_score.png")
    plt.close(fig_scores)
    
    fig_ppls.suptitle("Text Perplexity (PPL) Heatmaps by Character Trait (Phase 1 Sweep)", fontsize=16, fontweight='bold', y=0.98)
    fig_ppls.tight_layout()
    fig_ppls.savefig(FIGURES_DIR / "entropy_gating_phase1_traits_ppl.png", dpi=200, bbox_inches='tight')
    shutil.copy(FIGURES_DIR / "entropy_gating_phase1_traits_ppl.png", ARTIFACTS_DIR / "entropy_gating_phase1_traits_ppl.png")
    plt.close(fig_ppls)
    
    print("All individual trait heatmaps saved and copied successfully!")

if __name__ == "__main__":
    main()
