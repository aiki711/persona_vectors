#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/plot_entropy_gating_phase2.py
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

WORKSPACE = Path("/home/s2550009/persona_vectors")
RESULTS_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
THETA_LO = 1.2
K_LO_LIST = [1.5, 4.0]
THETA_HI_LIST = [3.0, 4.5, 6.0]
K_HI_LIST = [1.0, 2.0]

def load_metrics(k_lo: float, theta_hi: float, k_hi: float, trait: str) -> tuple[float, float]:
    csv_name = f"scores_masked_proj_rank_theta_{THETA_LO:.1f}_{theta_hi:.1f}_k_{k_lo:.1f}_{k_hi:.1f}_entropy_Val5.0.csv"
    csv_path = RESULTS_DIR / trait / csv_name
    if not csv_path.exists():
        csv_name = f"scores_masked_proj_rank_theta_{THETA_LO}_{theta_hi}_k_{k_lo}_{k_hi}_entropy_Val5.0.csv"
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
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_yticklabels(yticks, fontsize=10)
    
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            val = data[i, j]
            norm_val = im.norm(val)
            text_color = "white" if norm_val > 0.5 else "black"
            if cmap == "OrRd" and norm_val > 0.6:
                text_color = "white"
            ax.text(j, i, f"{val:{fmt[1:]}}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)
            
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    fig.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
    fig.tight_layout()
    
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    shutil.copy(out_path, ARTIFACTS_DIR / filename)
    print(f"Saved and copied heatmap: {filename}")

def main():
    plt.close("all")
    
    # Dimensions: [k_lo, theta_hi, k_hi]
    summary_scores = np.zeros((len(K_LO_LIST), len(THETA_HI_LIST), len(K_HI_LIST)))
    summary_ppls = np.zeros((len(K_LO_LIST), len(THETA_HI_LIST), len(K_HI_LIST)))
    
    for klo_idx, k_lo in enumerate(K_LO_LIST):
        for thi_idx, theta_hi in enumerate(THETA_HI_LIST):
            for khi_idx, k_hi in enumerate(K_HI_LIST):
                scores_trait, ppls_trait = [], []
                for trait in TRAITS:
                    s, p = load_metrics(k_lo, theta_hi, k_hi, trait)
                    scores_trait.append(s)
                    ppls_trait.append(p)
                summary_scores[klo_idx, thi_idx, khi_idx] = np.mean(scores_trait)
                summary_ppls[klo_idx, thi_idx, khi_idx] = np.mean(ppls_trait)
                
    # Plot for k_lo = 1.5
    plot_heatmap(
        summary_scores[0],
        f"Phase 2 Score Heatmap (k_lo=1.5, theta_lo=1.2)",
        "Fall Slope (k_hi)",
        "Fall Threshold (theta_hi)",
        K_HI_LIST,
        THETA_HI_LIST,
        "entropy_gating_phase2_klo1.5_summary_score.png",
        "YlGnBu",
        fmt=".3f"
    )
    plot_heatmap(
        summary_ppls[0],
        f"Phase 2 PPL Heatmap (k_lo=1.5, theta_lo=1.2)",
        "Fall Slope (k_hi)",
        "Fall Threshold (theta_hi)",
        K_HI_LIST,
        THETA_HI_LIST,
        "entropy_gating_phase2_klo1.5_summary_ppl.png",
        "OrRd",
        fmt=".3f"
    )
    
    # Plot for k_lo = 4.0
    plot_heatmap(
        summary_scores[1],
        f"Phase 2 Score Heatmap (k_lo=4.0, theta_lo=1.2)",
        "Fall Slope (k_hi)",
        "Fall Threshold (theta_hi)",
        K_HI_LIST,
        THETA_HI_LIST,
        "entropy_gating_phase2_klo4.0_summary_score.png",
        "YlGnBu",
        fmt=".3f"
    )
    plot_heatmap(
        summary_ppls[1],
        f"Phase 2 PPL Heatmap (k_lo=4.0, theta_lo=1.2)",
        "Fall Slope (k_hi)",
        "Fall Threshold (theta_hi)",
        K_HI_LIST,
        THETA_HI_LIST,
        "entropy_gating_phase2_klo4.0_summary_ppl.png",
        "OrRd",
        fmt=".3f"
    )

    print("Phase 2 Heatmaps generated successfully!")

if __name__ == "__main__":
    main()
