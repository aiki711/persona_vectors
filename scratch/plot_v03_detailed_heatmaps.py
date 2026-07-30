#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

WORKSPACE = Path("/home/s2550009/persona_vectors")
RISE_DIR = WORKSPACE / "exp_token_intensity/exp_v03_rise_sweep"
FALL_DIR = WORKSPACE / "exp_token_intensity/exp_v03_fall_sweep"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/3f7b9818-2c63-474f-b2e3-53654250dd23")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

THETA_LO_LIST = [0.1, 0.2, 0.3, 0.4, 0.6]
K_LO_LIST = [2.0, 3.0, 4.0, 5.0, 6.0]

THETA_HI_LIST = [1.5, 2.0, 2.5, 3.0, 3.5]
K_HI_LIST = [0.05, 0.1, 0.2, 0.3, 0.4]

def analyze_sweep(sweep_dir, theta_list, k_list, is_rise=True):
    prefix = "rise" if is_rise else "fall"
    trait_matrices_score = {t: np.full((len(k_list), len(theta_list)), np.nan) for t in TRAITS}
    trait_matrices_ppl = {t: np.full((len(k_list), len(theta_list)), np.nan) for t in TRAITS}
    
    rows = []
    
    for j, th in enumerate(theta_list):
        for i, k in enumerate(k_list):
            for trait in TRAITS:
                if is_rise:
                    csv_path = sweep_dir / trait / f"scores_masked_proj_rank_theta_{th:.1f}_99.0_k_{k:.1f}_1.0_entropy_plateau_Val5.0.csv"
                    if not csv_path.exists():
                        csv_path = sweep_dir / trait / f"scores_masked_proj_rank_theta_{th}_99.0_k_{k}_1.0_entropy_plateau_Val5.0.csv"
                else:
                    csv_path = sweep_dir / trait / f"scores_masked_proj_rank_theta_0.0_{th:.1f}_k_1.0_{k:.2f}_entropy_plateau_Val5.0.csv"
                    if not csv_path.exists():
                        csv_path = sweep_dir / trait / f"scores_masked_proj_rank_theta_0.0_{th:.1f}_k_1.0_{k:.1f}_entropy_plateau_Val5.0.csv"
                    if not csv_path.exists():
                        csv_path = sweep_dir / trait / f"scores_masked_proj_rank_theta_0.0_{th}_k_1.0_{k}_entropy_plateau_Val5.0.csv"
                
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    sc = df["dyn_score"].mean()
                    ppl = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean()
                    
                    trait_matrices_score[trait][i, j] = sc
                    trait_matrices_ppl[trait][i, j] = ppl
                    
                    rows.append({
                        "sweep": prefix,
                        "theta": th,
                        "k": k,
                        "trait": trait,
                        "score": sc,
                        "ppl": ppl
                    })
    
    df_all = pd.DataFrame(rows)
    
    summary_score = np.mean([trait_matrices_score[t] for t in TRAITS], axis=0)
    summary_ppl = np.mean([trait_matrices_ppl[t] for t in TRAITS], axis=0)
    
    plot_items = TRAITS + ["summary"]
    
    # Score Grid Plot (2x3)
    fig_sc, axes_sc = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes_sc = axes_sc.flatten()
    for idx, item in enumerate(plot_items):
        ax = axes_sc[idx]
        mat = summary_score if item == "summary" else trait_matrices_score[item]
        title_str = "Summary (All 5 Traits)" if item == "summary" else item.capitalize()
        sns.heatmap(
            mat, annot=True, fmt=".2f" if item != "summary" else ".3f", cmap="YlGnBu",
            xticklabels=[f"{x}" for x in theta_list],
            yticklabels=[f"{y}" for y in k_list],
            cbar=True, ax=ax
        )
        ax.set_title(f"{title_str} Score", fontweight="bold")
        ax.set_xlabel(r"$\theta$" + f" ({prefix})")
        ax.set_ylabel(r"$k$" + f" ({prefix})")
    plt.tight_layout()
    fig_sc.savefig(ARTIFACTS_DIR / f"v03_{prefix}_traits_score_grid.png", bbox_inches="tight")
    plt.close(fig_sc)
    
    # PPL Grid Plot (2x3)
    fig_ppl, axes_ppl = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes_ppl = axes_ppl.flatten()
    for idx, item in enumerate(plot_items):
        ax = axes_ppl[idx]
        mat = summary_ppl if item == "summary" else trait_matrices_ppl[item]
        title_str = "Summary (All 5 Traits)" if item == "summary" else item.capitalize()
        sns.heatmap(
            mat, annot=True, fmt=".2f", cmap="YlOrRd_r",
            xticklabels=[f"{x}" for x in theta_list],
            yticklabels=[f"{y}" for y in k_list],
            cbar=True, ax=ax
        )
        ax.set_title(f"{title_str} Perplexity (PPL)", fontweight="bold")
        ax.set_xlabel(r"$\theta$" + f" ({prefix})")
        ax.set_ylabel(r"$k$" + f" ({prefix})")
    plt.tight_layout()
    fig_ppl.savefig(ARTIFACTS_DIR / f"v03_{prefix}_traits_ppl_grid.png", bbox_inches="tight")
    plt.close(fig_ppl)
    
    return df_all

def main():
    df_rise = analyze_sweep(RISE_DIR, THETA_LO_LIST, K_LO_LIST, is_rise=True)
    df_fall = analyze_sweep(FALL_DIR, THETA_HI_LIST, K_HI_LIST, is_rise=False)
    
    df_rise.to_csv(ARTIFACTS_DIR / "v03_rise_sweep_detailed.csv", index=False)
    df_fall.to_csv(ARTIFACTS_DIR / "v03_fall_sweep_detailed.csv", index=False)
    
    print("Detailed analysis and heatmaps generated successfully.")

if __name__ == "__main__":
    main()
