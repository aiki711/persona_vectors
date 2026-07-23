#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_entropy_gating_heatmaps.py
# Generate summary heatmaps and per-trait heatmaps for Entropy Gating Sweeps (Phase 1 & Phase 2)
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import re

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_TITLES = {
    "extraversion": "Extraversion",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness": "Agreeableness"
}

def load_phase1_data():
    """Load Phase 1 (Rise-stage) sweep results where theta_hi=7.0 and k_hi=2.0."""
    data = []
    # Pattern: scores_masked_proj_rank_theta_{theta_lo}_7.0_k_{k_lo}_2.0_entropy_Val5.0.csv
    pattern = re.compile(r"scores_masked_proj_rank_theta_([\d\.]+)_7\.0_k_([\d\.]+)_2\.0_entropy_Val5\.0\.csv")
    
    # Collect all unique (theta_lo, k_lo) pairs in extraversion
    files = list((OUT_DIR / "extraversion").glob("scores_masked_proj_rank_theta_*_7.0_k_*_2.0_entropy_Val5.0.csv"))
    
    for f in files:
        m = pattern.search(f.name)
        if not m:
            continue
        th_lo = float(m.group(1))
        k_l = float(m.group(2))
        
        row = {"theta_lo": th_lo, "k_lo": k_l}
        trait_scores = {}
        trait_ppls = {}
        for trait in TRAITS:
            csv_path = OUT_DIR / trait / f.name
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    s = df["dyn_score"].mean()
                    p = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean()
                    trait_scores[trait] = s
                    trait_ppls[trait] = p
                except Exception:
                    pass
        if len(trait_scores) == 5:
            row["mean_score"] = np.mean(list(trait_scores.values()))
            row["mean_ppl"] = np.mean(list(trait_ppls.values()))
            for trait in TRAITS:
                row[f"score_{trait}"] = trait_scores[trait]
                row[f"ppl_{trait}"] = trait_ppls[trait]
            data.append(row)
            
    return pd.DataFrame(data)

def plot_phase1_heatmaps(df_p1):
    if df_p1.empty:
        print("No Phase 1 data found.")
        return

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 12

    # 1. Summary Score Heatmap
    piv_score = df_p1.pivot(index="k_lo", columns="theta_lo", values="mean_score")
    piv_score = piv_score.sort_index(ascending=False)

    plt.figure(figsize=(10, 6), dpi=300)
    sns.heatmap(piv_score, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Personality Alignment Score"},
                linewidths=0.5, linecolor="gray")
    plt.title("Phase 1: Overall Summary Heatmap (Alignment Score)\n[Entropy Gating Rise-Stage Sweep: theta_hi=7.0, k_hi=2.0]", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(r"Rise Threshold $\theta_{\mathrm{lo}}$", fontsize=12, fontweight="bold")
    plt.ylabel(r"Rise Slope $k_{\mathrm{lo}}$", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path1 = OUT_DIR / "entropy_gating_phase1_summary_score.png"
    plt.savefig(out_path1)
    plt.close()
    print(f"Saved: {out_path1}")

    # 2. Summary PPL Heatmap
    piv_ppl = df_p1.pivot(index="k_lo", columns="theta_lo", values="mean_ppl")
    piv_ppl = piv_ppl.sort_index(ascending=False)

    plt.figure(figsize=(10, 6), dpi=300)
    sns.heatmap(piv_ppl, annot=True, fmt=".2f", cmap="YlOrRd_r", cbar_kws={"label": "Perplexity (PPL, Lower is Better)"},
                linewidths=0.5, linecolor="gray")
    plt.title("Phase 1: Overall Summary Heatmap (Perplexity)\n[Entropy Gating Rise-Stage Sweep: theta_hi=7.0, k_hi=2.0]", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(r"Rise Threshold $\theta_{\mathrm{lo}}$", fontsize=12, fontweight="bold")
    plt.ylabel(r"Rise Slope $k_{\mathrm{lo}}$", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path2 = OUT_DIR / "entropy_gating_phase1_summary_ppl.png"
    plt.savefig(out_path2)
    plt.close()
    print(f"Saved: {out_path2}")

    # 3. Per-Trait Score Heatmaps (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    axes = axes.flatten()

    for idx, trait in enumerate(TRAITS):
        piv = df_p1.pivot(index="k_lo", columns="theta_lo", values=f"score_{trait}")
        piv = piv.sort_index(ascending=False)
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", ax=axes[idx], cbar=True, linewidths=0.5)
        axes[idx].set_title(TRAIT_TITLES[trait], fontsize=13, fontweight="bold")
        axes[idx].set_xlabel(r"$\theta_{\mathrm{lo}}$", fontsize=11)
        axes[idx].set_ylabel(r"$k_{\mathrm{lo}}$", fontsize=11)

    axes[5].axis("off")
    fig.suptitle("Phase 1: Per-Trait Alignment Score Heatmaps", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path3 = OUT_DIR / "entropy_gating_phase1_traits_score.png"
    plt.savefig(out_path3)
    plt.close()
    print(f"Saved: {out_path3}")

    # 4. Per-Trait PPL Heatmaps (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    axes = axes.flatten()

    for idx, trait in enumerate(TRAITS):
        piv = df_p1.pivot(index="k_lo", columns="theta_lo", values=f"ppl_{trait}")
        piv = piv.sort_index(ascending=False)
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="Reds_r", ax=axes[idx], cbar=True, linewidths=0.5)
        axes[idx].set_title(TRAIT_TITLES[trait], fontsize=13, fontweight="bold")
        axes[idx].set_xlabel(r"$\theta_{\mathrm{lo}}$", fontsize=11)
        axes[idx].set_ylabel(r"$k_{\mathrm{lo}}$", fontsize=11)

    axes[5].axis("off")
    fig.suptitle("Phase 1: Per-Trait Perplexity (PPL) Heatmaps", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path4 = OUT_DIR / "entropy_gating_phase1_traits_ppl.png"
    plt.savefig(out_path4)
    plt.close()
    print(f"Saved: {out_path4}")

    # Copy to artifacts directory
    if ARTIFACTS_DIR.exists():
        for p in [out_path1, out_path2, out_path3, out_path4]:
            shutil.copy(p, ARTIFACTS_DIR / p.name)
        print("Copied all Phase 1 heatmaps to artifacts directory.")

def main():
    print("Loading Phase 1 data...")
    df_p1 = load_phase1_data()
    print(f"Phase 1 configurations loaded: {len(df_p1)}")
    plot_phase1_heatmaps(df_p1)

if __name__ == "__main__":
    main()
