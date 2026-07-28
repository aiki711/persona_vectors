#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_entropy_gating_phase3_heatmaps.py
# Generate fine-grained summary heatmaps and per-trait heatmaps for Phase 3 Entropy Gating Sweep
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

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

THETA_LO = 1.2
K_LO = 1.5

THETA_HI_LIST = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
K_HI_LIST = [0.5, 1.0, 1.5, 2.0]

def load_phase3_data():
    """Load Phase 3 fine-grained sweep results."""
    data = []
    for th_hi in THETA_HI_LIST:
        for k_h in K_HI_LIST:
            row = {
                "theta_lo": THETA_LO,
                "k_lo": K_LO,
                "theta_hi": th_hi,
                "k_hi": k_h,
            }
            trait_scores = {}
            trait_ppls = {}
            for trait in TRAITS:
                csv_name = f"scores_masked_proj_rank_theta_{THETA_LO:.1f}_{th_hi:.1f}_k_{K_LO:.1f}_{k_h:.1f}_entropy_plateau_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name
                if not csv_path.exists():
                    csv_name = f"scores_masked_proj_rank_theta_{THETA_LO}_{th_hi}_k_{K_LO}_{k_h}_entropy_plateau_Val5.0.csv"
                    csv_path = OUT_DIR / trait / csv_name
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        s = df["dyn_score"].mean()
                        p = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean()
                        trait_scores[trait] = s
                        trait_ppls[trait] = p
                    except Exception as e:
                        pass
            if len(trait_scores) == 5:
                row["mean_score"] = np.mean(list(trait_scores.values()))
                row["mean_ppl"] = np.mean(list(trait_ppls.values()))
                for trait in TRAITS:
                    row[f"score_{trait}"] = trait_scores[trait]
                    row[f"ppl_{trait}"] = trait_ppls[trait]
                data.append(row)
    return pd.DataFrame(data)

def plot_phase3_heatmaps(df):
    if df.empty:
        print("No Phase 3 data found.")
        return

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 12

    # 1. Summary Score Heatmap
    piv_score = df.pivot(index="k_hi", columns="theta_hi", values="mean_score").sort_index(ascending=False)
    plt.figure(figsize=(11, 6), dpi=300)
    sns.heatmap(piv_score, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Personality Alignment Score"}, linewidths=0.5, linecolor="gray")
    plt.title(r"Phase 3: Extended Fall-Stage Alignment Score Summary ($\theta_{\mathrm{lo}}=1.2, k_{\mathrm{lo}}=1.5$)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(r"Fall Threshold $\theta_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.ylabel(r"Fall Slope $k_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_score = OUT_DIR / "entropy_gating_phase3_summary_score.png"
    plt.savefig(out_score, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_score}")

    # 2. Summary PPL Heatmap
    piv_ppl = df.pivot(index="k_hi", columns="theta_hi", values="mean_ppl").sort_index(ascending=False)
    plt.figure(figsize=(11, 6), dpi=300)
    sns.heatmap(piv_ppl, annot=True, fmt=".2f", cmap="YlOrRd_r", cbar_kws={"label": "Perplexity (PPL)"}, linewidths=0.5, linecolor="gray")
    plt.title(r"Phase 3: Extended Fall-Stage Perplexity (PPL) Summary ($\theta_{\mathrm{lo}}=1.2, k_{\mathrm{lo}}=1.5$)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(r"Fall Threshold $\theta_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.ylabel(r"Fall Slope $k_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_ppl = OUT_DIR / "entropy_gating_phase3_summary_ppl.png"
    plt.savefig(out_ppl, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_ppl}")

    # 3. Per-Trait Score Grid (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    axes = axes.flatten()

    for idx, trait in enumerate(TRAITS):
        piv = df.pivot(index="k_hi", columns="theta_hi", values=f"score_{trait}").sort_index(ascending=False)
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", ax=axes[idx], cbar=True, linewidths=0.5)
        axes[idx].set_title(TRAIT_TITLES[trait], fontsize=13, fontweight="bold")
        axes[idx].set_xlabel(r"$\theta_{\mathrm{hi}}$", fontsize=11)
        axes[idx].set_ylabel(r"$k_{\mathrm{hi}}$", fontsize=11)

    axes[5].axis("off")
    fig.suptitle("Phase 3: Per-Trait Alignment Score Heatmaps", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_traits_score = OUT_DIR / "entropy_gating_phase3_traits_score.png"
    plt.savefig(out_traits_score)
    plt.close()
    print(f"Saved: {out_traits_score}")

    # 4. Per-Trait PPL Grid (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    axes = axes.flatten()

    for idx, trait in enumerate(TRAITS):
        piv = df.pivot(index="k_hi", columns="theta_hi", values=f"ppl_{trait}").sort_index(ascending=False)
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="Reds_r", ax=axes[idx], cbar=True, linewidths=0.5)
        axes[idx].set_title(TRAIT_TITLES[trait], fontsize=13, fontweight="bold")
        axes[idx].set_xlabel(r"$\theta_{\mathrm{hi}}$", fontsize=11)
        axes[idx].set_ylabel(r"$k_{\mathrm{hi}}$", fontsize=11)

    axes[5].axis("off")
    fig.suptitle("Phase 3: Per-Trait Perplexity (PPL) Heatmaps", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_traits_ppl = OUT_DIR / "entropy_gating_phase3_traits_ppl.png"
    plt.savefig(out_traits_ppl)
    plt.close()
    print(f"Saved: {out_traits_ppl}")

    # Copy to artifacts directory
    if ARTIFACTS_DIR.exists():
        for p in [out_score, out_ppl, out_traits_score, out_traits_ppl]:
            shutil.copy(p, ARTIFACTS_DIR / p.name)
        print("Copied Phase 3 heatmaps to artifacts directory.")

def main():
    print("Loading Phase 3 data...")
    df = load_phase3_data()
    print(f"Loaded {len(df)} configurations out of 24.")
    plot_phase3_heatmaps(df)

if __name__ == "__main__":
    main()
