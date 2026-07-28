#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_v03_fall_heatmaps.py
# Plot extended fine-grained heatmaps (Score & PPL) for Fall-Stage Sweep on Mistral-7B-Instruct-v0.3
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_fall_sweep"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

THETA_HI_LIST = [1.5, 2.0, 2.5, 3.0, 3.5]
K_HI_LIST = [0.05, 0.1, 0.2, 0.3, 0.4]

def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.size"] = 11

    matrix_score = np.full((len(K_HI_LIST), len(THETA_HI_LIST)), np.nan)
    matrix_ppl = np.full((len(K_HI_LIST), len(THETA_HI_LIST)), np.nan)

    for j, th_hi in enumerate(THETA_HI_LIST):
        for i, k_h in enumerate(K_HI_LIST):
            scores, ppls = [], []
            for trait in TRAITS:
                csv_path = OUT_DIR / trait / f"scores_masked_proj_rank_theta_0.0_{th_hi:.1f}_k_1.0_{k_h:.2f}_entropy_plateau_Val5.0.csv"
                if not csv_path.exists():
                    csv_path = OUT_DIR / trait / f"scores_masked_proj_rank_theta_0.0_{th_hi:.1f}_k_1.0_{k_h:.1f}_entropy_plateau_Val5.0.csv"
                if not csv_path.exists():
                    csv_path = OUT_DIR / trait / f"scores_masked_proj_rank_theta_0.0_{th_hi}_k_1.0_{k_h}_entropy_plateau_Val5.0.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    scores.append(df["dyn_score"].mean())
                    ppls.append(df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean())
            if len(scores) == 5:
                matrix_score[i, j] = np.mean(scores)
                matrix_ppl[i, j] = np.mean(ppls)

    # 1. Summary Score Heatmap
    plt.figure(figsize=(9, 6), dpi=300)
    sns.heatmap(
        matrix_score, annot=True, fmt=".3f", cmap="YlGnBu",
        xticklabels=[f"{x:.1f}" for x in THETA_HI_LIST],
        yticklabels=[f"{y:.2f}" for y in K_HI_LIST],
        cbar_kws={"label": "Personality Alignment Score"}
    )
    plt.title("Fall-Stage Sweep: Alignment Score Summary (Mistral-7B-v0.3)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(r"Fall Threshold $\theta_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.ylabel(r"Fall Slope $k_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_score = OUT_DIR / "v03_fall_summary_score.png"
    plt.savefig(out_score, bbox_inches="tight")
    plt.close()

    # 2. Summary PPL Heatmap
    plt.figure(figsize=(9, 6), dpi=300)
    sns.heatmap(
        matrix_ppl, annot=True, fmt=".2f", cmap="YlOrRd_r",
        xticklabels=[f"{x:.1f}" for x in THETA_HI_LIST],
        yticklabels=[f"{y:.2f}" for y in K_HI_LIST],
        cbar_kws={"label": "Perplexity (PPL)"}
    )
    plt.title("Fall-Stage Sweep: Perplexity (PPL) Summary (Mistral-7B-v0.3)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(r"Fall Threshold $\theta_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.ylabel(r"Fall Slope $k_{\mathrm{hi}}$", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_ppl = OUT_DIR / "v03_fall_summary_ppl.png"
    plt.savefig(out_ppl, bbox_inches="tight")
    plt.close()

    if ARTIFACTS_DIR.exists():
        shutil.copy(out_score, ARTIFACTS_DIR / out_score.name)
        shutil.copy(out_ppl, ARTIFACTS_DIR / out_ppl.name)
        print("Updated Fall-Stage heatmaps in artifacts directory.")

if __name__ == "__main__":
    main()
