#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_entropy_gating_phase2_heatmaps.py
# Generate heatmaps and summary report for Phase 2 Fall-Stage Entropy Gating Sweep
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
K_LO_LIST = [1.5, 4.0]
THETA_HI_LIST = [3.0, 4.5, 6.0]
K_HI_LIST = [1.0, 2.0]

def load_phase2_data():
    """Load Phase 2 fall-stage sweep results."""
    data = []
    for k_l in K_LO_LIST:
        for th_hi in THETA_HI_LIST:
            for k_h in K_HI_LIST:
                row = {
                    "theta_lo": THETA_LO,
                    "k_lo": k_l,
                    "theta_hi": th_hi,
                    "k_hi": k_h,
                    "label": f"k_lo={k_l}, thi={th_hi}, khi={k_h}"
                }
                trait_scores = {}
                trait_ppls = {}
                for trait in TRAITS:
                    csv_name = f"scores_masked_proj_rank_theta_{THETA_LO:.1f}_{th_hi:.1f}_k_{k_l:.1f}_{k_h:.1f}_entropy_plateau_Val5.0.csv"
                    csv_path = OUT_DIR / trait / csv_name
                    if not csv_path.exists():
                        csv_name = f"scores_masked_proj_rank_theta_{THETA_LO}_{th_hi}_k_{k_l}_{k_h}_entropy_plateau_Val5.0.csv"
                        csv_path = OUT_DIR / trait / csv_name
                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path)
                            s = df["dyn_score"].mean()
                            p = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean()
                            trait_scores[trait] = s
                            trait_ppls[trait] = p
                        except Exception as e:
                            print(f"Error reading {csv_path}: {e}")
                if len(trait_scores) == 5:
                    row["mean_score"] = np.mean(list(trait_scores.values()))
                    row["mean_ppl"] = np.mean(list(trait_ppls.values()))
                    for trait in TRAITS:
                        row[f"score_{trait}"] = trait_scores[trait]
                        row[f"ppl_{trait}"] = trait_ppls[trait]
                    data.append(row)
    return pd.DataFrame(data)

def plot_phase2_heatmaps(df):
    if df.empty:
        print("No Phase 2 data found.")
        return

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 12

    # 1. Summary Score & PPL Heatmaps for k_lo=1.5 and k_lo=4.0
    for k_l in K_LO_LIST:
        df_sub = df[df["k_lo"] == k_l]
        if df_sub.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        piv_score = df_sub.pivot(index="k_hi", columns="theta_hi", values="mean_score").sort_index(ascending=False)
        sns.heatmap(piv_score, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax1, cbar_kws={"label": "Alignment Score"}, linewidths=0.5)
        ax1.set_title(f"Alignment Score (k_lo = {k_l})", fontsize=13, fontweight="bold")
        ax1.set_xlabel(r"Fall Threshold $\theta_{\mathrm{hi}}$", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"Fall Slope $k_{\mathrm{hi}}$", fontsize=11, fontweight="bold")

        piv_ppl = df_sub.pivot(index="k_hi", columns="theta_hi", values="mean_ppl").sort_index(ascending=False)
        sns.heatmap(piv_ppl, annot=True, fmt=".2f", cmap="YlOrRd_r", ax=ax2, cbar_kws={"label": "Perplexity (PPL)"}, linewidths=0.5)
        ax2.set_title(f"Perplexity (k_lo = {k_l})", fontsize=13, fontweight="bold")
        ax2.set_xlabel(r"Fall Threshold $\theta_{\mathrm{hi}}$", fontsize=11, fontweight="bold")
        ax2.set_ylabel(r"Fall Slope $k_{\mathrm{hi}}$", fontsize=11, fontweight="bold")

        fig.suptitle(f"Phase 2 Fall-Stage Entropy Gating Sweep Summary (k_lo = {k_l})", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = OUT_DIR / f"entropy_gating_phase2_summary_klo_{k_l}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        if ARTIFACTS_DIR.exists():
            shutil.copy(out_path, ARTIFACTS_DIR / out_path.name)

    # 2. Per-Trait Alignment Score Grid (for optimal k_lo = 1.5)
    df_k15 = df[df["k_lo"] == 1.5]
    if not df_k15.empty:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
        axes = axes.flatten()

        for idx, trait in enumerate(TRAITS):
            piv = df_k15.pivot(index="k_hi", columns="theta_hi", values=f"score_{trait}").sort_index(ascending=False)
            sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", ax=axes[idx], cbar=True, linewidths=0.5)
            axes[idx].set_title(TRAIT_TITLES[trait], fontsize=13, fontweight="bold")
            axes[idx].set_xlabel(r"$\theta_{\mathrm{hi}}$", fontsize=11)
            axes[idx].set_ylabel(r"$k_{\mathrm{hi}}$", fontsize=11)

        axes[5].axis("off")
        fig.suptitle(r"Phase 2: Per-Trait Alignment Score Heatmaps ($k_{\mathrm{lo}}=1.5$)", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_traits_score = OUT_DIR / "entropy_gating_phase2_traits_score.png"
        plt.savefig(out_traits_score)
        plt.close()
        print(f"Saved: {out_traits_score}")
        if ARTIFACTS_DIR.exists():
            shutil.copy(out_traits_score, ARTIFACTS_DIR / out_traits_score.name)

        # 3. Per-Trait PPL Grid
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
        axes = axes.flatten()

        for idx, trait in enumerate(TRAITS):
            piv = df_k15.pivot(index="k_hi", columns="theta_hi", values=f"ppl_{trait}").sort_index(ascending=False)
            sns.heatmap(piv, annot=True, fmt=".2f", cmap="Reds_r", ax=axes[idx], cbar=True, linewidths=0.5)
            axes[idx].set_title(TRAIT_TITLES[trait], fontsize=13, fontweight="bold")
            axes[idx].set_xlabel(r"$\theta_{\mathrm{hi}}$", fontsize=11)
            axes[idx].set_ylabel(r"$k_{\mathrm{hi}}$", fontsize=11)

        axes[5].axis("off")
        fig.suptitle(r"Phase 2: Per-Trait Perplexity Heatmaps ($k_{\mathrm{lo}}=1.5$)", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_traits_ppl = OUT_DIR / "entropy_gating_phase2_traits_ppl.png"
        plt.savefig(out_traits_ppl)
        plt.close()
        print(f"Saved: {out_traits_ppl}")
        if ARTIFACTS_DIR.exists():
            shutil.copy(out_traits_ppl, ARTIFACTS_DIR / out_traits_ppl.name)

def generate_markdown_report(df):
    df_sorted = df.sort_values(by="mean_score", ascending=False)
    report_path = OUT_DIR / "entropy_gating_phase2_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 (Fall-Stage) Entropy Gating Grid Sweep Report\n\n")
        f.write("This report presents the complete grid sweep optimization results for predictive entropy fall-stage parameters ($\\theta_{\\text{hi}}$ and $k_{\\text{hi}}$).\n\n")
        f.write("## Overall Ranking (Ordered by Mean Alignment Score)\n\n")
        f.write("| Rank | Rise Slope ($k_{\\text{lo}}$) | Fall Threshold ($\\theta_{\\text{hi}}$) | Fall Slope ($k_{\\text{hi}}$) | Mean Alignment Score | Mean Perplexity (PPL) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write(f"| {rank} | {row['k_lo']} | {row['theta_hi']} | {row['k_hi']} | **{row['mean_score']:.3f}** | **{row['mean_ppl']:.3f}** |\n")

        f.write("\n## Key Insights\n\n")
        best = df_sorted.iloc[0]
        f.write(f"- **Top Optimal Configuration**: $k_{{\\text{{lo}}}} = {best['k_lo']}$, $\\theta_{{\\text{{hi}}}} = {best['theta_hi']}$, $k_{{\\text{{hi}}}} = {best['k_hi']}$ yielding an alignment score of **{best['mean_score']:.3f}** and PPL of **{best['mean_ppl']:.3f}**.\n")
        f.write("- **Plateau Stability**: Controlling the fall-stage prevents over-steering on high-entropy tail tokens, maintaining low perplexity while sustaining robust persona steering strength.\n")

    print(f"Saved report to: {report_path}")
    if ARTIFACTS_DIR.exists():
        shutil.copy(report_path, ARTIFACTS_DIR / "entropy_gating_phase2_report.md")

def main():
    print("Loading Phase 2 data...")
    df = load_phase2_data()
    print(f"Loaded {len(df)} configurations out of 12.")
    plot_phase2_heatmaps(df)
    generate_markdown_report(df)
    print("Phase 2 plotting & report generation complete.")

if __name__ == "__main__":
    main()
