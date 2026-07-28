#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_anticipatory_comparison.py
# Plot and report comparison between 1-Token Delayed Gating vs. Anticipatory (Re-sampling) Gating.
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

WORKSPACE = Path("/home/s2550009/persona_vectors")
EXP_DIR = WORKSPACE / "exp_token_intensity/exp_resampling_vs_delayed"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
CONFIGS = ["Peak_Score", "Best_PPL"]
MODES = ["delayed", "anticipatory"]

def load_data():
    records = []
    for cfg in CONFIGS:
        for mode in MODES:
            scores, ppls = [], []
            trait_scores = {}
            trait_ppls = {}
            for trait in TRAITS:
                trait_dir = EXP_DIR / cfg / mode / trait
                # Look for CSV
                csv_files = list(trait_dir.glob("scores_*.csv")) + list(trait_dir.glob("scores.csv"))
                if not csv_files:
                    csv_files = list(trait_dir.glob("*.csv"))
                if csv_files:
                    try:
                        df = pd.read_csv(csv_files[0])
                        s = df["dyn_score"].mean()
                        p = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean()
                        scores.append(s)
                        ppls.append(p)
                        trait_scores[trait] = s
                        trait_ppls[trait] = p
                    except Exception as e:
                        pass
            if len(scores) == 5:
                rec = {
                    "config": cfg,
                    "mode": mode,
                    "mean_score": np.mean(scores),
                    "mean_ppl": np.mean(ppls)
                }
                for trait in TRAITS:
                    rec[f"score_{trait}"] = trait_scores[trait]
                    rec[f"ppl_{trait}"] = trait_ppls[trait]
                records.append(rec)
    return pd.DataFrame(records)

def plot_comparison(df):
    if df.empty:
        print("No comparison data found yet.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.size"] = 12

    # 1. Bar Plot for Alignment Score Comparison
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(CONFIGS))
    width = 0.35

    delayed_scores = [df[(df["config"]==c) & (df["mode"]=="delayed")]["mean_score"].values[0] if len(df[(df["config"]==c) & (df["mode"]=="delayed")])>0 else 0 for c in CONFIGS]
    anticipatory_scores = [df[(df["config"]==c) & (df["mode"]=="anticipatory")]["mean_score"].values[0] if len(df[(df["config"]==c) & (df["mode"]=="anticipatory")])>0 else 0 for c in CONFIGS]

    rects1 = ax.bar(x - width/2, delayed_scores, width, label="1-Token Delayed (Previous H_t-1)", color="#e74c3c", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, anticipatory_scores, width, label="Anticipatory Re-sampling (Current H_t)", color="#2ecc71", edgecolor="black", alpha=0.9)

    for rect in rects1 + rects2:
        h = rect.get_height()
        if h > 0:
            ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.04, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Personality Alignment Score", fontsize=12, fontweight="bold")
    ax.set_title("1-Token Delay vs. Anticipatory Re-sampling Steering Score", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Peak Score Model\n(θ_hi=6.0, k_hi=1.0)", "Best PPL Tradeoff Model\n(θ_hi=7.0, k_hi=1.0)"], fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 5.2)
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()

    out_score = EXP_DIR / "anticipatory_vs_delayed_score.png"
    plt.savefig(out_score, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_score}")

    # 2. Bar Plot for Perplexity (PPL) Comparison
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    delayed_ppls = [df[(df["config"]==c) & (df["mode"]=="delayed")]["mean_ppl"].values[0] if len(df[(df["config"]==c) & (df["mode"]=="delayed")])>0 else 0 for c in CONFIGS]
    anticipatory_ppls = [df[(df["config"]==c) & (df["mode"]=="anticipatory")]["mean_ppl"].values[0] if len(df[(df["config"]==c) & (df["mode"]=="anticipatory")])>0 else 0 for c in CONFIGS]

    rects1 = ax.bar(x - width/2, delayed_ppls, width, label="1-Token Delayed (Previous H_t-1)", color="#e74c3c", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, anticipatory_ppls, width, label="Anticipatory Re-sampling (Current H_t)", color="#2ecc71", edgecolor="black", alpha=0.9)

    for rect in rects1 + rects2:
        h = rect.get_height()
        if h > 0:
            ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Perplexity (PPL - Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("1-Token Delay vs. Anticipatory Re-sampling Perplexity (PPL)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Peak Score Model\n(θ_hi=6.0, k_hi=1.0)", "Best PPL Tradeoff Model\n(θ_hi=7.0, k_hi=1.0)"], fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 14.0)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()

    out_ppl = EXP_DIR / "anticipatory_vs_delayed_ppl.png"
    plt.savefig(out_ppl, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_ppl}")

    # Copy artifacts
    if ARTIFACTS_DIR.exists():
        shutil.copy(out_score, ARTIFACTS_DIR / out_score.name)
        shutil.copy(out_ppl, ARTIFACTS_DIR / out_ppl.name)
        print("Copied comparison plots to artifacts directory.")

    # Write summary report
    report_path = EXP_DIR / "anticipatory_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 1-Token Delayed vs. Anticipatory Re-sampling Gating Report\n\n")
        f.write("This report compares the performance of 1-token delayed entropy gating against real-time anticipatory re-sampling (Double Pass) gating.\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Model Config | Mode | Alignment Score | Perplexity (PPL) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for _, r in df.iterrows():
            f.write(f"| {r['config']} | **{r['mode']}** | **{r['mean_score']:.3f}** | **{r['mean_ppl']:.3f}** |\n")

    print(f"Saved report to: {report_path}")

def main():
    df = load_data()
    plot_comparison(df)

if __name__ == "__main__":
    main()
