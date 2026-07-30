#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_v03_combined_heatmaps.py
# Plot and summarize Combined Rise & Fall Dynamic Gating results for Mistral-7B-v0.3
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_v03_combined_sweep"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/3f7b9818-2c63-474f-b2e3-53654250dd23")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

COMBINED_CONFIGS = [
    {"name": "High Score x High Score", "th_lo": 0.2, "k_lo": 4.0, "th_hi": 2.0, "k_hi": 0.20},
    {"name": "High Score x Low PPL",   "th_lo": 0.2, "k_lo": 4.0, "th_hi": 3.0, "k_hi": 0.30},
    {"name": "Low PPL x High Score",   "th_lo": 0.3, "k_lo": 3.0, "th_hi": 2.0, "k_hi": 0.20},
    {"name": "Low PPL x Low PPL",      "th_lo": 0.3, "k_lo": 3.0, "th_hi": 3.0, "k_hi": 0.30},
]

def main():
    rows = []
    for cfg in COMBINED_CONFIGS:
        th_lo = cfg["th_lo"]
        k_lo = cfg["k_lo"]
        th_hi = cfg["th_hi"]
        k_hi = cfg["k_hi"]
        c_name = cfg["name"]

        for trait in TRAITS:
            csv_path = OUT_DIR / trait / f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_lo:.1f}_{k_hi:.2f}_entropy_plateau_Val5.0.csv"
            if not csv_path.exists():
                csv_path = OUT_DIR / trait / f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_lo}_{k_hi}_entropy_plateau_Val5.0.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                sc = df["dyn_score"].mean()
                ppl = df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean()
                rows.append({
                    "config": c_name,
                    "th_lo": th_lo,
                    "k_lo": k_lo,
                    "th_hi": th_hi,
                    "k_hi": k_hi,
                    "trait": trait,
                    "score": sc,
                    "ppl": ppl
                })

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print("No combined sweep results found yet.")
        return

    df_all.to_csv(ARTIFACTS_DIR / "v03_combined_sweep_detailed.csv", index=False)

    # Group by config
    summary = df_all.groupby("config")[["score", "ppl"]].mean().reset_index()
    print("\n=== Combined Rise-Fall Dynamic Gating Summary (5-Trait Mean) ===")
    print(summary.to_string(index=False))

    # Bar chart for Combined Configs
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    x = np.arange(len(summary))
    width = 0.35

    ax2 = ax1.twinx()
    rects1 = ax1.bar(x - width/2, summary["score"], width, label="Personality Score", color="#2b5c8f")
    rects2 = ax2.bar(x + width/2, summary["ppl"], width, label="Perplexity (PPL)", color="#e07a5f")

    ax1.set_ylabel("Personality Score (1-5)", color="#2b5c8f", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Perplexity (PPL)", color="#e07a5f", fontweight="bold", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["config"], fontsize=11, fontweight="bold")
    ax1.set_ylim(3.5, 4.5)
    ax2.set_ylim(8.0, 14.0)

    plt.title("Combined Rise & Fall Dynamic Gating Evaluation (Mistral-7B-v0.3)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "v03_combined_summary_bar.png", bbox_inches="tight")
    plt.close(fig)

    print("Combined plot generated and saved in artifacts directory.")

if __name__ == "__main__":
    main()
