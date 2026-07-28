#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_entropy_gating_tradeoff_scatter.py
# Plot 2D scatter tradeoff (Personality Alignment Score vs Perplexity)
# comparing No Steering, Logit-diff, No Gating, and Proposed Optimal Entropy Plateau.
#

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    plt.close("all")
    sns_style = "whitegrid"
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.size"] = 12

    # Data points
    # Format: label: (score, ppl, color, marker, size, text_offset_x, text_offset_y)
    methods = {
        "No Steering\n(Baseline)": {
            "score": 3.120, "ppl": 5.661,
            "color": "#7f8c8d", "marker": "o", "size": 140,
            "ha": "left", "va": "bottom", "dx": 0.04, "dy": 0.15
        },
        "Logit-diff\n(DLS)": {
            "score": 3.700, "ppl": 9.689,
            "color": "#e74c3c", "marker": "s", "size": 140,
            "ha": "center", "va": "top", "dx": 0.00, "dy": -0.30
        },
        "No Gating\n(Static Steering)": {
            "score": 4.340, "ppl": 10.463,
            "color": "#9b59b6", "marker": "^", "size": 150,
            "ha": "left", "va": "top", "dx": 0.03, "dy": -0.20
        },
        "Proposed Optimal\n(Max Score: θ_hi=6.0, k_hi=1.0)": {
            "score": 4.460, "ppl": 10.362,
            "color": "#2ecc71", "marker": "*", "size": 240,
            "ha": "left", "va": "bottom", "dx": 0.03, "dy": 0.15
        },
        "Proposed Optimal\n(Best PPL Tradeoff: θ_hi=7.0, k_hi=1.0)": {
            "score": 4.360, "ppl": 9.675,
            "color": "#27ae60", "marker": "D", "size": 160,
            "ha": "right", "va": "bottom", "dx": -0.03, "dy": 0.20
        }
    }

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    for name, d in methods.items():
        ax.scatter(
            d["score"], d["ppl"],
            color=d["color"], marker=d["marker"], s=d["size"],
            edgecolor="black", linewidth=1.5, alpha=0.95, zorder=5,
            label=name.replace("\n", " ")
        )
        # Annotation label
        ax.annotate(
            f"{name}\n(Score: {d['score']:.2f}, PPL: {d['ppl']:.2f})",
            (d["score"], d["ppl"]),
            xytext=(d["score"] + d["dx"], d["ppl"] + d["dy"]),
            fontsize=10, fontweight="bold", color=d["color"],
            ha=d["ha"], va=d["va"],
            arrowprops=dict(arrowstyle="->", color=d["color"], lw=1.0, alpha=0.7) if abs(d["dx"])+abs(d["dy"]) > 0.3 else None
        )

    # Highlight Pareto Direction (High Score, Low PPL)
    ax.annotate(
        "Pareto Ideal Direction\n(High Score, Low PPL)",
        xy=(4.48, 6.0), xytext=(3.80, 6.5),
        arrowprops=dict(facecolor="#2980b9", edgecolor="#2980b9", width=2, headwidth=8),
        fontsize=11, fontweight="bold", color="#2980b9",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ebf5fb", edgecolor="#2980b9", alpha=0.9)
    )

    ax.set_xlabel("Personality Alignment Score (Higher is Better)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Perplexity / PPL (Lower is Better)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title("Steering Performance Tradeoff (2D Comparison)", fontsize=15, fontweight="bold", pad=15)

    ax.set_xlim(2.9, 4.65)
    ax.set_ylim(5.0, 11.8)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Invert Y axis conceptually by visual annotation or custom layout
    # Keep standard PPL orientation but clearly indicate lower is better
    plt.tight_layout()

    out_file = OUT_DIR / "entropy_gating_tradeoff_scatter.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    print(f"Saved scatter plot to: {out_file}")

    # Copy to artifacts directory
    if ARTIFACTS_DIR.exists():
        shutil.copy(out_file, ARTIFACTS_DIR / out_file.name)
        print("Copied scatter plot to artifacts directory.")

if __name__ == "__main__":
    main()
