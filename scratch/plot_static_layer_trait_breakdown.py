#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_static_layer_trait_breakdown.py
#

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

def main():
    plt.close("all")

    # Categories (x-axis)
    categories = [
        "Extraversion",
        "Neuroticism",
        "Openness",
        "Conscientiousness",
        "Agreeableness",
        "Average"
    ]

    # Data structures (method -> list of 6 values)
    scores = {
        "Unsteered Baseline": [3.50, 2.10, 2.70, 3.10, 4.20, 3.12],
        "DLS Logit-diff": [3.10, 3.40, 4.60, 4.10, 3.30, 3.70],
        "PDF Proj Rank-Only": [5.00, 4.50, 3.70, 4.10, 4.40, 4.34]
    }

    ppls = {
        "Unsteered Baseline": [5.661, 5.661, 5.661, 5.661, 5.661, 5.661],
        "DLS Logit-diff": [10.024, 8.064, 9.147, 9.471, 11.740, 9.689],
        "PDF Proj Rank-Only": [11.469, 12.809, 11.383, 9.022, 7.631, 10.463]
    }

    methods = list(scores.keys())
    colors = ["#7f8c8d", "#e74c3c", "#3498db"]
    x = np.arange(len(categories))
    width = 0.18

    # Output directories
    out_dir = Path("exp_layer_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot Score separately
    plt.figure(figsize=(10, 5))
    for idx, method in enumerate(methods):
        offset = (idx - len(methods)/2 + 0.5) * width
        rects = plt.bar(x + offset, scores[method], width, label=method, color=colors[idx], edgecolor="black", alpha=0.9)
        
        if method in ["Unsteered Baseline", "PDF Proj Rank-Only"]:
            for rect in rects:
                h = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, h + 0.05, f"{h:.2f}", ha='center', va='bottom', fontsize=8, fontweight="bold")

    plt.ylabel("Steering Alignment Score", fontsize=11, fontweight="bold")
    plt.title("Steering Alignment Score", fontsize=12, fontweight="bold", pad=10)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.ylim(0, 5.6)
    plt.xticks(x, categories, fontsize=10, fontweight="bold")
    plt.legend(loc="lower right", ncol=2, framealpha=0.95)
    plt.tight_layout()
    
    score_path = out_dir / "static_layer_trait_score_comparison.png"
    plt.savefig(score_path, dpi=200, bbox_inches="tight")
    shutil.copy(score_path, artifact_dir / "static_layer_trait_score_comparison.png")
    print(f"Saved score plot to: {score_path}")

    # 2. Plot PPL separately
    plt.figure(figsize=(10, 5))
    for idx, method in enumerate(methods):
        offset = (idx - len(methods)/2 + 0.5) * width
        rects = plt.bar(x + offset, ppls[method], width, label=method, color=colors[idx], edgecolor="black", alpha=0.9)
        
        if method in ["Unsteered Baseline", "PDF Proj Rank-Only"]:
            for rect in rects:
                h = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, h + 0.15, f"{h:.1f}", ha='center', va='bottom', fontsize=8, fontweight="bold")

    plt.ylabel("Text Perplexity (PPL)", fontsize=11, fontweight="bold")
    plt.title("Text Perplexity (Language Quality)", fontsize=12, fontweight="bold", pad=10)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.ylim(0, 15.0)
    plt.xticks(x, categories, fontsize=10, fontweight="bold")
    plt.legend(loc="lower right", ncol=2, framealpha=0.95)
    plt.tight_layout()

    ppl_path = out_dir / "static_layer_trait_ppl_comparison.png"
    plt.savefig(ppl_path, dpi=200, bbox_inches="tight")
    shutil.copy(ppl_path, artifact_dir / "static_layer_trait_ppl_comparison.png")
    print(f"Saved ppl plot to: {ppl_path}")

if __name__ == "__main__":
    main()
