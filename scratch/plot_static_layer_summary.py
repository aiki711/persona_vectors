#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_static_layer_summary.py
#

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

def main():
    plt.close("all")

    # Data
    methods = [
        "Unsteered Baseline",
        "DLS Logit-diff",
        "DLS Proj Rank-Only\n(Static Layer)",
        "PDF Proj Rank-Only\n(Static Layer)"
    ]

    scores = [3.12, 3.70, 3.96, 4.35]
    ppls = [5.66, 9.69, 10.50, 10.40]

    colors = ["#7f8c8d", "#e74c3c", "#3498db", "#9b59b6"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Alignment Score
    bars1 = ax1.bar(methods, scores, color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax1.set_ylabel("Personality Alignment Score", fontsize=11, fontweight="bold")
    ax1.set_title("Steering Alignment Score (Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.set_ylim(0, 5.0)

    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight="bold")

    # Plot 2: Perplexity
    bars2 = ax2.bar(methods, ppls, color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax2.set_ylabel("Language Perplexity (PPL)", fontsize=11, fontweight="bold")
    ax2.set_title("Text Perplexity (Lower is Better / More Fluent)", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.set_ylim(0, 12.0)

    # Add values on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.18, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight="bold")

    plt.suptitle("Static Layer Selection: Baseline & Method Comparison (Mistral-7B)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save to exp_layer_selection
    out_dir = Path("exp_layer_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "static_layer_results_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved summary plot to: {out_path}")

    # Copy to artifacts directory
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, artifact_dir / "static_layer_results_comparison.png")
    print("Copied plot to artifacts.")

if __name__ == "__main__":
    main()
