#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def sigmoid(x, k, theta):
    z = -k * (x - theta)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(z))

def gating_function(ic, k_lo, k_hi, theta_lo, theta_hi, gating_mode):
    g = sigmoid(ic, k_lo, theta_lo) * sigmoid(ic, -k_hi, theta_hi)
    # Find theoretical max to normalize
    grid = np.linspace(min(theta_lo, theta_hi) - 2.0, max(theta_lo, theta_hi) + 2.0, 1000)
    g_grid = sigmoid(grid, k_lo, theta_lo) * sigmoid(grid, -k_hi, theta_hi)
    g_max = np.max(g_grid)
    return g / g_max

# Custom configs and evaluation results
CONFIGS = [
    # (name, theta_lo, theta_hi, k_lo, k_hi, mode, dls_s, dls_p, cos_s, cos_p, rank_s, rank_p)
    ("A-Conf 5 (User's Proposal)", 2.0, 7.0, 1.0, 4.0, "max_normalized",
     3.66, 9.52, 3.78, 9.29, 3.80, 9.16),
    ("A-Conf 6 (Ultra-Low Focus)", 1.0, 4.0, 0.5, 6.0, "max_normalized",
     3.88, 9.11, 3.92, 9.24, 4.18, 9.88),
    ("A-Conf 7 (Wide Cliff-Edge)", 2.0, 8.0, 0.8, 5.0, "max_normalized",
     3.72, 9.43, 4.08, 9.52, 3.98, 9.90),
    ("A-Conf 8 (Plateau Asym Blend)", 2.0, 6.0, 0.5, 8.0, "plateau",
     4.14, 10.45, 4.12, 10.07, 4.06, 9.91),
]

def main():
    plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    ic_values = np.linspace(0, 15, 500)
    
    for idx, (name, theta_lo, theta_hi, k_lo, k_hi, mode, dls_s, dls_p, cos_s, cos_p, rank_s, rank_p) in enumerate(CONFIGS):
        ax = axes[idx]
        gain = gating_function(ic_values, k_lo, k_hi, theta_lo, theta_hi, mode)
        
        # Draw curve in warm purple
        color = "#8e44ad"
        ax.plot(ic_values, gain, color=color, linewidth=2.5, label=name)
        ax.fill_between(ic_values, 0, gain, color=color, alpha=0.08)
        
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        # Display params & scores inside a text box
        box_text = (
            f"θ: {theta_lo}-{theta_hi}, k: {k_lo}-{k_hi}\n\n"
            f"DLS Proj Rank:\n"
            f"  Score: {dls_s:.2f}  |  PPL: {dls_p:.2f}\n"
            f"PDF Proj Cos:\n"
            f"  Score: {cos_s:.2f}  |  PPL: {cos_p:.2f}\n"
            f"PDF Proj Rank (Soft):\n"
            f"  Score: {rank_s:.2f}  |  PPL: {rank_p:.2f}"
        )
        
        ax.text(0.48, 0.45, box_text, transform=ax.transAxes, fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#dddddd", alpha=0.9))
        
        ax.set_title(name, fontsize=12, fontweight="bold", pad=8)
        
    # Set labels for outer axes
    for i in range(2):
        axes[2 + i].set_xlabel("Information Content (IC) [bits]", fontsize=11, labelpad=8)
    for i in range(2):
        axes[2 * i].set_ylabel("Gating Gain (Normalized)", fontsize=11, labelpad=8)
        
    plt.suptitle("Custom Gating Curve Shapes vs Resulting Scores & PPL (Mistral-7B)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save image
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "custom_gating_curves_and_scores_matrix.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved custom matrix plot to: {out_path}")
    
    # Copy to artifact path
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "custom_gating_curves_and_scores_matrix.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()
