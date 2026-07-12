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

# Gating configurations and parameters
CONFIGS = [
    ("P-Conf 2 (Base Plat)", 3.0, 7.0, 2.0, 2.0, "plateau", 
     3.620, 9.585, 3.720, 9.351), # DLS Score/PPL, PDF Score/PPL
    ("P-Conf 3 (Wider Plat)", 1.0, 9.0, 2.0, 2.0, "plateau", 
     3.960, 10.478, 4.200, 10.794),
    ("P-Conf 4 (Narrow Plat)", 4.0, 6.0, 2.0, 2.0, "plateau", 
     3.200, 8.727, 3.600, 9.559),
    
    ("P-Conf 5 (Sharp Plat)", 3.0, 7.0, 8.0, 8.0, "plateau", 
     3.620, 9.091, 3.580, 9.429),
    ("P-Conf 6 (Gentle Plat)", 3.0, 7.0, 0.5, 0.5, "plateau", 
     3.960, 9.443, 4.120, 9.964),
    ("A-Conf 1 (Gent/Sharp)", 3.0, 7.0, 0.5, 8.0, "max_normalized", 
     3.800, 9.408, 3.520, 8.989),
    
    ("A-Conf 2 (Sharp/Gent)", 3.0, 7.0, 8.0, 0.5, "max_normalized", 
     3.300, 9.654, 3.520, 9.191),
    ("A-Conf 3 (Low IC Focus)", 1.0, 5.0, 1.0, 4.0, "max_normalized", 
     3.740, 9.470, 4.120, 9.416),
    ("A-Conf 4 (High IC Focus)", 5.0, 9.0, 4.0, 1.0, "max_normalized", 
     3.100, 8.699, 3.220, 9.194),
]

def main():
    plt.close("all")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    ic_values = np.linspace(0, 15, 500)
    
    for idx, (name, theta_lo, theta_hi, k_lo, k_hi, mode, dls_s, dls_p, pdf_s, pdf_p) in enumerate(CONFIGS):
        ax = axes[idx]
        gain = gating_function(ic_values, k_lo, k_hi, theta_lo, theta_hi, mode)
        
        # Color-code Plateau vs Asymmetric
        color = "#3498db" if "P-Conf" in name else "#e74c3c"
        
        ax.plot(ic_values, gain, color=color, linewidth=2.5, label=name)
        ax.fill_between(ic_values, 0, gain, color=color, alpha=0.08)
        
        # Grid/Styling for subplots
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        # Display params & scores inside a text box
        box_text = (
            f"θ: {theta_lo}-{theta_hi}, k: {k_lo}-{k_hi}\n\n"
            f"DLS Proj Rank:\n"
            f"  Score: {dls_s:.2f}  |  PPL: {dls_p:.2f}\n"
            f"PDF Proj Rank (Soft):\n"
            f"  Score: {pdf_s:.2f}  |  PPL: {pdf_p:.2f}"
        )
        
        # Use simple markdown-like box without math syntax to prevent matplotlib parsing errors
        ax.text(0.5, 0.45, box_text, transform=ax.transAxes, fontsize=9.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#dddddd", alpha=0.9))
        
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        
    # Set labels for outer axes
    for i in range(3):
        axes[6 + i].set_xlabel("Information Content (IC) [bits]", fontsize=11, labelpad=8)
    for i in range(3):
        axes[3 * i].set_ylabel("Gating Gain (Normalized)", fontsize=11, labelpad=8)
        
    plt.suptitle("Gating Function Curve Shapes vs Resulting Scores & PPL (Mistral-7B)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save image
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gating_curves_and_scores_matrix.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved matrix plot to: {out_path}")
    
    # Copy to artifact path
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "gating_curves_and_scores_matrix.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()
