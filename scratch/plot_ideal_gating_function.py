#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_ideal_gating_function.py
#

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

def plateau_gating(x, theta_lo=2.0, theta_hi=7.0, k_lo=1.0, k_hi=4.0, alpha_max=5.0):
    y = np.zeros_like(x)
    for i, val in enumerate(x):
        if val < theta_lo:
            y[i] = alpha_max * (2.0 / (1.0 + np.exp(-k_lo * (val - theta_lo))))
        elif val <= theta_hi:
            y[i] = alpha_max * 1.0
        else:
            y[i] = alpha_max * (2.0 / (1.0 + np.exp(k_hi * (val - theta_hi))))
    return y

def main():
    plt.close("all")

    theta_lo = 2.0
    theta_hi = 7.0
    k_lo = 1.0
    k_hi = 4.0
    alpha_max = 5.0

    x = np.linspace(0, 12, 500)
    y = plateau_gating(x, theta_lo, theta_hi, k_lo, k_hi, alpha_max)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the curve
    ax.plot(x, y, color="#8e44ad", linewidth=3.5, label=r"Dynamic Intensity $\alpha_t$")
    ax.fill_between(x, 0, y, color="#8e44ad", alpha=0.1)

    # Add vertical dashed lines at thresholds
    ax.axvline(x=theta_lo, color="#3498db", linestyle="--", linewidth=2, label=r"$\theta_{lo} = 2.0$ (Syntax threshold)")
    ax.axvline(x=theta_hi, color="#e74c3c", linestyle="--", linewidth=2, label=r"$\theta_{hi} = 7.0$ (Fact / Distortion threshold)")

    # Highlight and label zones
    # Zone 1: Syntax Protection
    ax.axvspan(0, theta_lo, color="#3498db", alpha=0.05)
    ax.text(theta_lo / 2.0, alpha_max * 0.4, "Syntax Protection\n(Steering OFF)", 
            color="#2980b9", fontsize=10, fontweight="bold", ha="center")

    # Zone 2: Semantic Steering
    ax.axvspan(theta_lo, theta_hi, color="#2ecc71", alpha=0.05)
    ax.text((theta_lo + theta_hi) / 2.0, alpha_max * 0.5, "Semantic Steering\n(Steering ON at 100%)", 
            color="#27ae60", fontsize=11, fontweight="bold", ha="center")

    # Zone 3: Fact Protection
    ax.axvspan(theta_hi, 12, color="#e67e22", alpha=0.05)
    ax.text((theta_hi + 12) / 2.0, alpha_max * 0.4, "Fact & Rare Word\nProtection (Steering Cliff)", 
            color="#d35400", fontsize=10, fontweight="bold", ha="center")

    # Labels and Titles
    ax.set_xlabel("Token Information Content (IC) [bits]", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel(r"Steering Intensity $\alpha_t$", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Ideal Gating Function: Plateau-Asymmetric Sigmoid Curve", fontsize=14, fontweight="bold", pad=15)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, alpha_max * 1.15)
    ax.set_yticks(np.arange(0, alpha_max + 1, 1.0))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    plt.tight_layout()

    # Save to exp_layer_selection
    out_dir = Path("exp_layer_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ideal_gating_function.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved ideal gating curve to: {out_path}")

    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, artifact_dir / "ideal_gating_function.png")
    print("Copied plot to artifacts.")

if __name__ == "__main__":
    main()
