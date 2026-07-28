#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_optimal_alpha_function.py
# Plot the optimal dynamic gating intensity function alpha(H) vs Entropy H.
#

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def plateau_gating(h, alpha_max, theta_lo, k_lo, theta_hi, k_hi):
    f_lo = sigmoid(k_lo * (h - theta_lo))
    f_hi = sigmoid(-k_hi * (h - theta_hi))
    return alpha_max * f_lo * f_hi

def single_sigmoid_gating(h, alpha_max, theta_lo, k_lo):
    return alpha_max * sigmoid(k_lo * (h - theta_lo))

def main():
    plt.close("all")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.size"] = 12

    H = np.linspace(0.0, 10.0, 500)
    alpha_max = 5.0

    # 1. Optimal Configuration (Max Score: theta_hi = 6.0, k_hi = 1.0)
    alpha_opt_max = plateau_gating(H, alpha_max, theta_lo=1.2, k_lo=1.5, theta_hi=6.0, k_hi=1.0)

    # 2. Optimal Configuration (Best PPL Tradeoff: theta_hi = 7.0, k_hi = 1.0)
    alpha_opt_tradeoff = plateau_gating(H, alpha_max, theta_lo=1.2, k_lo=1.5, theta_hi=7.0, k_hi=1.0)

    # 3. Baseline Single Sigmoid Gating (Without Fall-Stage Cutoff)
    alpha_single = single_sigmoid_gating(H, alpha_max, theta_lo=1.2, k_lo=1.5)

    plt.figure(figsize=(10, 6), dpi=300)

    # Plot curves
    plt.plot(H, alpha_opt_max, color="#2ecc71", linewidth=3.0, label=r"Optimal Bumper Gate (Max Score: $\theta_{\mathrm{hi}}=6.0, k_{\mathrm{hi}}=1.0$)")
    plt.plot(H, alpha_opt_tradeoff, color="#27ae60", linewidth=2.5, linestyle="--", label=r"Optimal Bumper Gate (Best PPL: $\theta_{\mathrm{hi}}=7.0, k_{\mathrm{hi}}=1.0$)")
    plt.plot(H, alpha_single, color="#e74c3c", linewidth=2.0, linestyle=":", label=r"Single Sigmoid Gate (No Fall Cutoff: $f_{\mathrm{lo}}$ only)")

    # Threshold Markers
    plt.axvline(x=1.2, color="#34495e", linestyle="--", linewidth=1.2, alpha=0.7)
    plt.axvline(x=6.0, color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.7)
    plt.axvline(x=7.0, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.7)

    # Annotate Regions
    plt.axvspan(0.0, 1.2, color="#bdc3c7", alpha=0.2, label="Low-Entropy Region (No Steering)")
    plt.axvspan(1.2, 6.0, color="#abebc6", alpha=0.25, label="Stable Steering Plateau Region")
    plt.axvspan(6.0, 10.0, color="#f9ebea", alpha=0.2, label="High-Entropy Cutoff Region")

    # Text annotations
    plt.text(0.6, 2.5, "Cutoff\n(Deterministic\nTokens)", fontsize=10, fontweight="bold", ha="center", color="#7f8c8d")
    plt.text(3.6, 4.5, "Active Steering Plateau\n" + r"$\alpha(H) \approx \alpha_{\mathrm{max}} = 5.0$", fontsize=11, fontweight="bold", ha="center", color="#1e8449")
    plt.text(8.0, 2.5, "High-Entropy Cutoff\n(Prevents Text\nDegradation)", fontsize=10, fontweight="bold", ha="center", color="#922b21")

    # Threshold Text
    plt.text(1.25, 0.3, r"$\theta_{\mathrm{lo}} = 1.2$", fontsize=10, fontweight="bold", color="#34495e", rotation=90, va="bottom")
    plt.text(6.05, 0.3, r"$\theta_{\mathrm{hi}} = 6.0$", fontsize=10, fontweight="bold", color="#2ecc71", rotation=90, va="bottom")
    plt.text(7.05, 0.3, r"$\theta_{\mathrm{hi}} = 7.0$", fontsize=10, fontweight="bold", color="#27ae60", rotation=90, va="bottom")

    plt.xlabel(r"Predictive Next-Token Entropy $H$ (nats)", fontsize=13, fontweight="bold", labelpad=10)
    plt.ylabel(r"Steering Intensity $\alpha(H)$", fontsize=13, fontweight="bold", labelpad=10)
    plt.title(r"Optimal Predictive Entropy Bumper Gate Function $\alpha(H)$", fontsize=14, fontweight="bold", pad=15)

    plt.xlim(0.0, 10.0)
    plt.ylim(-0.2, 5.5)
    plt.legend(loc="upper right", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    out_file = OUT_DIR / "optimal_alpha_gating_function.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    print(f"Saved optimal alpha function plot to: {out_file}")

    # Copy to artifacts directory
    if ARTIFACTS_DIR.exists():
        shutil.copy(out_file, ARTIFACTS_DIR / out_file.name)
        print("Copied optimal alpha function plot to artifacts directory.")

if __name__ == "__main__":
    main()
