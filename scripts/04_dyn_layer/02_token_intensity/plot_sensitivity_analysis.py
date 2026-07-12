#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/plot_sensitivity_analysis.py
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def binned_average(x, y, bins=30):
    bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    for i in range(bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(y[mask]))
        else:
            bin_means.append(np.nan)
    bin_means = np.array(bin_means)
    # Simple linear interpolation for NaNs
    nans = np.isnan(bin_means)
    if np.any(nans):
        if not np.all(nans):
            bin_means[nans] = np.interp(bin_centers[nans], bin_centers[~nans], bin_means[~nans])
        else:
            bin_means[nans] = 0.0
    return bin_centers, bin_means

def main():
    plt.close("all")
    
    csv_path = Path("exp_token_intensity/exp_sensitivity_analysis/results/token_sensitivity_records.csv")
    if not csv_path.exists():
        print(f"Error: Results CSV file does not exist at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    # Remove outliers or NaNs
    df = df.dropna(subset=["ic", "kl", "align_gain"])
    # Ignore negative or extreme values that might be artifacts of tokenizer EOS
    df = df[df["ic"] < 25.0]

    ic = df["ic"].values
    kl = df["kl"].values
    gain = df["align_gain"].values

    # 1. Plot Scatter and Trends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Alignment Gain vs IC
    ax1.scatter(ic, gain, color="#3498db", alpha=0.15, s=12, label="Token Data Points")
    bin_centers, bin_gain = binned_average(ic, gain, bins=40)
    ax1.plot(bin_centers, bin_gain, color="#2980b9", linewidth=3.0, label="Binned Avg Trend")
    
    # Fit polynomial curve
    p_gain = np.polyfit(ic, gain, deg=4)
    grid_ic = np.linspace(0.0, 18.0, 200)
    ax1.plot(grid_ic, np.polyval(p_gain, grid_ic), color="#e74c3c", linestyle="--", linewidth=2.0, label="4-deg Poly Fit")
    
    ax1.set_xlabel("Information Content (IC) [bits]", fontsize=11)
    ax1.set_ylabel("Alignment Gain (Projection Shift)", fontsize=11)
    ax1.set_title("Alignment Gain vs. Token Information Content", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_xlim(0, 18)
    ax1.legend(loc="upper right")

    # Plot 2: KL Divergence (Distortion) vs IC
    ax2.scatter(ic, kl, color="#2ecc71", alpha=0.15, s=12, label="Token Data Points")
    _, bin_kl = binned_average(ic, kl, bins=40)
    ax2.plot(bin_centers, bin_kl, color="#27ae60", linewidth=3.0, label="Binned Avg Trend")
    
    # Fit polynomial curve (forcing non-negative or matching trend)
    p_kl = np.polyfit(ic, kl, deg=4)
    ax2.plot(grid_ic, np.polyval(p_kl, grid_ic), color="#e74c3c", linestyle="--", linewidth=2.0, label="4-deg Poly Fit")

    ax2.set_xlabel("Information Content (IC) [bits]", fontsize=11)
    ax2.set_ylabel("Distortion (KL Divergence)", fontsize=11)
    ax2.set_title("Language Probability Distortion vs. Token IC", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_xlim(0, 18)
    ax2.set_ylim(0, max(np.percentile(kl, 98), 0.5))
    ax2.legend(loc="upper right")

    plt.suptitle("Sensitivity Analysis: Intervention Effects by Information Content", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save Plot 1
    figures_dir = Path("exp_token_intensity/exp_sensitivity_analysis/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot1_path = figures_dir / "sensitivity_scatter_trends.png"
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    print(f"Saved scatter trends to: {plot1_path}")

    # 2. Compute Gating Efficiency Curve
    # Use binned values to prevent overfitting noise
    # We enforce non-negative values for gain and distortion
    f_align = np.polyval(p_gain, grid_ic)
    f_dist = np.polyval(p_kl, grid_ic)
    
    # Keep curves bounded and clean
    f_align = np.maximum(f_align, 0.0)
    f_dist = np.maximum(f_dist, 1e-4) # Avoid division by zero

    efficiency = f_align / f_dist
    
    # Normalize to 0-1
    efficiency = np.maximum(efficiency, 0.0)
    if np.max(efficiency) > 0:
        efficiency_norm = efficiency / np.max(efficiency)
    else:
        efficiency_norm = efficiency

    # Plot Gating Efficiency
    plt.figure(figsize=(8, 6))
    plt.plot(grid_ic, efficiency_norm, color="#8e44ad", linewidth=3.0, label="Empirical Gating Efficiency")
    plt.fill_between(grid_ic, 0, efficiency_norm, color="#8e44ad", alpha=0.1)
    
    # Highlight Peak / Sweet Spot
    peak_idx = np.argmax(efficiency_norm)
    peak_ic = grid_ic[peak_idx]
    plt.axvline(x=peak_ic, color="#e74c3c", linestyle=":", linewidth=2.0, label=f"Optimal Peak IC = {peak_ic:.2f}")
    
    # Also find cliff edge (where efficiency drops below 0.1)
    cliff_edge_idx = np.where((grid_ic > peak_ic) & (efficiency_norm < 0.1))[0]
    if len(cliff_edge_idx) > 0:
        cliff_ic = grid_ic[cliff_edge_idx[0]]
        plt.axvline(x=cliff_ic, color="#d35400", linestyle=":", linewidth=2.0, label=f"Optimal Cliff Edge IC = {cliff_ic:.2f}")

    plt.xlabel("Information Content (IC) [bits]", fontsize=11)
    plt.ylabel("Optimal Gating Intensity (Normalized)", fontsize=11)
    plt.title("Empirically Derived Optimal Gating Function", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.xlim(0, 15)
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right")
    
    # Save Plot 2
    plot2_path = figures_dir / "gating_efficiency_curve.png"
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    print(f"Saved gating efficiency curve to: {plot2_path}")

    # Copy to artifacts directory
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(plot1_path, artifact_dir / "sensitivity_scatter_trends.png")
    shutil.copy(plot2_path, artifact_dir / "gating_efficiency_curve.png")
    print("Successfully copied figures to artifacts.")

if __name__ == "__main__":
    main()
