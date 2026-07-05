#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters for Gentle Gating (Conf 6)
THETA_LO = 3.0
THETA_HI = 7.0
K_LO = 0.5
K_HI = 0.5

def sigmoid_lo(ic, theta_lo, k_lo):
    return 1.0 / (1.0 + np.exp(-k_lo * (ic - theta_lo)))

def sigmoid_hi(ic, theta_hi, k_hi):
    return 1.0 / (1.0 + np.exp(k_hi * (ic - theta_hi)))

def standard_gating(ic, theta_lo, theta_hi, k_lo, k_hi):
    return sigmoid_lo(ic, theta_lo, k_lo) * sigmoid_hi(ic, theta_hi, k_hi)

def max_normalized_gating(ic, theta_lo, theta_hi, k_lo, k_hi):
    # Calculate G_max using a grid search
    ic_grid = np.linspace(0.0, 30.0, 3000)
    g_grid = standard_gating(ic_grid, theta_lo, theta_hi, k_lo, k_hi)
    g_max = np.max(g_grid)
    
    return standard_gating(ic, theta_lo, theta_hi, k_lo, k_hi) / g_max

def plateau_gating(ic, theta_lo, theta_hi, k_lo, k_hi):
    # Piecewise definition
    cond_left = ic < theta_lo
    cond_mid = (ic >= theta_lo) & (ic <= theta_hi)
    cond_right = ic > theta_hi
    
    # We use np.piecewise to evaluate cleanly for array inputs
    def left_func(x):
        return 2.0 / (1.0 + np.exp(-k_lo * (x - theta_lo)))
        
    def mid_func(x):
        return np.ones_like(x)
        
    def right_func(x):
        return 2.0 / (1.0 + np.exp(k_hi * (x - theta_hi)))
        
    # Build array
    res = np.zeros_like(ic)
    # Handle scalar or array
    if np.isscalar(ic):
        if ic < theta_lo:
            return left_func(ic)
        elif ic <= theta_hi:
            return 1.0
        else:
            return right_func(ic)
            
    res[cond_left] = left_func(ic[cond_left])
    res[cond_mid] = 1.0
    res[cond_right] = right_func(ic[cond_right])
    return res

def main():
    ic_values = np.linspace(0, 12, 1000)
    
    plt.close("all")
    plt.figure(figsize=(10, 6))
    
    g_standard = standard_gating(ic_values, THETA_LO, THETA_HI, K_LO, K_HI)
    g_max_norm = max_normalized_gating(ic_values, THETA_LO, THETA_HI, K_LO, K_HI)
    g_plateau = plateau_gating(ic_values, THETA_LO, THETA_HI, K_LO, K_HI)
    
    plt.plot(ic_values, g_standard, label="Standard Gentle Gating (Conf 6)", color="#9467bd", lw=2.5)
    plt.plot(ic_values, g_max_norm, label="Max-Normalized Gentle Gating (Conf 6)", color="#1f77b4", lw=2.5, linestyle="--")
    plt.plot(ic_values, g_plateau, label="Plateau Gentle Gating (Conf 6)", color="#2ca02c", lw=2.5, linestyle="-.")
    
    plt.axvline(THETA_LO, color="red", linestyle=":", alpha=0.5, label="$\\theta_{lo}=3.0$")
    plt.axvline(THETA_HI, color="red", linestyle=":", alpha=0.5, label="$\\theta_{hi}=7.0$")
    
    plt.title("Comparison of Gating Improvements for Gentle Gating (Conf 6)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Surprisal $IC_t$ (bits)", fontsize=12)
    plt.ylabel("Gate Factor $G(IC_t)$", fontsize=12)
    plt.xlim(0, 12)
    plt.ylim(-0.05, 1.05)
    plt.grid(linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "improved_gating_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved improved gating plot to: {out_path}")
    
    # Copy to artifacts directory
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "improved_gating_comparison.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()
