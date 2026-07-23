#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_entropy_sharp_alpha.py
#

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def sigmoid(x, k, theta):
    z = -k * (x - theta)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(z))

def gating_function(h, k_H, theta_H, max_alpha=5.0):
    gain = sigmoid(h, k_H, theta_H)
    return gain * max_alpha

def main():
    max_alpha = 5.0
    theta_H = 1.5
    k_H = 8.0
    
    plt.close("all")
    fig, (ax_curve, ax_tokens) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Left Subplot: Alpha Gating Curve for Entropy-Sharp-1.5
    h_values = np.linspace(0, 8, 500)
    alpha_curve = gating_function(h_values, k_H, theta_H, max_alpha)
    
    ax_curve.plot(h_values, alpha_curve, color="#8e44ad", linewidth=3.5, label=r"$\alpha_t$ (Entropy-Sharp-1.5)")
    ax_curve.fill_between(h_values, 0, alpha_curve, color="#8e44ad", alpha=0.08)
    
    # Draw zones
    ax_curve.axvspan(0, 1.5, color="#e74c3c", alpha=0.04)
    ax_curve.text(0.75, 2.5, "Syntax Protected\n(Steering OFF)\n" + r"$\alpha \approx 0$", color="#c0392b", fontsize=9.5, fontweight="bold", ha="center", va="center")
    
    ax_curve.axvspan(1.5, 8, color="#2ecc71", alpha=0.04)
    ax_curve.text(4.75, 2.5, "Steering Active\n(Full Intensity)\n" + r"$\alpha = 5.0$", color="#27ae60", fontsize=10, fontweight="bold", ha="center", va="center")
    
    ax_curve.axvline(x=1.5, color="#8e44ad", linestyle="--", linewidth=2.0)
    
    ax_curve.set_title(r"Predictive Entropy Gate ($\theta_H=1.5, k_H=8.0$)", fontsize=13, fontweight="bold", pad=12)
    ax_curve.set_xlabel("Predictive Entropy H [bits]\n<- Grammar / Meaning ->", fontsize=11, labelpad=8)
    ax_curve.set_ylabel("Applied Steering Intensity (Alpha)", fontsize=11, labelpad=8)
    ax_curve.set_xlim(0, 8)
    ax_curve.set_ylim(-0.2, 5.5)
    ax_curve.grid(True, linestyle=":", alpha=0.5)
    ax_curve.legend(loc="upper right", fontsize=9.5)
    
    # 2. Right Subplot: Token Sequence Simulation
    tokens = ["The", "astronaut", "accidentally", "stained", "the", "highly", "confidential", "blueprint", "of", "Voyager-1"]
    h_sim = np.array([1.0, 6.5, 5.5, 7.0, 0.8, 4.8, 7.8, 8.5, 1.2, 11.5])
    
    alpha_sim = gating_function(h_sim, k_H, theta_H, max_alpha)
    
    # Color bars: Purple if steering is active, Gray if protected
    colors = ["#8e44ad" if a > 2.5 else "#b0b0b0" for a in alpha_sim]
    
    bars = ax_tokens.bar(np.arange(len(tokens)), alpha_sim, color=colors, edgecolor="black", alpha=0.85, zorder=3)
    
    # Annotate H value on top of each bar
    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        ax_tokens.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"H: {h_sim[idx]:.1f}", 
                       ha='center', va='bottom', fontsize=8.5, fontweight="bold", color="#555555")
        
    ax_tokens.set_xticks(np.arange(len(tokens)))
    ax_tokens.set_xticklabels([f"'{t}'" for t in tokens], fontsize=10, fontweight="bold", rotation=30, ha="right")
    
    # Color text labels: Red for steered tokens (H >= 1.5)
    for idx, label in enumerate(ax_tokens.get_xticklabels()):
        if h_sim[idx] >= 1.5:
            label.set_color("#e74c3c")
            
    ax_tokens.set_title("Step-by-Step Steering Alpha Simulation", fontsize=13, fontweight="bold", pad=12)
    ax_tokens.set_xlabel("Generated Token Sequence", fontsize=11, labelpad=8)
    ax_tokens.set_ylabel("Applied Steering Intensity (Alpha)", fontsize=11, labelpad=8)
    ax_tokens.set_ylim(-0.2, 5.5)
    ax_tokens.grid(axis='y', linestyle=":", alpha=0.5)
    
    plt.suptitle("Intervention Control Gating Multiplier (Entropy-Sharp-1.5)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save image
    out_dir = Path("exp_token_intensity/exp_entropy_gating")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entropy_sharp_1.5_alpha_gating.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved Entropy-Sharp-1.5 gating plot to: {out_path}")
    
    # Copy to artifacts
    try:
        artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / "entropy_sharp_1.5_alpha_gating.png"
        shutil.copy(out_path, dest_path)
        print(f"Copied to artifact path: {dest_path}")
    except Exception as e:
        print(f"Error copying to artifact: {e}")

if __name__ == "__main__":
    main()
