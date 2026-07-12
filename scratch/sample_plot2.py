#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 1. Mathematically define fixed Positive Samples uniformly from -30 to 120 degrees
    angles = np.linspace(np.radians(-30), np.radians(120), 10)
    D_pos = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # w_avg is fixed exactly at 45 degrees (piercing the dead center of samples)
    theta_ref = np.radians(45)
    w_avg = np.array([np.cos(theta_ref), np.sin(theta_ref)])
    
    # --- Define Current States (d_h) for both scenarios ---
    # Scenario A: Well aligned (Angle = 40°, extremely close to w_avg)
    theta_h_A = np.radians(40)
    d_h_A = np.array([np.cos(theta_h_A), np.sin(theta_h_A)])
    
    # Scenario B: Deficient / Pulled away by negative prompt bias (Angle = -60°)
    theta_h_B = np.radians(-60)
    d_h_B = np.array([np.cos(theta_h_B), np.sin(theta_h_B)])
    
    # --- Calculate Exact Cosine Similarities for both scenarios ---
    S_i_A = np.dot(D_pos, d_h_A)
    S_center_A = np.dot(w_avg, d_h_A) # cos(5°) ≈ 0.99
    below_mask_A = S_i_A <= S_center_A
    
    S_i_B = np.dot(D_pos, d_h_B)
    S_center_B = np.dot(w_avg, d_h_B) # cos(105°) ≈ -0.26
    below_mask_B = S_i_B <= S_center_B

    # Setup 1-row, 2-column layout (Left: Integrated 2D, Right: Two-tier 1D)
    plt.close("all")
    fig, (ax_2d, ax_1d) = plt.subplots(1, 2, figsize=(14, 6))
    
    # =========================================================================
    # 1. LEFT GRAPH: Integrated 2D Vector Space (Both Scenarios Combined)
    # =========================================================================
    unit_circle = plt.Circle((0, 0), 1.0, color='#e5e5e5', fill=False, linestyle='--', linewidth=1.2, zorder=1)
    ax_2d.add_patch(unit_circle)
    
    # Plot positive baseline samples in a neutral corporate blue
    ax_2d.scatter(D_pos[:, 0], D_pos[:, 1], color="#2c3e50", s=100, zorder=3, label="Positive Samples")
    
    for i, (x, y) in enumerate(D_pos):
        ax_2d.annotate(f"{i+1}", (x*1.16, y*1.16), ha='center', va='center', fontsize=8.5, color="#555555", fontweight="bold")
        
    # Plot Average Vector w_avg as a prominent green star/point
    ax_2d.scatter(w_avg[0], w_avg[1], color="#2ca02c", marker="*", s=250, edgecolor="black", zorder=6, label="Avg Vector")
    ax_2d.quiver(0, 0, d_h_A[0], d_h_A[1], angles='xy', scale_units='xy', scale=1, 
                 color="#ff7f0e", width=0.007, zorder=5, label="State A")
    ax_2d.quiver(0, 0, d_h_B[0], d_h_B[1], angles='xy', scale_units='xy', scale=1, 
                 color="#d63031", width=0.007, zorder=5, label="State B")
    
    ax_2d.set_title("Integrated 2D Vector Space", fontsize=15, fontweight="bold", pad=12)
    ax_2d.set_xlim(-1.3, 1.3)
    ax_2d.set_ylim(-1.3, 1.3)
    ax_2d.axhline(0, color="#dddddd", linestyle=":", lw=1)
    ax_2d.axvline(0, color="#dddddd", linestyle=":", lw=1)
    ax_2d.set_xlabel("Dimension 1", fontsize=15)
    ax_2d.set_ylabel("Dimension 2", fontsize=15)
    ax_2d.grid(True, linestyle=":", alpha=0.4)
    ax_2d.set_aspect('equal')
    ax_2d.legend(loc="lower left", fontsize=15, frameon=True, facecolor="#ffffff")

    # =========================================================================
    # 2. RIGHT GRAPH: Integrated Two-Tier 1D Rankings
    # =========================================================================
    # Set distinct vertical centers for the two tiers to prevent collision
    y_center_A = 0.15
    y_center_B = -0.15
    
    # Fixed tiny jitters to handle overlapping symmetric pairs within each tier
    jitter = np.array([0.012 if i % 2 == 0 else -0.012 for i in range(10)])
    y_ticks_A = y_center_A + jitter
    y_ticks_B = y_center_B + jitter
    
    # --- Tier 1 (Upper): Scenario A Plotting ---
    ax_1d.scatter(S_i_A[below_mask_A], y_ticks_A[below_mask_A], color="#1f77b4", s=130, zorder=3)
    ax_1d.scatter(S_i_A[~below_mask_A], y_ticks_A[~below_mask_A], color="#b0b0b0", s=130, zorder=3)
    ax_1d.vlines(S_center_A, y_center_A - 0.08, y_center_A + 0.08, colors="#ff7f0e", linestyles="--", lw=2.5, zorder=4)
    ax_1d.fill_between([-1.05, S_center_A], y_center_A - 0.06, y_center_A + 0.06, color="#1f77b4", alpha=0.06, zorder=1)
    ax_1d.annotate('', xy=(S_center_A, y_center_A - 0.06), xytext=(-1.0, y_center_A - 0.06), 
                   arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.2))
    
    for i, val in enumerate(S_i_A):
        offset_y = 0.04 if i % 2 == 0 else -0.04
        ax_1d.annotate(f"{i+1}", (val, y_center_A + offset_y), ha='center', 
                       va='bottom' if i % 2 == 0 else 'top', fontsize=7.5, color="#555555", fontweight="bold")

    # --- Tier 2 (Lower): Scenario B Plotting ---
    ax_1d.scatter(S_i_B[below_mask_B], y_ticks_B[below_mask_B], color="#1f77b4", s=130, zorder=3)
    ax_1d.scatter(S_i_B[~below_mask_B], y_ticks_B[~below_mask_B], color="#b0b0b0", s=130, zorder=3)
    ax_1d.vlines(S_center_B, y_center_B - 0.08, y_center_B + 0.08, colors="#d63031", linestyles="--", lw=2.5, zorder=4)
    ax_1d.fill_between([-1.05, S_center_B], y_center_B - 0.06, y_center_B + 0.06, color="#1f77b4", alpha=0.06, zorder=1)
    ax_1d.annotate('', xy=(S_center_B, y_center_B - 0.06), xytext=(-1.0, y_center_B - 0.06), 
                   arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.2))
    
    for i, val in enumerate(S_i_B):
        offset_y = 0.04 if i % 2 == 0 else -0.04
        ax_1d.annotate(f"{i+1}", (val, y_center_B + offset_y), ha='center', 
                       va='bottom' if i % 2 == 0 else 'top', fontsize=7.5, color="#555555", fontweight="bold")

    # --- Add Structural Text Labels & Explanatory Boxes ---
    # Y-axis Tier Labels on the far left of 1D plot
    ax_1d.text(-1.12, y_center_A, "State A", ha='right', va='center', fontsize=15, fontweight='bold', color='#ff7f0e')
    ax_1d.text(-1.12, y_center_B, "State B", ha='right', va='center', fontsize=15, fontweight='bold', color='#d63031')
    
    # Retain the exact counting percentages
    count_A = np.sum(below_mask_A)
    count_B = np.sum(below_mask_B)
    ax_1d.text((-1.0 + S_center_A)/2, y_center_A - 0.09, f"Count ({int(count_A/10*100)}%)", color="#1f77b4", ha='center', va='top', fontsize=8.5, fontweight="bold")
    ax_1d.text((-1.0 + S_center_B)/2, y_center_B - 0.09, f"Count ({int(count_B/10*100)}%)", color="#1f77b4", ha='center', va='top', fontsize=8.5, fontweight="bold")

    # 1D Layout tuning
    ax_1d.set_title("Two-Tier 1D Relative Rankings", fontsize=15, fontweight="bold", pad=12)
    ax_1d.set_xlabel("Cosine Similarity", fontsize=15, labelpad=8)
    ax_1d.set_xlim(-1.05, 1.05)
    ax_1d.set_ylim(-0.46, 0.46) # Expanded vertically to accommodate tiers and labels safely
    ax_1d.set_yticks([])
    ax_1d.spines['left'].set_visible(False)
    ax_1d.spines['top'].set_visible(False)
    ax_1d.spines['right'].set_visible(False)
    ax_1d.grid(axis='x', linestyle=':', alpha=0.5)
    
    # Custom legend for 1D indicating what blue/grey means (drawn only once)
    handles, labels = ax_1d.get_legend_handles_labels()
    ax_1d.legend(handles[:2], labels[:2], loc="upper right", fontsize=8.5, frameon=True, facecolor="#ffffff")

    plt.tight_layout()
    
    # Save chart
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proj_rank_low_vs_high_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved highly condensed unified plot to: {out_path}")
    
    # Copy to artifact path
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "proj_rank_low_vs_high_comparison.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()