#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_scenario(ax, S_i, S_center, title_text, is_low_score=True):
    # Split samples based on S_center
    below_center = S_i[S_i <= S_center]
    above_center = S_i[S_i > S_center]
    
    # 1. Plot individual samples (Blue for <= S_center, Grey for > S_center)
    ax.scatter(below_center, np.zeros_like(below_center), color="#1f77b4", s=150, zorder=3, 
               label=f"Samples (S_i <= S_center)")
    ax.scatter(above_center, np.zeros_like(above_center), color="#b0b0b0", s=150, zorder=3, 
               label=f"Samples (S_i > S_center)")
    
    # Annotate sample indices
    for i, val in enumerate(S_i):
        ax.annotate(f"S_{i+1}", (val, 0.02), ha='center', va='bottom', fontsize=8, color="#555555")
        
    # 2. Plot the baseline S_center vertical line (Orange)
    ax.axvline(S_center, color="#ff7f0e", linestyle="--", lw=2.5, zorder=2, 
               label=f"Avg Similarity (S_center = {S_center:.2f})")
    
    # 3. Highlight the percentile area with light blue
    ax.axvspan(0.1, S_center, color="#1f77b4", alpha=0.08, zorder=1)
    
    # 4. Add visual guidance arrows
    ax.annotate('', xy=(S_center, -0.05), xytext=(0.1, -0.05),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.5))
    
    pct = len(below_center) / len(S_i)
    final_score = 1.0 - pct
    
    ax.text((0.1 + S_center)/2, -0.09, f"Count range ({int(pct*100)}%)", 
            color="#1f77b4", ha='center', va='top', fontsize=9, fontweight="bold")
    
    # Explanatory text box for the scoring logic
    if is_low_score:
        box_color = "#e6f4ea" # Light green for aligned state
        status_text = "STATUS: Well Aligned"
    else:
        box_color = "#fce8e6" # Light red for unaligned state
        status_text = "STATUS: Deficient"

    text_box = (
        f"[Mathematical Logic]\n"
        f"- S_center clears {len(below_center)} / 10 samples\n"
        f"- Percentile = {pct:.2f}\n"
        f"- Final Score = 1.0 - {pct:.2f} = {final_score:.2f}\n\n"
        f"{status_text}"
    )
    ax.text(0.12, 0.22, text_box, fontsize=9.5, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.5", fc=box_color, ec="#dddddd"))

    # Subplot layout tuning
    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cosine Similarity", fontsize=10, labelpad=8)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(-0.15, 0.25)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="#ffffff")

def main():
    # 10 baseline positive samples
    S_i = np.array([0.25, 0.38, 0.45, 0.52, 0.58, 0.64, 0.71, 0.79, 0.85, 0.92])
    
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5)) # 1 row, 2 columns layout
    
    # Left Plot: Low Final Score Scenario (S_center is high)
    plot_scenario(axes[0], S_i, S_center=0.88, 
                  title_text="Scenario A: Low Final Score (S_center is High)", 
                  is_low_score=True)
    
    # Right Plot: High Final Score Scenario (S_center is low)
    plot_scenario(axes[1], S_i, S_center=0.30, 
                  title_text="Scenario B: High Final Score (S_center is Low)", 
                  is_low_score=False)
    
    plt.suptitle("Comparison of proj_rank Score Mechanics (N = 10)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save chart
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proj_rank_low_vs_high_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison plot to: {out_path}")
    
    # Copy to artifact path
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "proj_rank_low_vs_high_comparison.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()