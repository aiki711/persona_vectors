#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Fix random seed for reproducibility of the positive cluster
    np.random.seed(42)
    
    # 1. Define Origin (0,0) as the Midpoint m'
    # 2. Generate 10 Positive Samples (D_pos) clustered around (0.6, 0.6)
    cluster_center = np.array([0.6, 0.6])
    D_pos = cluster_center + np.random.normal(0, 0.12, size=(10, 2))
    
    # 3. Calculate the Average Positive Vector (w_avg)
    w_avg = np.mean(D_pos, axis=0)
    
    # 4. Define Current Hidden State (d_h) for two scenarios
    # Scenario A: Well aligned with w_avg (Small angle / High similarity)
    d_h_A = np.array([0.65, 0.52]) 
    # Scenario B: Disaligned / Pulled by prompt bias (Large angle / Low similarity)
    d_h_B = np.array([-0.45, 0.20])
    
    # Set up the 2D Vector Space Plot (1 row, 2 columns)
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    
    scenarios = [
        {"ax": axes[0], "d_h": d_h_A, "title": "Scenario A: Low Score (Already Aligned)", "color": "#e6f4ea"},
        {"ax": axes[1], "d_h": d_h_B, "title": "Scenario B: High Score (Deficient / Biased)", "color": "#fce8e6"}
    ]
    
    for sc in scenarios:
        ax = sc["ax"]
        d_h = sc["d_h"]
        
        # Plot Midpoint as Origin
        ax.scatter(0, 0, color="black", s=100, zorder=5, label="Midpoint $m'$ (Origin)")
        
        # Plot Positive Samples as a 2D density cluster
        ax.scatter(D_pos[:, 0], D_pos[:, 1], color="#1f77b4", s=80, alpha=0.7, zorder=3, label="Positive Samples ($D_{pos}$)")
        
        # Draw Average Positive Vector (w_avg) arrow
        ax.quiver(0, 0, w_avg[0], w_avg[1], angles='xy', scale_units='xy', scale=1, 
                  color="#2ca02c", width=0.008, zorder=4, label="Avg Pos Direction ($w_{avg}$)")
        
        # Draw Current Hidden State (d_h) arrow
        ax.quiver(0, 0, d_h[0], d_h[1], angles='xy', scale_units='xy', scale=1, 
                  color="#ff7f0e", width=0.008, zorder=4, label="Current State ($d_h$)")
        
        # Visual styling
        ax.set_title(sc["title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(-0.8, 1.1)
        ax.set_ylim(-0.8, 1.1)
        ax.axhline(0, color="#cccccc", linestyle=":", lw=1)
        ax.axvline(0, color="#cccccc", linestyle=":", lw=1)
        ax.set_xlabel("Latent Dimension X (PC 1)", fontsize=10)
        ax.set_ylabel("Latent Dimension Y (PC 2)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="#ffffff")
        
        # Calculate cosine similarity for the text box explanation
        cos_sim = np.dot(w_avg, d_h) / (np.linalg.norm(w_avg) * np.linalg.norm(d_h))
        
        # Status explanation box
        box_text = (
            f"[Geometric Analysis]\n"
            f"- Origin (0,0) represents Midpoint\n"
            f"- Angle between $w_{{avg}}$ & $d_h$ determines Sim\n"
            f"- Current CosSim = {cos_sim:.2f}\n\n"
            f"Resulting Rank Score:\n"
            f"-> {'0.10 (Brake Applied)' if sc['color'] == '#e6f4ea' else '0.90 (Full Accel)'}"
        )
        ax.text(-0.75, 1.0, box_text, fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.4", fc=sc["color"], ec="#dddddd"))

    plt.suptitle("2D Vector Projection of Hidden State Space (Before Rank Transformation)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Save image
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proj_rank_2d_vector_space.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved 2D vector space plot to: {out_path}")
    
    # Copy to artifact path
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "proj_rank_2d_vector_space.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()