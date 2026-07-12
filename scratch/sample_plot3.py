#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec  # 隙間制御のために追加

def plot_layer_tier(ax, layer_name, S_i, S_center, bg_color):
    below_mask = S_i <= S_center
    below_count = np.sum(below_mask)
    pct = below_count / len(S_i)
    final_score = 1.0 - pct

    # Fixed vertical jitter to keep 10 points clearly visible
    y_jitter = np.array([0.015 if i % 2 == 0 else -0.015 for i in range(10)])
    
    # 1. Plot individual samples relative to the current layer's state d_h
    ax.scatter(S_i[below_mask], y_jitter[below_mask], color="#1f77b4", s=130, zorder=3)
    ax.scatter(S_i[~below_mask], y_jitter[~below_mask], color="#b0b0b0", s=130, zorder=3)
    
    # Alternate labels up and down to guarantee legibility
    for i, val in enumerate(S_i):
        offset_y = 0.045 if i % 2 == 0 else -0.045
        va_dir = 'bottom' if i % 2 == 0 else 'top'
        ax.annotate(f"{i+1}", (val, offset_y), ha='center', va=va_dir, fontsize=7.5, color="#555555", fontweight="bold")
        
    # 2. Draw the vertical barrier line for S_center (Orange)
    ax.axvline(S_center, color="#ff7f0e", linestyle="--", lw=2.5, zorder=4, 
               label="Avg Similarity (S_center)" if layer_name == "Layer 4" else "")
    
    # 3. Highlight the percentile count area
    ax.axvspan(-1.05, S_center, color="#1f77b4", alpha=0.06, zorder=1)
    ax.annotate('', xy=(S_center, -0.09), xytext=(-1.0, -0.09), arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.2))
    ax.text((-1.0 + S_center)/2, -0.13, f"Count ({int(pct*100)}%)", color="#1f77b4", ha='center', va='top', fontsize=8.5, fontweight="bold")
    
    # 4. Math logic explanation box for this layer
    box_text = (
        f"- Percentile = {pct:.2f}\n"
        f"- Rank Score = {final_score:.2f}"
    )
    ax.text(-0.95, 0.25, box_text, fontsize=9, va='top', ha='left', bbox=dict(boxstyle="round,pad=0.4", fc=bg_color, ec="#dddddd"))

    # Subplot styling
    ax.set_ylabel(f"{layer_name}", fontsize=11, fontweight="bold", color="#2c3e50", rotation=0, labelpad=45, va='center')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.20, 0.29)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.5)

def main():
    # 1. Define Positive Samples with a smooth density gradient (Asymmetric distribution)
    grid = np.linspace(0, 1, 10)
    angles = np.radians(-10) + np.radians(120) * (grid ** 1.8)
    D_pos = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Calculate the true geometric mean vector (w_avg) dynamically
    w_avg_raw = np.mean(D_pos, axis=0)
    w_avg = w_avg_raw / np.linalg.norm(w_avg_raw)
    
    # 2. Define 3 representative layers with distinct alignment behaviors
    layers_data = [
        {
            "name": "Layer 1", 
            "theta_h": np.radians(-130), 
            "bg": "#fce8e6", 
        },
        {
            "name": "Layer 2", 
            "theta_h": np.radians(35),  
            "bg": "#e6f4ea", 
        },
        {
            "name": "Layer N", 
            "theta_h": np.radians(75),  
            "bg": "#fff7e6", 
        }
    ]
    
    plt.close("all")
    
    # 【変更点】隙間を確保するため、縦サイズを少し広げ（6->7）、GridSpec（4行構成）を定義
    # height_ratios の 3番目（0.4）が Layer 2 と Layer N の間の隙間の広さになります
    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 0.4, 1])
    
    # 各レイヤーのアキシスを配置し、X軸を同期 (sharex)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Layer 2 と Layer N の間に「見えないダミーの空間」を作成して隙間を開ける
    ax_space = fig.add_subplot(gs[2])
    ax_space.axis('off') # 枠線や目盛りを完全非表示にする
    
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    axes = [ax1, ax2, ax3]
    
    # 途中のレイヤー（Layer 1, Layer 2）のX軸の目盛り数値を非表示にする（sharex=Trueの挙動再現）
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    for i, ly in enumerate(layers_data):
        d_h = np.array([np.cos(ly["theta_h"]), np.sin(ly["theta_h"])])
        
        # Calculate rigorous cosine similarities relative to the CURRENT layer state d_h
        S_i = np.dot(D_pos, d_h)
        S_center = np.dot(w_avg, d_h)
        
        plot_layer_tier(axes[i], ly["name"], S_i, S_center, ly["bg"])
        
    # Shared X-axis label at the bottom (Layer N の下部に配置)
    axes[-1].set_xlabel("Cosine Similarity", fontsize=15, labelpad=10)
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#ffffff")
    
    plt.tight_layout()
    
    # Save chart
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proj_rank_layer_evolution.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved rigorously synchronized multi-layer plot to: {out_path}")
    
    # Copy to artifact path
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "proj_rank_layer_evolution.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()