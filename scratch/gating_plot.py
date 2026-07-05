#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CONFIGS = {
    "Conf 1: No Gating":       {"theta_lo": 0.0, "theta_hi": 99.0, "k_lo": 1.0, "k_hi": 1.0},
    "Conf 2: Base Gating":     {"theta_lo": 3.0, "theta_hi": 7.0,  "k_lo": 2.0, "k_hi": 2.0},
    "Conf 3: Wider Gating":    {"theta_lo": 1.0, "theta_hi": 9.0,  "k_lo": 2.0, "k_hi": 2.0},
    "Conf 4: Narrower Gating": {"theta_lo": 4.0, "theta_hi": 6.0,  "k_lo": 2.0, "k_hi": 2.0},
    "Conf 5: Sharp Gating":    {"theta_lo": 3.0, "theta_hi": 7.0,  "k_lo": 8.0, "k_hi": 8.0},
    "Conf 6: Gentle Gating":   {"theta_lo": 3.0, "theta_hi": 7.0,  "k_lo": 0.5, "k_hi": 0.5}
}

# 視覚的なグループ分けを意識したカラーパレット
# ベース(青), 幅違い(緑系), 傾き違い(赤・紫)
COLORS = ["#7f7f7f", "#1f77b4", "#2ca02c", "#a6d854", "#d62728", "#9467bd"]

def sigmoid_lo(ic, theta_lo, k_lo):
    return 1.0 / (1.0 + np.exp(-k_lo * (ic - theta_lo)))

def sigmoid_hi(ic, theta_hi, k_hi):
    return 1.0 / (1.0 + np.exp(k_hi * (ic - theta_hi)))

def gate_function(ic, p, name=""):
    if "No Gating" in name:
        return np.ones_like(ic)
    return sigmoid_lo(ic, p["theta_lo"], p["k_lo"]) * sigmoid_hi(ic, p["theta_hi"], p["k_hi"])

def main():
    ic_values = np.linspace(0, 12, 1000)
    
    plt.close("all")
    fig = plt.figure(figsize=(15, 13))
    
    # 1. 個別グラフでの視覚的工夫（theta と k の役割を明示）
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.8], hspace=0.4, wspace=0.25)
    for i, (name, params) in enumerate(CONFIGS.items()):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        g_values = gate_function(ic_values, params, name)
        
        # グラフ線と領域の塗りつぶし
        ax.plot(ic_values, g_values, color=COLORS[i], lw=3, label=name)
        ax.fill_between(ic_values, 0, g_values, color=COLORS[i], alpha=0.15)
        
        # 【工夫点】しきい値 theta の位置を、そのConfの専用色で縦線プロット
        if "No Gating" not in name and params["theta_hi"] < 50:
            t_lo, t_hi = params["theta_lo"], params["theta_hi"]
            ax.axvline(t_lo, color=COLORS[i], linestyle="--", lw=1.5, alpha=0.8)
            ax.axvline(t_hi, color=COLORS[i], linestyle="--", lw=1.5, alpha=0.8)
            
            # グラフ内に theta の値をテキスト表示して視覚支援
            ax.text(t_lo - 0.2, 0.5, f'$\\theta_{{lo}}={t_lo}$', color=COLORS[i], 
                    ha='right', va='center', fontsize=9, fontweight='bold')
            ax.text(t_hi + 0.2, 0.5, f'$\\theta_{{hi}}={t_hi}$', color=COLORS[i], 
                    ha='left', va='center', fontsize=9, fontweight='bold')
            
            # 【工夫点】傾き k の特徴をテキストで直感的に補足
            ax.text(6.0, 0.1, f'$k_{{lo}}={params["k_lo"]}, k_{{hi}}={params["k_hi"]}$', 
                    color="#444444", ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="gray", lw=0.5, alpha=0.8))

        ax.set_title(name, fontsize=12, fontweight="bold", color="#222222")
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Gate Factor G(IC_t)", fontsize=9)
        ax.grid(linestyle=":", alpha=0.5)

    # 2. 重ね合わせ比較グラフ（パラメータによる形状変化のダイナミックな比較）
    ax_all = fig.add_subplot(grid[3, :])
    for i, (name, params) in enumerate(CONFIGS.items()):
        g_values = gate_function(ic_values, params, name)
        linestyle = "--" if "Conf 1" in name else "-"
        alpha = 0.5 if "Conf 1" in name else 0.9
        ax_all.plot(ic_values, g_values, color=COLORS[i], lw=2.5, linestyle=linestyle, alpha=alpha, label=name)
        
    ax_all.set_title("Comparison of All Gating Configurations (Mathematical Behavior)", fontsize=14, fontweight="bold", pad=12)
    ax_all.set_xlabel("Surprisal $IC_t$ (bits)", fontsize=12)
    ax_all.set_ylabel("Gate Factor $G(IC_t)$", fontsize=12)
    ax_all.set_xlim(0, 12)
    ax_all.set_ylim(-0.05, 1.05)
    ax_all.set_xticks(range(0, 13))
    ax_all.grid(linestyle=":", alpha=0.6)
    
    # 凡例をわかりやすく右上に配置
    ax_all.legend(loc="upper right", fontsize=10, frameon=True, shadow=True, facecolor="#ffffff")
    
    plt.suptitle("Visualizing $\\theta$ (Threshold) and $k$ (Slope) in Alpha Gating Functions", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # 保存処理
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gating_configurations_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved enhanced visualization plot to: {out_path}")
    
    # アーティファクトへのコピー
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "gating_configurations_comparison.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()