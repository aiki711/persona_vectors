#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def main():
    # 1. 数学的に固定された正例サンプルを一様に定義 (-30度から120度)
    angles = np.linspace(np.radians(-30), np.radians(120), 10)
    D_pos = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # w_avg を45度に固定
    theta_ref = np.radians(45)
    w_avg = np.array([np.cos(theta_ref), np.sin(theta_ref)])
    
    # --- 各レイヤーのベクトル状態を定義 ---
    # Layer A: 良好に整列している状態 (40度)
    theta_h_A = np.radians(40)
    d_h_A = np.array([np.cos(theta_h_A), np.sin(theta_h_A)])
    
    # Layer B: ネガティブプロンプトのバイアス等で引っ張られ、乖離した状態 (-60度)
    theta_h_B = np.radians(-60)
    d_h_B = np.array([np.cos(theta_h_B), np.sin(theta_h_B)])
    
    # --- コサイン類似度の計算 ---
    S_i_A = np.dot(D_pos, d_h_A)
    S_center_A = np.dot(w_avg, d_h_A)
    below_mask_A = S_i_A <= S_center_A
    
    S_i_B = np.dot(D_pos, d_h_B)
    S_center_B = np.dot(w_avg, d_h_B)
    below_mask_B = S_i_B <= S_center_B

    plt.close("all")
    fig = plt.figure(figsize=(14, 8.5)) 
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 0.5]) 
    
    ax_2d_A = fig.add_subplot(gs[0, 0])
    ax_2d_B = fig.add_subplot(gs[0, 1])
    ax_1d_A = fig.add_subplot(gs[1, 0])
    ax_1d_B = fig.add_subplot(gs[1, 1])
    
    # 描画用共通関数
    def setup_2d_plot(ax, d_h, color, label_name, title):
        unit_circle = plt.Circle((0, 0), 1.0, color='#e5e5e5', fill=False, linestyle='--', linewidth=1.5, zorder=1)
        ax.add_patch(unit_circle)
        
        # 【変更点】正例サンプルの点を大きく (s=100 -> s=220)
        ax.scatter(D_pos[:, 0], D_pos[:, 1], color="#2c3e50", s=220, zorder=3, label="Positive Samples")
        
        # 点が大きくなったので文字の配置半径を少し外側に調整 (1.16 -> 1.20)
        for i, (x, y) in enumerate(D_pos):
            ax.annotate(f"{i+1}", (x*1.20, y*1.20), ha='center', va='center', fontsize=10, color="#333333", fontweight="bold")
            
        # 【変更点】平均ベクトル（星）をさらに巨大化 (s=250 -> s=500)、枠線も太く
        ax.scatter(w_avg[0], w_avg[1], color="#2ca02c", marker="*", s=500, edgecolor="black", linewidth=1.5, zorder=6, label="Avg Vector")
        
        # 【変更点】状態ベクトルの矢印を太く力強く (width=0.007 -> width=0.012)
        ax.quiver(0, 0, d_h[0], d_h[1], angles='xy', scale_units='xy', scale=1, color=color, width=0.012, zorder=5, label=label_name)
        
        ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color="#dddddd", linestyle=":", lw=1.2)
        ax.axvline(0, color="#dddddd", linestyle=":", lw=1.2)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_aspect('equal')
        ax.legend(loc="lower left", fontsize=14, frameon=True, facecolor="#ffffff")

    def setup_1d_plot(ax, S_i, S_center, below_mask, color, title):
        y_center = 0.0
        # 点が大きくなるため、上下のジッター（重なり回避幅）を少し広げて調整
        jitter = np.array([0.03 if i % 2 == 0 else -0.03 for i in range(10)])
        y_ticks = y_center + jitter
        
        # 【変更点】1D帯の中のプロット点を大きく (s=110 -> s=220)
        ax.scatter(S_i[below_mask], y_ticks[below_mask], color="#1f77b4", s=220, zorder=3)
        ax.scatter(S_i[~below_mask], y_ticks[~below_mask], color="#b0b0b0", s=220, zorder=3)
        
        # 【変更点】閾値線（点線）をさらに太く (lw=2.5 -> lw=4.0)
        ax.vlines(S_center, y_center - 0.15, y_center + 0.15, colors=color, linestyles="--", lw=4.0, zorder=4)
        ax.fill_between([-1.05, S_center], y_center - 0.15, y_center + 0.15, color="#1f77b4", alpha=0.06, zorder=1)
        
        # 【変更点】下部の範囲を示す矢印を太く (lw=1.2 -> lw=2.0)
        ax.annotate('', xy=(S_center, y_center - 0.15), xytext=(-1.0, y_center - 0.15), 
                    arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=2.0))
        
        # 点が大きくなったので、文字の上下オフセットを微調整 (0.045 -> 0.08)
        for i, val in enumerate(S_i):
            offset_y = 0.08 if i % 2 == 0 else -0.08
            ax.annotate(f"{i+1}", (val, y_center + offset_y), ha='center', 
                        va='bottom' if i % 2 == 0 else 'top', fontsize=9, color="#333333", fontweight="bold")

        ax.set_title(title, fontsize=20, fontweight="bold", pad=8)
        ax.set_xlabel("Cosine Similarity", fontsize=15, labelpad=3)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.25, 0.35)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle=':', alpha=0.5)

    # =========================================================================
    # 各サブプロットの実行（左列にLayer A、右列にLayer Bを配置）
    # =========================================================================
    # 左列 (0列目): Layer A 一式
    setup_2d_plot(ax_2d_A, d_h_A, "#ff7f0e", "Layer A State", "Layer A: 2D Vector Space")
    setup_1d_plot(ax_1d_A, S_i_A, S_center_A, below_mask_A, "#ff7f0e", "Layer A: Relative Rankings")

    # 右列 (1列目): Layer B 一式
    setup_2d_plot(ax_2d_B, d_h_B, "#d63031", "Layer B State", "Layer B: 2D Vector Space")
    setup_1d_plot(ax_1d_B, S_i_B, S_center_B, below_mask_B, "#d63031", "Layer B: Relative Rankings")

    plt.tight_layout()
    
    # 画像保存の処理
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proj_rank_low_vs_high_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved independent layers plot to: {out_path}")
    
    # 成果物パスへのコピー
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "proj_rank_low_vs_high_comparison.png"
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()