import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def sigmoid(x, k, theta):
    # オーバーフロー防止用のクリッピング
    z = -k * (x - theta)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(z))

def gating_function(ic, k_lo, k_hi, theta_lo, theta_hi, max_value=1.0):
    """最大値を1.0に正規化したゲートゲインを返す"""
    g = sigmoid(ic, k_lo, theta_lo) * sigmoid(ic, -k_hi, theta_hi)
    # グリッドサーチで理論上の最大値を正確に取得
    grid = np.linspace(theta_lo - 1.0, theta_hi + 1.0, 1000)
    g_grid = sigmoid(grid, k_lo, theta_lo) * sigmoid(grid, -k_hi, theta_hi)
    g_max = np.max(g_grid)
    return (g / g_max) * max_value

def main():
    # 最大上限を 1.0 (0%〜100%) に設定
    max_gain = 1.0
    
    # A-Conf 3 (Low IC Focus) パラメータ
    theta_lo = 6.0   # 機能語（1~3）の終わり際から介入開始
    theta_hi = 10.0   # 内容語（4~10）の終わり際までしっかり介入をキープ！
    k_lo = 1.5
    k_hi = 2.5
    
    plt.close("all")
    fig, (ax_curve, ax_tokens) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 左側のプロット: ゲイン曲線の描画
    ic_values = np.linspace(0, 15, 500)
    gain_curve = gating_function(ic_values, k_lo, k_hi, theta_lo, theta_hi, max_gain)
    
    ax_curve.plot(ic_values, gain_curve, color="#e74c3c", linewidth=3, label="Gating Gain $G_{norm}$ (0.0 - 1.0)")
    ax_curve.fill_between(ic_values, 0, gain_curve, color="#e74c3c", alpha=0.08)
    
    ax_curve.set_title("Gating Function: Gain vs Token Information Content", fontsize=13, fontweight="bold", pad=12)
    ax_curve.set_xlabel("Information Content (IC) [bits]\n<- Common (Low IC) / Rare (High IC) ->", fontsize=11, labelpad=8)
    ax_curve.set_ylabel("Intervention Rate (Gain)", fontsize=11, labelpad=8)
    ax_curve.set_xlim(0, 15)
    ax_curve.set_ylim(0, 1.1)  # 0〜1が綺麗に収まるように1.1に設定
    ax_curve.grid(True, linestyle=":", alpha=0.5)
    ax_curve.legend(loc="upper right", fontsize=9.5, frameon=True)
    
    # 2. 右側のプロット: トークン系列のシミュレーション
    tokens = ["The", "astronaut", "accidentally", "stained", "the", "highly", "confidential", "blueprint", "of", "Voyager-1"]
    ic_sim = np.array([1.0, 6.5, 5.5, 7.0, 0.8, 4.8, 7.8, 8.5, 1.2, 13.5])
    
    gain_sim = gating_function(ic_sim, k_lo, k_hi, theta_lo, theta_hi, max_gain)
    
    # ゲインが 0.4（40%）以上なら赤（フル介入）、それ未満ならグレー（保護・ブレーキ）に色分け
    colors = ["#e74c3c" if g > 0.4 else "#b0b0b0" for g in gain_sim]
    
    bars = ax_tokens.bar(np.arange(len(tokens)), gain_sim, color=colors, edgecolor="black", alpha=0.85, zorder=3)
    
    # 棒の上に IC の値をテキスト表示
    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        ax_tokens.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"IC: {ic_sim[idx]:.1f}", 
                       ha='center', va='bottom', fontsize=8.5, fontweight="bold", color="#555555")
        
    ax_tokens.set_xticks(np.arange(len(tokens)))
    ax_tokens.set_xticklabels([f"'{t}'" for t in tokens], fontsize=10, fontweight="bold", rotation=30, ha="right")
    
    for idx, label in enumerate(ax_tokens.get_xticklabels()):
        ic_val = ic_sim[idx]
        if 6.0 <= ic_val <= 10.0:
            label.set_color("#e74c3c")
    
    ax_tokens.set_title("Step-by-Step Simulation of Gating Gain (Plateau Mode)", fontsize=13, fontweight="bold", pad=12)
    ax_tokens.set_xlabel("Generated Token Sequence", fontsize=11, labelpad=8)
    ax_tokens.set_ylabel("Applied Intervention Rate (Gain)", fontsize=11, labelpad=8)
    ax_tokens.set_ylim(0, 1.1)
    ax_tokens.grid(axis='y', linestyle=":", alpha=0.5)
    
    plt.suptitle("Mechanism of Token-Level Dynamic Gating Gain Control (Normalized 0.0 - 1.0)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # 画像の保存処理
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "surprisal_gating_normalized.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved normalized plot to: {out_path}")
    
    try:
        artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / "surprisal_gating_normalized.png"
        import shutil
        shutil.copy(out_path, dest_path)
        print(f"Copied to artifact path: {dest_path}")
    except Exception as e:
        print(f"Error copying to artifact: {e}")

if __name__ == "__main__":
    main()