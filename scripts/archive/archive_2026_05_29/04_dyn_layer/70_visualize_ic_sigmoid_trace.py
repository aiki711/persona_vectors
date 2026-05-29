#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 70_visualize_ic_sigmoid_trace.py
#
# ICシグモイドステアリングのトークンごとの IC・α_t の時系列を可視化するスクリプト。
# 以下を確認するためのデバッグ・解析ツール:
#   1. 機能語（低IC）のタイミングで α ≈ 0 になっているか
#   2. 内容語（高IC）のタイミングで α が適切にブーストされているか
#   3. シグモイドとハードゲートの α 分布の違い
#
# Usage:
#   python scripts/04_dyn_layer/70_visualize_ic_sigmoid_trace.py \
#     --config config/mistral_7b.yaml \
#     --vector_bank vectors/mean_diff_vectors.npz \
#     --axis neuroticism \
#     --target_layer 15 \
#     --alpha_max 20.0 \
#     --ic_theta 3.0 \
#     --ic_k 0.8 \
#     --out_dir exp_ic_sigmoid_alpha/figures/debug \
#     --prompt "Tell me about your feelings today."
#

import argparse
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

# ==================== ICスケーリング関数（69と共通） ====================

def sigmoid_alpha(ic: float, alpha_max: float, k: float, theta: float) -> float:
    x = k * (ic - theta)
    sig = 1.0 / (1.0 + math.exp(-x))
    return alpha_max * sig


def hard_gate_alpha(ic: float, alpha_max: float, theta: float) -> float:
    return alpha_max if ic >= theta else 0.0


def compute_ic_of_token(logits: torch.Tensor, selected_token_id: int) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    prob = probs[selected_token_id].clamp(min=1e-10).item()
    return -np.log2(prob)


# ==================== トレース付き生成 ====================

def generate_with_trace(
    model, tokenizer, prompt, w_dev,
    target_layer, alpha_max, ic_theta, ic_k,
    max_new_tokens=60,
    temperature=0.7,
):
    """
    sigmoid・ハードゲート両方のαを同時に記録しながら生成する（デバッグ用）。
    実際の介入は sigmoid モードで行い、ハードゲートは記録のみ。
    """
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    stack, _, _ = get_layer_stack(model)

    gen_ids = inputs.input_ids
    prompt_len = gen_ids.shape[1]

    prev_ic = 0.0
    trace = []

    for step in range(max_new_tokens):
        # 両モードのαを記録
        alpha_sigmoid = sigmoid_alpha(prev_ic, alpha_max, ic_k, ic_theta)
        alpha_hard = hard_gate_alpha(prev_ic, alpha_max, ic_theta)
        alpha_fixed = alpha_max

        # 実際の介入はシグモイドモードで
        _alpha = alpha_sigmoid

        def hook(mod, inp, out, _a=_alpha):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all(): return out
            if hs.size(1) != 1: return out
            hs_f32 = hs.to(torch.float32)
            steered = hs_f32 + _a * w_dev.view(1, 1, -1)
            if not torch.isfinite(steered).all(): return out
            steered = steered.to(hs.dtype)
            return (steered, *out[1:]) if isinstance(out, tuple) else steered

        handle = stack[target_layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                outputs = model(gen_ids)
            logits = outputs.logits[:, -1, :]
        finally:
            handle.remove()

        # サンプリング
        logits_scaled = logits.float() / temperature
        probs = F.softmax(logits_scaled, dim=-1)
        next_token = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)
        selected_id = next_token.item()
        token_str = tokenizer.decode([selected_id], skip_special_tokens=False)

        # 次ステップ用IC
        next_ic = compute_ic_of_token(logits[0], selected_id)

        trace.append({
            "step": step,
            "token": token_str.replace("\n", "↵").replace(" ", "·"),
            "ic": prev_ic,               # このステップで使われたIC（前トークン）
            "alpha_sigmoid": alpha_sigmoid,
            "alpha_hard": alpha_hard,
            "alpha_fixed": alpha_fixed,
        })

        prev_ic = next_ic
        gen_ids = torch.cat([gen_ids, next_token], dim=-1)

        if selected_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    return text, pd.DataFrame(trace)


# ==================== 可視化 ====================

def plot_sigmoid_curve(alpha_max, ic_k, ic_theta, out_path):
    """シグモイドゲーティング関数の形状を可視化"""
    ics = np.linspace(0, 10, 200)
    sigmoid_vals = [sigmoid_alpha(ic, alpha_max, ic_k, ic_theta) for ic in ics]
    hard_vals = [hard_gate_alpha(ic, alpha_max, ic_theta) for ic in ics]
    fixed_vals = [alpha_max] * len(ics)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ics, sigmoid_vals, label=f"Sigmoid (k={ic_k}, θ={ic_theta})", color="#2196F3", linewidth=2.5)
    ax.plot(ics, hard_vals, label=f"Hard Gate (θ={ic_theta})", color="#F44336", linewidth=2, linestyle="--")
    ax.axhline(alpha_max, label=f"Fixed (α={alpha_max})", color="#9E9E9E", linewidth=1.5, linestyle=":")
    ax.axvline(ic_theta, color="orange", linewidth=1.5, linestyle="--", alpha=0.7, label=f"θ={ic_theta} bits")
    ax.set_xlabel("Information Content IC (bits)", fontsize=12)
    ax.set_ylabel("Applied α", fontsize=12)
    ax.set_title("IC-based α Scaling Functions", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, alpha_max * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_token_trace(df: pd.DataFrame, out_path: Path, title_suffix: str, ic_theta: float):
    """トークンごとのIC・α_t 時系列プロット"""
    n = len(df)
    x = np.arange(n)
    labels = df["token"].tolist()

    fig = plt.figure(figsize=(max(14, n * 0.4), 14))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    # --- Plot 1: α比較 ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, df["alpha_fixed"], color="#9E9E9E", linewidth=1.2, linestyle=":", label="Fixed α", alpha=0.8)
    ax1.step(x, df["alpha_hard"], color="#F44336", linewidth=1.5, linestyle="--", label="Hard Gate α", where="mid")
    ax1.plot(x, df["alpha_sigmoid"], color="#2196F3", linewidth=2.0, marker="o", markersize=3, label="Sigmoid α")
    ax1.set_ylabel("Applied α", fontsize=10)
    ax1.set_title("Applied α per Token (Sigmoid vs Hard Gate vs Fixed)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)

    # --- Plot 2: IC値の棒グラフ ---
    ax2 = fig.add_subplot(gs[1])
    colors = ["#EF5350" if ic >= ic_theta else "#90CAF9" for ic in df["ic"]]
    ax2.bar(x, df["ic"], color=colors, alpha=0.85)
    ax2.axhline(ic_theta, color="orange", linestyle="--", linewidth=1.5, label=f"θ={ic_theta} bits")
    ax2.set_ylabel("IC = -log₂ P (bits)", fontsize=10)
    ax2.set_title("Information Content per Token (Red = High IC, above θ)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)

    # --- Plot 3: 散布図 IC vs α_sigmoid ---
    ax3 = fig.add_subplot(gs[2])
    ic_sorted = np.linspace(0, df["ic"].max() * 1.1, 100)
    ax3.scatter(df["ic"], df["alpha_sigmoid"], color="#2196F3", alpha=0.7, s=30, zorder=5, label="Tokens")
    ax3.plot(
        ic_sorted,
        [sigmoid_alpha(ic, df["alpha_sigmoid"].max() / df["alpha_sigmoid"].max() * df["alpha_sigmoid"].max()
                       if df["alpha_sigmoid"].max() > 0 else 1.0,
                       1.0, ic_theta) for ic in ic_sorted],
        color="gray", linewidth=1.5, linestyle="--", label="Sigmoid curve (reference)"
    )
    ax3.axvline(ic_theta, color="orange", linestyle="--", linewidth=1.2, label=f"θ={ic_theta}")
    ax3.set_xlabel("IC (bits)", fontsize=10)
    ax3.set_ylabel("α (Sigmoid)", fontsize=10)
    ax3.set_title("IC vs Applied α scatter", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    plt.suptitle(f"IC-Sigmoid Token Trace: {title_suffix}", fontsize=13, y=1.01)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True)
    ap.add_argument("--axis", type=str, default="neuroticism")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high")
    ap.add_argument("--target_layer", type=int, required=True)
    ap.add_argument("--alpha_max", type=float, default=20.0)
    ap.add_argument("--ic_theta", type=float, default=3.0)
    ap.add_argument("--ic_k", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--out_dir", default="exp_ic_sigmoid_alpha/figures/debug")
    ap.add_argument("--prompt", type=str,
                    default="Tell me how you feel about your life right now.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # シグモイド曲線だけなら GPU なしで描画可能
    suffix = f"{args.axis}_L{args.target_layer}_amax{args.alpha_max}_theta{args.ic_theta}_k{args.ic_k}"
    plot_sigmoid_curve(
        alpha_max=args.alpha_max, ic_k=args.ic_k, ic_theta=args.ic_theta,
        out_path=out_dir / f"sigmoid_curve_{suffix}.png"
    )

    # モデルが必要な場合のみロード
    print("\nLoading model for token-level trace...")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    v_data = np.load(args.vector_bank)
    w_key = f"{args.target_layer}|{args.axis}|w"
    if w_key not in v_data:
        print(f"[ERROR] Key not found: {w_key}")
        return
    direction_mult = 1.0 if args.direction == "high" else -1.0
    w = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult

    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()
    w_dev = w.to(device)

    print(f"Generating token trace for prompt: \"{args.prompt[:60]}...\"")
    text, df = generate_with_trace(
        model, tokenizer, args.prompt, w_dev,
        target_layer=args.target_layer,
        alpha_max=args.alpha_max,
        ic_theta=args.ic_theta,
        ic_k=args.ic_k,
        max_new_tokens=args.max_new_tokens,
    )

    # CSV保存
    csv_path = out_dir / f"trace_{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved trace CSV: {csv_path}")

    # トレースプロット
    plot_token_trace(df, out_dir / f"trace_{suffix}.png", suffix, args.ic_theta)

    print(f"\nGenerated text:\n{text}")

    # 統計サマリー
    high_ic_ratio = (df["ic"] >= args.ic_theta).mean() * 100
    zero_alpha_ratio = (df["alpha_sigmoid"] < 0.01).mean() * 100
    print(f"\n--- Token Statistics ---")
    print(f"  High-IC tokens (>= θ={args.ic_theta}) : {high_ic_ratio:.1f}%")
    print(f"  Zero-alpha tokens (<0.01) [sigmoid]  : {zero_alpha_ratio:.1f}%")
    print(f"  Mean IC: {df['ic'].mean():.2f} bits, Std IC: {df['ic'].std():.2f} bits")
    print(f"  Mean α (sigmoid): {df['alpha_sigmoid'].mean():.3f}")
    print(f"  Mean α (hard)   : {df['alpha_hard'].mean():.3f}")


if __name__ == "__main__":
    main()
