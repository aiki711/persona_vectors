#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 69_run_ic_sigmoid_steering.py
#
# IC-Adaptive Steering with Sigmoid Soft-Gating (Phase 2 implementation)
#
# 設計方針（2026-05-20 議論に基づく）:
#   - 介入強度 α を "直前トークンのIC（1ステップラグ近似）" に基づき動的に決定
#   - スケーリング関数: シグモイドソフトゲーティング（案B）
#       α_t = α_max * sigmoid(k * (IC_{t-1} - θ))
#   - ハードゲート（案C）もモード切り替えでサポート（ベースライン比較用）
#   - DLSとは独立した固定層で動作（Phase 2: 単体検証フェーズ）
#
# 既存の50_run_ic_adaptive_steering.py との差分:
#   - τ（マージン）ベースの介入量算出を廃止 → 純粋な固定α_max×シグモイドゲーティング
#   - max_prob ベースの surprisal → 実際に選ばれた前トークンのICを使用（論理的に正確）
#   - ハイパーパラメータ (k, θ, α_max) を全てCLI引数で制御可能
#   - 各トークンの IC・α_t のトレースログを JSONL に保存（デバッグ・解析用）
#
# Usage:
#   python scripts/04_dyn_layer/69_run_ic_sigmoid_steering.py \
#     --config config/mistral_7b.yaml \
#     --vector_bank vectors/mean_diff_vectors.npz \
#     --prompts inputs/test_prompts_10.jsonl \
#     --out_dir exp_ic_sigmoid_alpha/results \
#     --axis neuroticism \
#     --target_layer 15 \
#     --alpha_max 20.0 \
#     --ic_theta 3.0 \
#     --ic_k 0.8 \
#     --ic_mode sigmoid
#

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

# ==================== IC スケーリング関数 ====================

def sigmoid_alpha(ic: float, alpha_max: float, k: float, theta: float) -> float:
    """
    案B: シグモイドソフトゲーティング
    α_t = α_max * σ(k * (IC - θ))
    IC が θ より十分小さい → α ≈ 0  (機能語・低エントロピー領域)
    IC が θ より十分大きい → α ≈ α_max (内容語・高エントロピー領域、サチュレーション)
    ※ 超高IC（固有名詞・稀少語）は抑制されない点に注意
    """
    import math
    x = k * (ic - theta)
    sig = 1.0 / (1.0 + math.exp(-x))
    return alpha_max * sig


def hard_gate_alpha(ic: float, alpha_max: float, theta: float) -> float:
    """
    案C: パーセンタイルハードゲート（比較ベースライン用）
    IC >= θ なら α_max、そうでなければ 0
    """
    return alpha_max if ic >= theta else 0.0


def soft_plateau_alpha(
    ic: float,
    alpha_max: float,
    theta_lo: float,
    theta_hi: float,
    k_lo: float = 1.0,
    k_hi: float = 1.0,
) -> float:
    """
    案D: 非対称ソフトプラトー（Asymmetric Soft-Plateau Gate）

    α_t = α_max * σ(k_lo * (IC - θ_lo)) * σ(-k_hi * (IC - θ_hi))

    低IC（機能語、 IC < θ_lo）: 左側シグモイドにより α ≈ 0 に抑制
    中高IC（内容語、 θ_lo <= IC <= θ_hi）: プラトー領域で α ≈ α_max
    超高IC（固有名詞・稀少語、 IC > θ_hi）: 右側シグモイドにより α → 0 に減衰

    → 案B（片側シグモイド）では対処できなかった「稀少語・固有名詞への過剰介入」問題を解決。
       さらに、k_hi を大きくすることで案C（ハードゲート）に近似する連続性も持つ。

    引数:
        theta_lo: 低IC側の変曲点（例: 2.5～3.5 bit: 機能語境界）
        theta_hi: 高IC側の変曲点（例: 10.0～12.0 bit: 固有名詞・稀少語境界）
        k_lo:     左側の傾き（大きいほどシャープな立ち上がり）
        k_hi:     右側の傾き（大きいほどシャープな切り落とし）
    """
    import math
    left  = 1.0 / (1.0 + math.exp(-k_lo * (ic - theta_lo)))
    right = 1.0 / (1.0 + math.exp( k_hi * (ic - theta_hi)))
    return alpha_max * left * right


# ==================== IC の計算 ====================

def compute_ic_of_token(logits: torch.Tensor, selected_token_id: int) -> float:
    """
    与えられたロジット分布において、selected_token_id が選ばれた場合のICを計算
    IC = -log2 P(x_t | x_{<t})
    logits: shape [vocab_size]
    """
    probs = torch.softmax(logits.float(), dim=-1)
    prob = probs[selected_token_id].clamp(min=1e-10).item()
    return -np.log2(prob)


# ==================== ステアリングフック付き生成ループ ====================

def generate_with_ic_sigmoid(
    model, tokenizer, prompt, w_dev,
    target_layer,
    alpha_max, ic_theta, ic_k,
    ic_mode="sigmoid",
    max_new_tokens=150,
    temperature=0.7,
    repetition_penalty=1.1,
    **kwargs,
):
    """
    kwargs:
        ic_theta_hi (float): soft_plateau モード時の高IC側変曲点（デフォルト 11.0 bit）
        ic_k_hi     (float): soft_plateau モード時の高IC側傾き（デフォルト 1.0）

    ICソフトゲーティング付き生成ループ。
    1ステップラグ近似:
      ステップ t でのα_t は、ステップ t-1 で実際に選択されたトークンのIC から計算する。
    これにより追加のフォワードパスを一切必要とせず1パスで実装できる。
    """
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    stack, _, _ = get_layer_stack(model)

    gen_ids = inputs.input_ids
    prompt_len = gen_ids.shape[1]

    prev_ic = 0.0          # 最初のステップのICはゼロ（デフォルト: 介入なし）
    alpha_t = 0.0
    token_trace = []       # デバッグ・解析用トレースログ

    # 繰り返しペナルティのためのトークン頻度カウンタ
    token_freq = {}

    for step in range(max_new_tokens):
        # --- ステップ t の α_t を直前ICから決定 ---
        if ic_mode == "sigmoid":
            alpha_t = sigmoid_alpha(prev_ic, alpha_max, ic_k, ic_theta)
        elif ic_mode == "hard_gate":
            alpha_t = hard_gate_alpha(prev_ic, alpha_max, ic_theta)
        elif ic_mode == "soft_plateau":
            # kwargs から theta_hi, k_lo, k_hi を取得
            alpha_t = soft_plateau_alpha(
                prev_ic, alpha_max,
                theta_lo=ic_theta,
                theta_hi=kwargs.get("ic_theta_hi", 11.0),
                k_lo=ic_k,
                k_hi=kwargs.get("ic_k_hi", 1.0),
            )
        elif ic_mode == "fixed":
            alpha_t = alpha_max
        else:
            raise ValueError(f"Unknown ic_mode: {ic_mode}")

        # --- フック登録（alpha_t をクロージャでキャプチャ）---
        _current_alpha = alpha_t  # 確実に現在値をキャプチャ

        def hook(mod, inp, out, _alpha=_current_alpha):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all():
                return out
            if hs.size(1) != 1:  # 生成フェーズのみ（prefill はスキップ）
                return out
            hs_f32 = hs.to(torch.float32)
            steered = hs_f32 + _alpha * w_dev.view(1, 1, -1)
            if not torch.isfinite(steered).all():
                return out
            steered = steered.to(hs.dtype)
            return (steered, *out[1:]) if isinstance(out, tuple) else steered

        handle = stack[target_layer].register_forward_hook(hook)

        try:
            with torch.no_grad():
                outputs = model(gen_ids)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        finally:
            handle.remove()

        # --- 繰り返しペナルティ適用 ---
        logits_f32 = logits.float().clone()
        for tok_id, freq in token_freq.items():
            if logits_f32[0, tok_id] > 0:
                logits_f32[0, tok_id] /= (repetition_penalty ** freq)
            else:
                logits_f32[0, tok_id] *= (repetition_penalty ** freq)

        # --- 温度スケーリング & サンプリング ---
        logits_scaled = logits_f32 / temperature
        probs = F.softmax(logits_scaled, dim=-1)
        next_token = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)
        selected_id = next_token.item()

        # --- 次ステップ用IC算出（1ステップラグ近似）---
        prev_ic = compute_ic_of_token(logits_f32[0], selected_id)

        # トレースログ記録
        token_str = tokenizer.decode([selected_id], skip_special_tokens=False)
        token_trace.append({
            "step": step,
            "token": token_str,
            "token_id": selected_id,
            "ic": prev_ic,
            "alpha_t": alpha_t,
        })

        # 頻度カウント更新
        token_freq[selected_id] = token_freq.get(selected_id, 0) + 1

        gen_ids = torch.cat([gen_ids, next_token], dim=-1)

        if selected_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    return text, gen_ids[0], token_trace


# ==================== PPL 計算 ====================

@torch.no_grad()
def calc_ppl(model, ids):
    labels = ids.clone()
    out = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
    return torch.exp(out.loss).item()


# ==================== Main ====================

def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                item = json.loads(line)
            except Exception:
                item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
    return prompts


def main():
    ap = argparse.ArgumentParser(
        description="IC Sigmoid Steering: Dynamic alpha based on Information Content of previous token"
    )
    # モデル・データ設定
    ap.add_argument("--config", "-c", required=True, help="YAML config (model_name, quant etc.)")
    ap.add_argument("--vector_bank", required=True, help=".npz vector bank")
    ap.add_argument("--prompts", required=True, help="Prompts JSONL")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--axis", type=str, default="extraversion", help="Personality trait axis")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high",
                    help="Steering direction (high/low)")
    ap.add_argument("--target_layer", type=int, required=True,
                    help="Layer to steer (fixed layer from Phase 1 analysis)")
    # ICスケーリングパラメータ
    ap.add_argument("--alpha_max", type=float, default=20.0,
                    help="Maximum steering alpha (saturation value in sigmoid)")
    ap.add_argument("--ic_theta", type=float, default=3.0,
                    help="IC threshold / sigmoid inflection point (bits)")
    ap.add_argument("--ic_k", type=float, default=0.8,
                    help="Sigmoid steepness parameter k")
    ap.add_argument("--ic_mode", type=str,
                    choices=["sigmoid", "hard_gate", "soft_plateau", "fixed"],
                    default="sigmoid",
                    help="Alpha scaling mode: sigmoid (B) / hard_gate (C) / soft_plateau (D) / fixed (baseline)")
    # soft_plateau 専用パラメータ
    ap.add_argument("--ic_theta_hi", type=float, default=11.0,
                    help="[soft_plateau] High-IC cutoff inflection point (bits). Default 11.0")
    ap.add_argument("--ic_k_hi", type=float, default=1.0,
                    help="[soft_plateau] Sigmoid steepness for high-IC cutoff. Default 1.0")
    # 生成設定
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num_prompts", type=int, default=10)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"ic_{args.ic_mode}_L{args.target_layer}_amax{args.alpha_max}_theta{args.ic_theta}_k{args.ic_k}.jsonl"
    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # ベクトルのロード
    v_data = np.load(args.vector_bank)
    w_key = f"{args.target_layer}|{args.axis}|w"
    if w_key not in v_data:
        print(f"[ERROR] Key not found: {w_key}")
        return
    direction_mult = 1.0 if args.direction == "high" else -1.0
    w = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult

    # プロンプトのロード
    prompts = load_prompts(args.prompts)[:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts")

    # モデルのロード
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()
    w_dev = w.to(device)

    print(f"\n=== IC Sigmoid Steering ===")
    print(f"  Axis        : {args.axis}")
    print(f"  Layer       : {args.target_layer}")
    print(f"  IC Mode     : {args.ic_mode}")
    print(f"  alpha_max   : {args.alpha_max}")
    print(f"  ic_theta    : {args.ic_theta} bits (low cutoff)")
    print(f"  ic_k        : {args.ic_k}")
    if args.ic_mode == "soft_plateau":
        print(f"  ic_theta_hi : {args.ic_theta_hi} bits (high cutoff)")
        print(f"  ic_k_hi     : {args.ic_k_hi}")

    results = []

    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts, desc="Generating")):
        # ベースライン生成（ステアリングなし）
        with torch.no_grad():
            formatted = _format_prompt(tokenizer, p_text)
            base_inputs = tokenizer(formatted, return_tensors="pt").to(device)
            base_outputs = model.generate(
                **base_inputs, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
        base_plen = base_inputs.input_ids.shape[1]
        base_text = tokenizer.decode(base_outputs[0][base_plen:], skip_special_tokens=True)
        base_ppl = calc_ppl(model, base_outputs[0])

        # ICアダプティブ生成
        ic_text, ic_ids, token_trace = generate_with_ic_sigmoid(
            model, tokenizer, p_text, w_dev,
            target_layer=args.target_layer,
            alpha_max=args.alpha_max,
            ic_theta=args.ic_theta,
            ic_k=args.ic_k,
            ic_mode=args.ic_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            ic_theta_hi=args.ic_theta_hi,
            ic_k_hi=args.ic_k_hi,
        )
        ic_ppl = calc_ppl(model, ic_ids)

        # トレースの統計サマリー
        ics = [t["ic"] for t in token_trace]
        alphas = [t["alpha_t"] for t in token_trace]
        trace_summary = {
            "ic_mean": float(np.mean(ics)) if ics else 0.0,
            "ic_std": float(np.std(ics)) if ics else 0.0,
            "alpha_mean": float(np.mean(alphas)) if alphas else 0.0,
            "alpha_std": float(np.std(alphas)) if alphas else 0.0,
            "zero_alpha_ratio": float(sum(1 for a in alphas if a < 0.01) / len(alphas)) if alphas else 0.0,
        }

        results.append({
            "idx": idx,
            "orig_idx": orig_idx,
            "prompt": p_text,
            "base_text": base_text,
            "base_ppl": base_ppl,
            "ic_text": ic_text,
            "ic_ppl": ic_ppl,
            "ic_mode": args.ic_mode,
            "target_layer": args.target_layer,
            "alpha_max": args.alpha_max,
            "ic_theta": args.ic_theta,
            "ic_k": args.ic_k,
            "ic_theta_hi": args.ic_theta_hi if args.ic_mode == "soft_plateau" else None,
            "ic_k_hi": args.ic_k_hi if args.ic_mode == "soft_plateau" else None,
            "trace_summary": trace_summary,
            # 詳細なトークントレースは大きいのでオプション（分析時はここを有効化）
            # "token_trace": token_trace,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # サマリー出力
    avg_base_ppl = np.mean([r["base_ppl"] for r in results])
    avg_ic_ppl = np.mean([r["ic_ppl"] for r in results])
    avg_alpha = np.mean([r["trace_summary"]["alpha_mean"] for r in results])
    avg_zero_ratio = np.mean([r["trace_summary"]["zero_alpha_ratio"] for r in results])

    print(f"\n--- Summary ---")
    print(f"  Base PPL    : {avg_base_ppl:.2f}")
    print(f"  IC PPL      : {avg_ic_ppl:.2f}  (Delta: {avg_ic_ppl - avg_base_ppl:+.2f})")
    print(f"  Avg alpha   : {avg_alpha:.3f}")
    print(f"  Zero-alpha  : {avg_zero_ratio*100:.1f}% of tokens (suppressed interventions)")
    print(f"  Saved to    : {out_file}")


if __name__ == "__main__":
    main()
