#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 73_run_dyn_ic_fusion.py
#
# 提案手法の最終形態:
#   1. 第一段階 (Prefill段階での動的層選択 - Relative Anti-alignment):
#      プロンプト入力時のPrefill完了（最終トークン）時点の隠れ状態 h から
#      中点 midpoint m を引いた相対ベクトル h - m と、ステアリングベクトル w の
#      コサイン類似度を測定。類似度が最も負に大きく振れている最適な1層（best_layer）を特定し固定。
#   2. 第二段階 (生成段階でのトークンレベル動的重み制御 - 1ステップラグIC適応制御):
#      固定した best_layer に対してのみ、直前トークンの自己情報量（IC）に基づいて
#      動的に制御重み α_t を適用してステアリングベクトルを加算する。
#      重み関数は シグモイド型 もしくは 台形型ソフトゲーティング（非対称ソフトプラトー）をサポート。
#
# Outputs:
#   exp_steering_dyn_ic_fusion/results/{trait}/fusion_{ic_mode}_Val{alpha_max}.jsonl
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

LAYERS = list(range(32))  # 全32層を対象

# ==================== Utility ====================

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)


@torch.no_grad()
def calc_ppl(model, ids):
    labels = ids.clone()
    out = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
    return torch.exp(out.loss).item()


def compute_ic_of_token(logits: torch.Tensor, selected_token_id: int) -> float:
    """
    logits: shape [vocab_size]
    IC = -log2 P(x_t | x_{<t})
    """
    probs = torch.softmax(logits.float(), dim=-1)
    prob = probs[selected_token_id].clamp(min=1e-10).item()
    return -np.log2(prob)


# ==================== IC スケーリング関数 ====================

def sigmoid_alpha(ic: float, alpha_max: float, k: float, theta: float) -> float:
    """
    シグモイドソフトゲーティング
    α_t = α_max * σ(k * (IC - θ))
    """
    import math
    x = k * (ic - theta)
    sig = 1.0 / (1.0 + math.exp(-x))
    return alpha_max * sig


def soft_plateau_alpha(
    ic: float,
    alpha_max: float,
    theta_lo: float,
    theta_hi: float,
    k_lo: float = 1.0,
    k_hi: float = 1.0,
) -> float:
    """
    台形型ソフトゲーティング関数（非対称ソフトプラトー）
    α_t = α_max * σ(k_lo * (IC - θ_lo)) * σ(-k_hi * (IC - θ_hi))
    低IC（機能語）と超高IC（固有名詞・稀少語）を抑制し、内容語（中〜高IC）のみに介入。
    """
    import math
    left  = 1.0 / (1.0 + math.exp(-k_lo * (ic - theta_lo)))
    right = 1.0 / (1.0 + math.exp( k_hi * (ic - theta_hi)))
    return alpha_max * left * right


# ==================== Prefill: Dynamic Layer Selection (Relative Anti-alignment) ====================

def select_layer_relative_anti_alignment(model, input_ids, layer_w_dev, layer_midpoint_dev, target_direction):
    """
    Prefill完了段階における相対アンチアライメントに基づいて最適な1層を選択する。
    """
    saved_h = {}
    handles = []
    stack, _, _ = get_layer_stack(model)

    def get_hook(L):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            saved_h[L] = hs[0, -1, :].detach().float()
        return hook

    for L in layer_w_dev.keys():
        handles.append(stack[L].register_forward_hook(get_hook(L)))

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for h in handles:
            h.remove()

    scores = {}
    for L, w_dev in layer_w_dev.items():
        h = saved_h[L]
        m = layer_midpoint_dev.get(L, None)

        if m is not None:
            # 中点基準の相対ベクトル (h - m)
            h_rel = h - m
        else:
            h_rel = h

        cos_sim = F.cosine_similarity(h_rel.unsqueeze(0), w_dev.unsqueeze(0)).item()

        if target_direction == "high":
            # 制御方向と真逆（負例側）へ引きずられている層を選ぶ → -cos_sim を最大化
            scores[L] = -cos_sim
        else:
            scores[L] = cos_sim

    best_layer = max(scores, key=lambda L: scores[L])
    return best_layer, scores


# ==================== Generation: DLS + IC-Adaptive Fusion ====================

def generate_fusion(
    model, tokenizer, prompt,
    best_layer, w_dev,
    alpha_max, ic_theta, ic_k,
    ic_mode="soft_plateau",
    max_new_tokens=150,
    temperature=0.7,
    repetition_penalty=1.1,
    **kwargs
):
    """
    決定された best_layer のみに対して、直前ICに応じた動的重み α_t で介入を行い生成する。
    """
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    stack, _, _ = get_layer_stack(model)

    gen_ids = inputs.input_ids
    prompt_len = gen_ids.shape[1]

    prev_ic = 0.0          # 初回ステップのICは0（介入なし）
    alpha_t = 0.0
    alphas_so_far = []     # Track the alpha values used for each generated token
    token_trace = []
    token_freq = {}

    for step in range(max_new_tokens):
        # --- ステップ t の α_t を直前ICから決定 ---
        if ic_mode == "sigmoid":
            alpha_t = sigmoid_alpha(prev_ic, alpha_max, ic_k, ic_theta)
        elif ic_mode == "soft_plateau":
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

        # --- best_layer にフックを登録 ---
        _current_alpha = alpha_t
        _alphas_past = list(alphas_so_far)

        def hook(mod, inp, out, _alpha=_current_alpha, _alphas_past=_alphas_past):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all():
                return out
            seq_len = hs.size(1)
            if seq_len <= prompt_len:
                return out
            hs_f32 = hs.to(torch.float32)
            steered = hs_f32.clone()
            for idx in range(prompt_len, seq_len):
                gen_idx = idx - prompt_len
                a = _alphas_past[gen_idx] if gen_idx < len(_alphas_past) else _alpha
                steered[:, idx, :] = hs_f32[:, idx, :] + a * w_dev.view(1, 1, -1)
            if not torch.isfinite(steered).all():
                return out
            return (steered.to(hs.dtype), *out[1:]) if isinstance(out, tuple) else steered.to(hs.dtype)

        handle = stack[best_layer].register_forward_hook(hook)

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

        # --- サンプリング ---
        logits_scaled = logits_f32 / temperature
        probs = F.softmax(logits_scaled, dim=-1)
        next_token = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)
        selected_id = next_token.item()

        # --- 次ステップ用IC算出（1ステップラグ）---
        prev_ic = compute_ic_of_token(logits_f32[0], selected_id)

        token_str = tokenizer.decode([selected_id], skip_special_tokens=False)
        token_trace.append({
            "step": step,
            "token": token_str,
            "token_id": selected_id,
            "ic": prev_ic,
            "alpha_t": alpha_t,
        })

        token_freq[selected_id] = token_freq.get(selected_id, 0) + 1
        gen_ids = torch.cat([gen_ids, next_token], dim=-1)

        # Record the alpha_t used for this step's generation.
        if step >= 1:
            alphas_so_far.append(alpha_t)

        if selected_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    return text, gen_ids[0], token_trace


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
    ap = argparse.ArgumentParser(description="DLS + IC-Adaptive Fusion Steering")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True, help="mean_diff_vectors.npz (containing midpoint)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--axis", type=str, default="extraversion")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high")
    
    # ICスケーリング
    ap.add_argument("--alpha_max", type=float, default=20.0)
    ap.add_argument("--ic_theta", type=float, default=3.0, help="Low IC cutoff theta_lo (bits)")
    ap.add_argument("--ic_k", type=float, default=0.8, help="Low IC transition k_lo")
    ap.add_argument("--ic_mode", type=str, choices=["sigmoid", "soft_plateau", "fixed"], default="soft_plateau")
    
    # soft_plateau 専用
    ap.add_argument("--ic_theta_hi", type=float, default=11.0, help="High IC cutoff theta_hi (bits)")
    ap.add_argument("--ic_k_hi", type=float, default=1.0, help="High IC transition k_hi")
    
    # レイヤー探索範囲
    ap.add_argument("--layers", type=str, default="", help="Comma-separated layers for DLS search")
    
    # 生成設定
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num_prompts", type=int, default=10)
    ap.add_argument("--norm_mode", type=str, choices=["none", "midpoint"], default="none", help="Normalization mode for steering vectors")
    args = ap.parse_args()

    global LAYERS
    if args.layers:
        LAYERS = [int(x.strip()) for x in args.layers.split(",")]

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"fusion_{args.ic_mode}_Val{args.alpha_max}.jsonl"
    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # ベクトルのロード
    v_data = np.load(args.vector_bank)
    direction_mult = 1.0 if args.direction == "high" else -1.0
    
    layer_w = {}
    layer_midpoint = {}
    for L in LAYERS:
        w_key  = f"{L}|{args.axis}|w"
        mp_key = f"{L}|{args.axis}|midpoint"
        if w_key in v_data:
            layer_w[L] = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult
        if mp_key in v_data:
            layer_midpoint[L] = torch.tensor(v_data[mp_key], dtype=torch.float32)
            
        if args.norm_mode == "midpoint" and L in layer_w and L in layer_midpoint:
            w_norm = torch.norm(layer_w[L]).item()
            m_norm = torch.norm(layer_midpoint[L]).item()
            layer_w[L] = (layer_w[L] / (w_norm + 1e-10)) * m_norm

    if not layer_w:
        print("[ERROR] No layer vectors found.")
        return

    # プロンプトのロード
    prompts = load_prompts(args.prompts)[:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts")

    # モデルのロード
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    layer_w_dev = {L: w.to(device) for L, w in layer_w.items()}
    layer_midpoint_dev = {L: m.to(device) for L, m in layer_midpoint.items()}

    print(f"\n=== DLS + IC-Adaptive Fusion ===")
    print(f"  Axis        : {args.axis}")
    print(f"  DLS Search  : {len(layer_w_dev)} layers")
    print(f"  IC Mode     : {args.ic_mode}")
    print(f"  alpha_max   : {args.alpha_max}")
    print(f"  ic_theta_lo : {args.ic_theta} bits")
    print(f"  ic_k_lo     : {args.ic_k}")
    if args.ic_mode == "soft_plateau":
        print(f"  ic_theta_hi : {args.ic_theta_hi} bits")
        print(f"  ic_k_hi     : {args.ic_k_hi}")

    results = []

    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts, desc="Generating")):
        inputs = format_and_tokenize(tokenizer, p_text, device)
        
        # 1. Baseline generation
        with torch.no_grad():
            base_outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
        base_plen = inputs.input_ids.shape[1]
        base_text = tokenizer.decode(base_outputs[0][base_plen:], skip_special_tokens=True)
        base_ppl  = calc_ppl(model, base_outputs[0])

        # 2. DLS (Prefill完了段階で最適な1層を選択)
        best_layer, dls_scores = select_layer_relative_anti_alignment(
            model, inputs.input_ids,
            layer_w_dev, layer_midpoint_dev,
            args.direction
        )

        # 3. Fusion Generation (best_layer に対してのみ IC 適応制御)
        fusion_text, fusion_ids, token_trace = generate_fusion(
            model, tokenizer, p_text,
            best_layer=best_layer,
            w_dev=layer_w_dev[best_layer],
            alpha_max=args.alpha_max,
            ic_theta=args.ic_theta,
            ic_k=args.ic_k,
            ic_mode=args.ic_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            ic_theta_hi=args.ic_theta_hi,
            ic_k_hi=args.ic_k_hi,
        )
        fusion_ppl = calc_ppl(model, fusion_ids)

        # 統計サマリー
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
            "fusion_text": fusion_text,
            "fusion_ppl": fusion_ppl,
            "best_layer": best_layer,
            "dls_scores": {str(L): float(v) for L, v in dls_scores.items()},
            "ic_mode": args.ic_mode,
            "alpha_max": args.alpha_max,
            "ic_theta": args.ic_theta,
            "ic_k": args.ic_k,
            "ic_theta_hi": args.ic_theta_hi if args.ic_mode == "soft_plateau" else None,
            "ic_k_hi": args.ic_k_hi if args.ic_mode == "soft_plateau" else None,
            "trace_summary": trace_summary,
            "token_trace": token_trace,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_base_ppl = np.mean([r["base_ppl"] for r in results])
    avg_fusion_ppl = np.mean([r["fusion_ppl"] for r in results])
    avg_alpha = np.mean([r["trace_summary"]["alpha_mean"] for r in results])
    avg_zero_ratio = np.mean([r["trace_summary"]["zero_alpha_ratio"] for r in results])

    print(f"\n--- Fusion Summary ---")
    print(f"  Base PPL    : {avg_base_ppl:.2f}")
    print(f"  Fusion PPL  : {avg_fusion_ppl:.2f}  (Delta: {avg_fusion_ppl - avg_base_ppl:+.2f})")
    print(f"  Avg alpha   : {avg_alpha:.3f}")
    print(f"  Zero-alpha  : {avg_zero_ratio*100:.1f}% of tokens (suppressed)")
    print(f"  Saved to    : {out_file}")


if __name__ == "__main__":
    main()
