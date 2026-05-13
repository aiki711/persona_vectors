#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 63_run_dyn_layer_zscore.py
#
# Dynamic Layer Selection Comparison (Z-score Normalized):
#   z-score normalization を適用し、Layer 0 への偏りを補正したバージョン。
#   元の 61_run_dyn_layer_compare.py は変更せず、このスクリプトで独立実験。
#
# 変更点（vs 61_run_dyn_layer_compare.py）:
#   - select_layer_logit_diff: norms を z-score 正規化してから argmax
#   - select_layer_anti_alignment: scores を z-score 正規化してから argmax
#   - 出力 JSONL に z_scores (正規化後のスコア) も記録
#
# Output JSONL per method and alpha:
#   exp_steering_dyn_layer_zscore/results/{trait}/{method}_Val{alpha}.jsonl

from __future__ import annotations

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

LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

# ==================== Utility ====================

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)

def get_base_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
    return out.logits[0, -1, :].float()

def get_steered_logits(model, input_ids, layer, w_dev, alpha):
    stack, _, _ = get_layer_stack(model)
    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not torch.isfinite(hs).all(): return out
        hs_f32 = hs.to(torch.float32)
        steered = hs_f32 + alpha * w_dev.view(1, 1, -1)
        return (steered.to(hs.dtype), *out[1:]) if isinstance(out, tuple) else steered.to(hs.dtype)

    handle = stack[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0, -1, :].float()
    finally:
        handle.remove()
    return logits


def zscore_normalize(score_dict: dict) -> dict:
    """
    各プロンプトのスコア辞書を z-score 正規化して返す。
    標準偏差が 0 に近い場合（全層で同じ値）はそのまま返す。
    """
    vals = np.array(list(score_dict.values()), dtype=float)
    mean = vals.mean()
    std  = vals.std()
    if std < 1e-8:
        return score_dict  # 全層が同値 → 正規化不要（任意の層を選ばせる）
    return {L: (v - mean) / std for L, v in score_dict.items()}


# ==================== Layer Selection Methods (z-score normalized) ====================

def select_layer_logit_diff(model, input_ids, layer_w_dev, alpha):
    """
    Bhandari et al. のロジット差ノルムを z-score 正規化してから argmax。
    Layer 0 の絶対的な大きさによる偏りを補正する。
    """
    base_logits = get_base_logits(model, input_ids)
    raw_norms = {}
    for L, w_dev in layer_w_dev.items():
        steered_logits = get_steered_logits(model, input_ids, L, w_dev, alpha)
        raw_norms[L] = (steered_logits - base_logits).norm().item()

    z_norms = zscore_normalize(raw_norms)
    best_layer = max(z_norms, key=lambda L: z_norms[L])
    return best_layer, raw_norms, z_norms


def select_layer_anti_alignment(model, input_ids, layer_w_dev, target_direction):
    """
    提案手法（コサイン類似度）を z-score 正規化してから argmax。
    各層の -cos_sim を z-score 正規化することで、全層が正/負に偏っている場合でも
    相対的に最も逆向きな層を選べるようにする。
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

    raw_scores = {}
    for L, w_dev in layer_w_dev.items():
        h = saved_h[L]
        cos_sim = F.cosine_similarity(h.unsqueeze(0), w_dev.unsqueeze(0)).item()
        if target_direction == "high":
            raw_scores[L] = -cos_sim   # 最も逆向き（最も負の cos_sim）を選ぶ
        else:
            raw_scores[L] = cos_sim

    z_scores = zscore_normalize(raw_scores)
    best_layer = max(z_scores, key=lambda L: z_scores[L])
    return best_layer, raw_scores, z_scores


# ==================== Generation ====================

def generate_with_steered_layer(model, tokenizer, prompt, w_dev, alpha, layer, max_new_tokens=150):
    device = _infer_main_device(model)
    inputs = format_and_tokenize(tokenizer, prompt, device)
    stack, _, _ = get_layer_stack(model)

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not torch.isfinite(hs).all() or hs.size(1) != 1: return out
        steered = hs.to(torch.float32) + alpha * w_dev.view(1, 1, -1)
        return (steered.to(hs.dtype), *out[1:]) if isinstance(out, tuple) else steered.to(hs.dtype)

    handle = stack[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
    finally:
        handle.remove()

    prompt_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True), outputs[0]


@torch.no_grad()
def calc_ppl(model, ids):
    out = model(ids.unsqueeze(0), labels=ids.clone().unsqueeze(0))
    return torch.exp(out.loss).item()


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", required=True)
    ap.add_argument("--vector_bank",  required=True)
    ap.add_argument("--prompts",      required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--axis",         type=str, default="extraversion")
    ap.add_argument("--alpha",        type=float, required=True)
    ap.add_argument("--direction",    type=str, choices=["high", "low"], default="high")
    ap.add_argument("--method",       type=str, choices=["logit_diff", "anti_alignment"], required=True)
    ap.add_argument("--layers",       type=str, default="",
                    help="カンマ区切りで探索層を制限 (例: 12,15,18,21,24)")
    args = ap.parse_args()

    global LAYERS
    if args.layers:
        LAYERS = [int(x.strip()) for x in args.layers.split(",")]
        print(f"  Restricting search space to layers: {LAYERS}")

    direction_mult = 1.0 if args.direction == "high" else -1.0

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{args.method}_Val{args.alpha}.jsonl"
    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # Load vectors
    v_data = np.load(args.vector_bank)
    layer_w = {}
    for L in LAYERS:
        w_key = f"{L}|{args.axis}|w"
        if w_key in v_data:
            layer_w[L] = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult

    if not layer_w:
        return print("[ERROR] No layer vectors found.")

    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"): continue
            if line.endswith(","): line = line[:-1]
            try: item = json.loads(line)
            except: item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
    prompts = prompts[:10]

    print(f"=== DLS Z-score Compare: {args.method} ===")
    print(f"  Axis  : {args.axis}")
    print(f"  Alpha : {args.alpha}")
    print(f"  Layers: {LAYERS}")

    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    layer_w_dev = {L: w.to(device) for L, w in layer_w.items()}
    results = []

    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts)):
        inputs = format_and_tokenize(tokenizer, p_text, device)

        # Baseline
        with torch.no_grad():
            base_outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
        prompt_len = inputs.input_ids.shape[1]
        base_text = tokenizer.decode(base_outputs[0][prompt_len:], skip_special_tokens=True)
        base_ppl = calc_ppl(model, base_outputs[0])

        # Layer Selection (z-score normalized)
        if args.method == "logit_diff":
            best_layer, raw_scores, z_scores = select_layer_logit_diff(
                model, inputs.input_ids, layer_w_dev, args.alpha
            )
        else:
            best_layer, raw_scores, z_scores = select_layer_anti_alignment(
                model, inputs.input_ids, layer_w_dev, args.direction
            )

        # Generate
        dyn_text, dyn_ids = generate_with_steered_layer(
            model, tokenizer, p_text, layer_w_dev[best_layer], args.alpha, best_layer
        )
        dyn_ppl = calc_ppl(model, dyn_ids)

        results.append({
            "idx": idx, "orig_idx": orig_idx, "prompt": p_text,
            "base_text": base_text, "base_ppl": base_ppl,
            "dyn_text": dyn_text, "dyn_ppl": dyn_ppl,
            "dyn_layer": best_layer,
            "raw_scores":  {str(L): float(v) for L, v in raw_scores.items()},
            "z_scores":    {str(L): float(v) for L, v in z_scores.items()},
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
