#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 65_run_dyn_layer_relative.py
#
# 改良版 Dynamic Layer Selection:
#   - relative_anti_alignment: 各層の隠れ状態 h から、その層の
#     正負ベクトルの中心（midpoint）を引いた相対ベクトル (h - m) と
#     制御方向 w との Cosine Similarity に基づいて介入層を選択する。
#
#     スコア（target_direction="high"の場合）:
#       score[l] = -cosine_similarity(h[l] - midpoint[l], w_norm[l])
#     →スコアが最大（= midpoint 基準で最も制御方向と逆を向いている）層に介入。
#
# Output:
#   exp_steering_dyn_layer_relative/results/{trait}/relative_anti_alignment_Val{alpha}.jsonl
#

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


@torch.no_grad()
def calc_ppl(model, ids):
    out = model(ids.unsqueeze(0), labels=ids.clone().unsqueeze(0))
    return torch.exp(out.loss).item()


# ==================== Layer Selection: Relative Anti-Alignment ====================

def select_layer_relative_anti_alignment(model, input_ids, layer_w_dev, layer_midpoint_dev, target_direction):
    """
    改良版提案手法:
    各層 l の base hidden state h[l] から、その層の midpoint m[l] を引いた
    相対ベクトル (h[l] - m[l]) と 制御ベクトル w_norm[l] のコサイン類似度を測る。

    target_direction == "high": モデルがネガティブ側（中心より低い側）にある層で介入。
        score[l] = -cosine_similarity(h[l] - m[l], w_norm[l])
    target_direction == "low":  モデルがポジティブ側にある層で介入。
        score[l] = +cosine_similarity(h[l] - m[l], w_norm[l])
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
            # 中心基準の相対ベクトル
            h_rel = h - m
        else:
            # midpoint がなければフォールバック（原点基準）
            h_rel = h

        # Cosine Similarity: (h - m) と w_norm の角度
        cos_sim = F.cosine_similarity(h_rel.unsqueeze(0), w_dev.unsqueeze(0)).item()

        if target_direction == "high":
            # ネガティブ側にある（中心より低い側）層に介入 → コサインが最も負の層を選ぶ
            scores[L] = -cos_sim
        else:
            # ポジティブ側にある層に介入
            scores[L] = cos_sim

    best_layer = max(scores, key=lambda L: scores[L])
    return best_layer, scores


# ==================== Generation ====================

def generate_with_steered_layer(model, tokenizer, prompt, w_dev, alpha, layer, max_new_tokens=150):
    device = _infer_main_device(model)
    inputs = format_and_tokenize(tokenizer, prompt, device)
    stack, _, _ = get_layer_stack(model)

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not torch.isfinite(hs).all() or hs.size(1) != 1:
            return out
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


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", required=True)
    ap.add_argument("--vector_bank",  required=True,
                    help="mean_diff_vectors.npz (must contain |midpoint| keys)")
    ap.add_argument("--prompts",      required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--axis",         type=str, default="extraversion")
    ap.add_argument("--alpha",        type=float, required=True)
    ap.add_argument("--direction",    type=str, choices=["high", "low"], default="high")
    ap.add_argument("--layers",       type=str, default="",
                    help="Comma-separated list of layers to restrict DLS search space")
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

    method_name = "relative_anti_alignment"
    out_file = out_dir / f"{method_name}_Val{args.alpha}.jsonl"
    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # Load vectors and midpoints
    v_data = np.load(args.vector_bank)
    layer_w = {}
    layer_midpoint = {}
    for L in LAYERS:
        w_key  = f"{L}|{args.axis}|w"
        mp_key = f"{L}|{args.axis}|midpoint"
        if w_key in v_data:
            layer_w[L] = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult
        if mp_key in v_data:
            layer_midpoint[L] = torch.tensor(v_data[mp_key], dtype=torch.float32)
        else:
            # midpoint がない場合は警告（古いベクトルバンク）
            if w_key in v_data:
                print(f"  [WARNING] midpoint not found for Layer {L}. Falling back to origin-based cosine.")

    if not layer_w:
        return print("[ERROR] No layer vectors found.")

    missing_mp = [L for L in layer_w if L not in layer_midpoint]
    if missing_mp:
        print(f"  [WARNING] midpoint missing for layers: {missing_mp}")
        print("  → Please regenerate vector bank with 30b_train_mean_diff.py (updated version).")

    # Load prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                item = json.loads(line)
            except:
                item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
    prompts = prompts[:10]

    print(f"=== Relative Anti-Alignment DLS ===")
    print(f"  Axis     : {args.axis}")
    print(f"  Alpha    : {args.alpha}")
    print(f"  Direction: {args.direction}")
    print(f"  midpoints: {len(layer_midpoint)}/{len(layer_w)} layers have midpoint")

    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    layer_w_dev = {L: w.to(device) for L, w in layer_w.items()}
    layer_midpoint_dev = {L: m.to(device) for L, m in layer_midpoint.items()}

    results = []
    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts)):
        inputs = format_and_tokenize(tokenizer, p_text, device)

        # Baseline generation
        with torch.no_grad():
            base_outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
        prompt_len = inputs.input_ids.shape[1]
        base_text = tokenizer.decode(base_outputs[0][prompt_len:], skip_special_tokens=True)
        base_ppl  = calc_ppl(model, base_outputs[0])

        # Layer Selection: relative anti-alignment
        best_layer, scores = select_layer_relative_anti_alignment(
            model, inputs.input_ids,
            layer_w_dev, layer_midpoint_dev,
            args.direction
        )

        # Steered generation
        dyn_text, dyn_ids = generate_with_steered_layer(
            model, tokenizer, p_text, layer_w_dev[best_layer], args.alpha, best_layer
        )
        dyn_ppl = calc_ppl(model, dyn_ids)

        results.append({
            "idx": idx, "orig_idx": orig_idx, "prompt": p_text,
            "base_text": base_text, "base_ppl": base_ppl,
            "dyn_text": dyn_text,  "dyn_ppl": dyn_ppl,
            "dyn_layer": best_layer,
            "scores": {str(L): float(v) for L, v in scores.items()},
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Done] Saved {len(results)} results to {out_file}")


if __name__ == "__main__":
    main()
