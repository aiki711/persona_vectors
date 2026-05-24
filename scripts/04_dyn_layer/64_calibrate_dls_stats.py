#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 64_calibrate_dls_stats.py
#
# キャリブレーション用スクリプト：
# 複数のプロンプトを用いて、各層・各特性・各手法におけるスコアの平均と標準偏差を算出する。
# この統計量を 63_run_dyn_layer_zscore.py で読み込むことで、層ごとのバイアスを補正する。
#
# Usage:
#   python scripts/04_dyn_layer/64_calibrate_dls_stats.py \
#     --config config/llama3_8b.yaml \
#     --vector_bank vectors/llama3_8b_persona_vectors.npz \
#     --prompts data/calibration_prompts.jsonl \
#     --out_file data/dls_calibration_stats.json

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_file", default="data/dls_calibration_stats.json")
    ap.add_argument("--alpha", type=float, default=10.0, help="Calibration alpha (typically a mid-range value)")
    ap.add_argument("--num_prompts", type=int, default=50)
    ap.add_argument("--norm_mode", type=str, choices=["none", "midpoint"], default="none")
    ap.add_argument("--layers", type=str, default="", help="Comma-separated list of layers to calibrate")
    args = ap.parse_args()

    global LAYERS
    if args.layers:
        LAYERS = [int(x.strip()) for x in args.layers.split(",")]
        print(f"Calibrating layers: {LAYERS}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Load prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: item = json.loads(line)
            except: item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                prompts.append(item["input"])
            elif isinstance(item, str):
                prompts.append(item)
    
    prompts = prompts[:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts for calibration.")

    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    v_data = np.load(args.vector_bank)
    stack, _, _ = get_layer_stack(model)

    # Structure: stats[trait][method][layer] = [scores...]
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for p_text in tqdm(prompts, desc="Calibrating"):
        inputs = format_and_tokenize(tokenizer, p_text, device)
        
        # Pre-capture hidden states for anti_alignment
        saved_h = {}
        handles = []
        def get_hook(L):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                saved_h[L] = hs[0, -1, :].detach().float()
            return hook
        for L in LAYERS:
            handles.append(stack[L].register_forward_hook(get_hook(L)))
        
        with torch.no_grad():
            _ = model(inputs.input_ids)
            base_logits = model(inputs.input_ids).logits[0, -1, :].float()
        
        for h in handles: h.remove()

        for axis in TRAITS:
            # Prepare vector
            w_key = f"{12}|{axis}|w" # Just to check if axis exists, we need vectors for all layers
            if w_key not in v_data: continue

            for L in LAYERS:
                w_vec = torch.tensor(v_data[f"{L}|{axis}|w"], device=device, dtype=torch.float32)
                
                if args.norm_mode == "midpoint":
                    mp_key = f"{L}|{axis}|midpoint"
                    if mp_key in v_data:
                        m_vec = torch.tensor(v_data[mp_key], device=device, dtype=torch.float32)
                        w_norm = torch.norm(w_vec).item()
                        m_norm = torch.norm(m_vec).item()
                        w_vec = (w_vec / (w_norm + 1e-10)) * m_norm

                # 1. logit_diff score
                steered_logits = get_steered_logits(model, inputs.input_ids, L, w_vec, args.alpha)
                l_diff = (steered_logits - base_logits).norm().item()
                raw_data[axis]["logit_diff"][L].append(l_diff)

                # 2. anti_alignment score (we use 'high' direction for calibration)
                h = saved_h[L]
                cos_sim = F.cosine_similarity(h.unsqueeze(0), w_vec.unsqueeze(0)).item()
                raw_data[axis]["anti_alignment"][L].append(-cos_sim)

    # Calculate mean and std
    final_stats = defaultdict(lambda: defaultdict(dict))
    for axis in raw_data:
        for method in raw_data[axis]:
            for L in raw_data[axis][method]:
                scores = raw_data[axis][method][L]
                final_stats[axis][method][str(L)] = {
                    "mean": float(np.mean(scores)),
                    "std":  float(np.std(scores))
                }

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nCalibration complete. Stats saved to: {args.out_file}")

if __name__ == "__main__":
    main()
