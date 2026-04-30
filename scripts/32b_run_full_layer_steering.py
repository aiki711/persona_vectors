#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 32b_run_full_layer_steering.py
#
# Full-Layer Comparison: Adaptive (tau) vs Constant (alpha)
# Using mean_diff vectors and midpoint-based thresholding.
#
# Adaptive rule:
#   d(h) = w_norm · h + b_norm
#   alpha = max(0, tau - d(h))
#
# Constant rule:
#   alpha = fixed value
#

from __future__ import annotations

import argparse
import os
import json
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
import contextlib

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt
)

@dataclass
class AdaptiveSteererPure:
    """
    Adaptive steering without max_alpha clipping.
    Pushes activation until it reaches the margin 'tau'.
    """
    model: torch.nn.Module
    layer: int
    w: torch.Tensor    # Normal vector (H,) - float32
    b: float           # Bias
    tau: float         # Target margin
    answer_only: bool = True

    def __post_init__(self):
        self.handle = None

    def __enter__(self):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all(): return out
            if self.answer_only and hs.size(1) != 1: return out

            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            device = hs.device
            w_dev = self.w.to(device)

            # d(h) = w \cdot h + b
            dot_product = (hs_f32 * w_dev).sum(dim=-1) # (B, 1)
            dist = dot_product + self.b # (B, 1)
            
            # alpha = max(0, tau - dist)
            alpha = torch.clamp((self.tau - dist), min=0.0) # (B, 1)
            
            # Add to hidden states
            add = w_dev.view(1, 1, -1)
            steered_hs_f32 = hs_f32 + alpha.unsqueeze(-1) * add
            
            if not torch.isfinite(steered_hs_f32).all(): return out
            steered_hs = steered_hs_f32.to(orig_dtype)

            if isinstance(out, tuple): return (steered_hs, *out[1:])
            return steered_hs

        stack, _, _ = get_layer_stack(self.model)
        target_mod = stack[self.layer]
        self.handle = target_mod.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

@dataclass
class ConstantSteerer:
    """
    Constant steering (adding fixed alpha * w).
    """
    model: torch.nn.Module
    layer: int
    w: torch.Tensor
    alpha: float
    answer_only: bool = True

    def __post_init__(self):
        self.handle = None

    def __enter__(self):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all(): return out
            if self.answer_only and hs.size(1) != 1: return out

            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            device = hs.device
            w_dev = self.w.to(device)

            # Add to hidden states
            add = w_dev.view(1, 1, -1)
            steered_hs_f32 = hs_f32 + self.alpha * add
            
            if not torch.isfinite(steered_hs_f32).all(): return out
            steered_hs = steered_hs_f32.to(orig_dtype)

            if isinstance(out, tuple): return (steered_hs, *out[1:])
            return steered_hs

        stack, _, _ = get_layer_stack(self.model)
        target_mod = stack[self.layer]
        self.handle = target_mod.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

def generate_text(model, tokenizer, prompt, max_new_tokens=150):
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = outputs[0][prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text, outputs[0]

@torch.no_grad()
def calc_sequence_ppl(model, input_ids):
    labels = input_ids.clone()
    outputs = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
    loss = outputs.loss
    return torch.exp(loss).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True, help="Path to mean_diff_vectors.npz")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--axis", type=str, default="extraversion")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high")
    ap.add_argument("--tau", type=float, default=4.0, help="Target margin for adaptive")
    ap.add_argument("--alpha", type=float, default=4.0, help="Alpha for constant")
    ap.add_argument("--mode", type=str, choices=["adaptive", "constant", "both"], default="both")
    ap.add_argument("--tag", type=str, default=None)
    
    args = ap.parse_args()
    direction_mult = 1.0 if args.direction == "high" else -1.0

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model_name")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Vectors for ALL LAYERS
    v_data = np.load(args.vector_bank)
    layer_w = {}
    layer_b = {}
    # Detect max layers
    available_layers = []
    for k in v_data.keys():
        if k.endswith(f"|{args.axis}|w"):
            available_layers.append(int(k.split("|")[0]))
    layers = sorted(list(set(available_layers)))
    
    print(f"=== 32b_run_full_layer_steering.py ===")
    print(f"  Model  : {model_name}")
    print(f"  Axis   : {args.axis} (Layers: 0-{max(layers)})")
    print(f"  Mode   : {args.mode}")
    print(f"  Params : Tau={args.tau}, Alpha={args.alpha}")

    for l in layers:
        w_key = f"{l}|{args.axis}|w"
        b_key = f"{l}|{args.axis}|b"
        # Multiplying by direction_mult
        layer_w[l] = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult
        layer_b[l] = float(v_data[b_key][0]) * direction_mult
        
    # 2. Load Prompts
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
    prompts = prompts[:10] # Using 10 prompts for investigation as per discussion

    # 3. Load Model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
    model.eval()

    results = []
    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts)):
        res = {"idx": idx, "orig_idx": orig_idx, "prompt": p_text}
        
        # --- Baseline ---
        txt_b, ids_b = generate_text(model, tokenizer, p_text)
        res["base_text"] = txt_b
        res["base_ppl"] = calc_sequence_ppl(model, ids_b)
        
        # --- Constant Steering (if needed) ---
        if args.mode in ["constant", "both"]:
            with contextlib.ExitStack() as stack:
                for l in layers:
                    stack.enter_context(ConstantSteerer(model, l, layer_w[l], args.alpha, answer_only=True))
                txt_c, ids_c = generate_text(model, tokenizer, p_text)
                res["const_text"] = txt_c
                res["const_ppl"] = calc_sequence_ppl(model, ids_c)
        
        # --- Adaptive Steering (if needed) ---
        if args.mode in ["adaptive", "both"]:
            with contextlib.ExitStack() as stack:
                for l in layers:
                    stack.enter_context(AdaptiveSteererPure(model, l, layer_w[l], layer_b[l], tau=args.tau, answer_only=True))
                txt_a, ids_a = generate_text(model, tokenizer, p_text)
                res["adapt_text"] = txt_a
                res["adapt_ppl"] = calc_sequence_ppl(model, ids_a)
        
        results.append(res)

    # Save
    tag_str = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"investigation_{args.axis}_{args.direction}_T{args.tau}_A{args.alpha}{tag_str}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Print Summary
    def safe_avg(key):
        vals = [r[key] for r in results if key in r]
        return sum(vals)/len(vals) if vals else 0.0

    print("\n--- Summary ---")
    print(f"Avg PPL - Base    : {safe_avg('base_ppl'):.2f}")
    if "const_ppl" in results[0]:
        print(f"Avg PPL - Constant: {safe_avg('const_ppl'):.2f}")
    if "adapt_ppl" in results[0]:
        print(f"Avg PPL - Adaptive: {safe_avg('adapt_ppl'):.2f}")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
