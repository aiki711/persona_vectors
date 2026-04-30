#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 40_run_layer_sweep.py
#
# Single-Layer Steering: Intervene on ONE specified layer only.
# Compares Constant vs Adaptive steering at the given layer.
#
# Usage:
#   python 40_run_layer_sweep.py \
#     --config config/mistral_7b.yaml \
#     --vector_bank exp_adaptive_steering/vectors/mean_diff_vectors.npz \
#     --prompts exp_adaptive_steering/test_prompts_10.jsonl \
#     --out_dir exp_steering_layer_sweep/results/extraversion \
#     --axis extraversion \
#     --target_layer 15 \
#     --tau 0.09 \
#     --alpha 0.09 \
#     --mode both

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

# ==================== Steerer Classes ====================

@dataclass
class AdaptiveSteererPure:
    """
    Adaptive steering: push activation until it reaches margin 'tau'.
    d(h) = w·h + b; add = max(0, tau - d(h)) * w
    """
    model: torch.nn.Module
    layer: int
    w: torch.Tensor
    b: float
    tau: float
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

            dot_product = (hs_f32 * w_dev).sum(dim=-1)
            dist = dot_product + self.b
            alpha = torch.clamp((self.tau - dist), min=0.0)
            add = w_dev.view(1, 1, -1)
            steered = hs_f32 + alpha.unsqueeze(-1) * add

            if not torch.isfinite(steered).all(): return out
            steered = steered.to(orig_dtype)
            if isinstance(out, tuple): return (steered, *out[1:])
            return steered

        stack, _, _ = get_layer_stack(self.model)
        self.handle = stack[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle: self.handle.remove(); self.handle = None


@dataclass
class ConstantSteerer:
    """
    Constant steering: always add alpha * w to hidden states.
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

            add = w_dev.view(1, 1, -1)
            steered = hs_f32 + self.alpha * add

            if not torch.isfinite(steered).all(): return out
            steered = steered.to(orig_dtype)
            if isinstance(out, tuple): return (steered, *out[1:])
            return steered

        stack, _, _ = get_layer_stack(self.model)
        self.handle = stack[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle: self.handle.remove(); self.handle = None


# ==================== Utility Functions ====================

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
    return tokenizer.decode(gen_ids, skip_special_tokens=True), outputs[0]


@torch.no_grad()
def calc_sequence_ppl(model, input_ids):
    labels = input_ids.clone()
    outputs = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
    return torch.exp(outputs.loss).item()


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--axis", type=str, default="extraversion")
    ap.add_argument("--target_layer", type=int, required=True, help="Single layer to intervene on")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high")
    ap.add_argument("--tau", type=float, default=0.09)
    ap.add_argument("--alpha", type=float, default=0.09)
    ap.add_argument("--mode", type=str, choices=["adaptive", "constant", "both"], default="both")
    args = ap.parse_args()

    direction_mult = 1.0 if args.direction == "high" else -1.0

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model_name")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Vectors for target layer
    v_data = np.load(args.vector_bank)
    w_key = f"{args.target_layer}|{args.axis}|w"
    b_key = f"{args.target_layer}|{args.axis}|b"
    if w_key not in v_data:
        print(f"[ERROR] Key {w_key} not found in vector bank.")
        return
    w = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult
    b = float(v_data[b_key][0]) * direction_mult

    print(f"=== 40_run_layer_sweep.py ===")
    print(f"  Model       : {model_name}")
    print(f"  Axis        : {args.axis}")
    print(f"  Target Layer: {args.target_layer}")
    print(f"  Mode        : {args.mode}")
    print(f"  Params      : Tau={args.tau}, Alpha={args.alpha}")

    # Load Prompts
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

    # Load Model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
    model.eval()

    results = []
    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts, desc=f"Layer {args.target_layer} Val={args.alpha}")):
        res = {"idx": idx, "orig_idx": orig_idx, "prompt": p_text}

        # Baseline
        txt_b, ids_b = generate_text(model, tokenizer, p_text)
        res["base_text"] = txt_b
        res["base_ppl"] = calc_sequence_ppl(model, ids_b)

        # Constant Steering
        if args.mode in ["constant", "both"]:
            with ConstantSteerer(model, args.target_layer, w, args.alpha, answer_only=True):
                txt_c, ids_c = generate_text(model, tokenizer, p_text)
                res["const_text"] = txt_c
                res["const_ppl"] = calc_sequence_ppl(model, ids_c)

        # Adaptive Steering
        if args.mode in ["adaptive", "both"]:
            with AdaptiveSteererPure(model, args.target_layer, w, b, tau=args.tau, answer_only=True):
                txt_a, ids_a = generate_text(model, tokenizer, p_text)
                res["adapt_text"] = txt_a
                res["adapt_ppl"] = calc_sequence_ppl(model, ids_a)

        results.append(res)

    # Save
    out_file = out_dir / f"layer_{args.target_layer}_Val{args.alpha:g}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def safe_avg(key):
        vals = [r[key] for r in results if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n--- Summary (Layer={args.target_layer}, Val={args.alpha}) ---")
    print(f"  Base     PPL={safe_avg('base_ppl'):.2f}")
    if "const_ppl" in results[0]:
        print(f"  Constant PPL={safe_avg('const_ppl'):.2f}")
    if "adapt_ppl" in results[0]:
        print(f"  Adaptive PPL={safe_avg('adapt_ppl'):.2f}")
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    main()
