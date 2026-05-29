#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 82_run_dyn_layer_proj_prior.py
#
# Dynamic Layer Selection using:
#   1. Dot Product Projection (Approach 1)
#   2. Data-Driven Layer Priors (Approach 3)
#
# Real-time selection during generation without requiring extra forward passes or z-score stats.
#

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

LAYERS = list(range(32))
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)

def load_layer_priors(input_dir: Path, axis: str) -> dict:
    """
    Loads 32-layer sweep results from input_dir and calculates prior weights W_l.
    W_l = max(0, max_safe_score - 3.0)
    where safe means ppl <= 25.0.
    """
    trait_dir = input_dir / axis
    priors = {L: 0.0 for L in LAYERS}
    
    for L in LAYERS:
        max_safe_dev = 0.0
        has_any_data = False
        
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{L}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_layer_{L}_Val{val}.csv"
                if not csv_path.exists():
                    continue
            try:
                df = pd.read_csv(csv_path)
                mean_score = df["const_score"].mean()
                mean_ppl = df["const_ppl"].mean()
                has_any_data = True
                
                if mean_ppl <= 25.0:
                    dev = mean_score - 3.0
                    if dev > max_safe_dev:
                        max_safe_dev = dev
            except Exception:
                pass
        
        if has_any_data:
            priors[L] = max(0.0, max_safe_dev)
        else:
            # Fallback if no sweep data exists yet for this layer
            if 10 <= L <= 22:
                priors[L] = 1.0
            else:
                priors[L] = 0.0
                
    # Normalize priors so the max weight is 1.0
    max_w = max(priors.values())
    if max_w > 1e-8:
        for L in priors:
            priors[L] /= max_w
            
    return priors

def select_layer_proj_prior(model, input_ids, layer_w_dev, target_direction, layer_priors, score_mode="projection"):
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
        if score_mode == "cosine":
            # Compute cosine similarity
            h_unit = h / (torch.norm(h) + 1e-10)
            w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
            score = torch.dot(h_unit, w_unit).item()
        else:
            # Normalize w_dev to a unit vector and compute projection
            w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
            score = torch.dot(h, w_unit).item()
        
        if target_direction == "high":
            raw_scores[L] = -score
        else:
            raw_scores[L] = score

    final_scores = {}
    for L in layer_w_dev.keys():
        w_prior = layer_priors.get(L, 0.0)
        if w_prior > 1e-5:
            # Shift low-prior layers downwards. Since raw_scores can be negative,
            # subtraction is mathematically correct (multiplication would make negative scores closer to 0, which favors low priors).
            final_scores[L] = raw_scores[L] - (1.0 - w_prior) * 100.0
        else:
            final_scores[L] = -1e9  # Exclude masked layers completely

    best_layer = max(final_scores, key=lambda L: final_scores[L])
    return best_layer, raw_scores, final_scores

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", required=True)
    ap.add_argument("--vector_bank",  required=True)
    ap.add_argument("--prompts",      required=True)
    ap.add_argument("--input_dir",    default="exp_steering_layer_analysis/results", help="単層スイープ結果ディレクトリ")
    ap.add_argument("--out_dir",      default="exp_steering_dyn_layer_proj_prior/results")
    ap.add_argument("--axis",         type=str, default="extraversion")
    ap.add_argument("--alpha",        type=float, required=True)
    ap.add_argument("--direction",    type=str, choices=["high", "low"], default="high")
    ap.add_argument("--norm_mode",    type=str, choices=["none", "midpoint", "raw_norm"], default="raw_norm")
    ap.add_argument("--no_prior",     action="store_true", help="Bypass prior weights and use only raw score")
    ap.add_argument("--score_mode",   type=str, choices=["projection", "cosine"], default="projection", help="layer selection score mode")
    args = ap.parse_args()

    direction_mult = 1.0 if args.direction == "high" else -1.0
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_mode == "cosine":
        method_name = "cos_only" if args.no_prior else "cos_prior"
    else:
        method_name = "proj_only" if args.no_prior else "proj_prior"
    out_file = out_dir / f"{method_name}_Val{args.alpha}.jsonl"
    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # Load layer priors
    if args.no_prior:
        print("Bypassing layer priors (prior weights set to 1.0 for all layers)...")
        layer_priors = {L: 1.0 for L in LAYERS}
    else:
        print(f"Loading layer priors for {args.axis} from {input_dir}...")
        layer_priors = load_layer_priors(input_dir, args.axis)
        print("Layer Prior Weights:")
        for L in sorted(layer_priors.keys()):
            if layer_priors[L] > 0.0:
                print(f"  Layer {L:2d}: {layer_priors[L]:.4f}")

    # Load vectors
    v_data = np.load(args.vector_bank)
    layer_w = {}
    for L in LAYERS:
        w_key = f"{L}|{args.axis}|w"
        raw_norm_key = f"{L}|{args.axis}|raw_norm"
        mp_key = f"{L}|{args.axis}|midpoint"
        if w_key in v_data:
            w_vec = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult
            
            # Scale using original raw norm of difference vector (raw_norm)
            if args.norm_mode in ["midpoint", "raw_norm"]:
                if raw_norm_key in v_data:
                    r_norm = float(v_data[raw_norm_key][0])
                    w_norm = torch.norm(w_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * r_norm
                elif mp_key in v_data:
                    # Fallback to midpoint norm if raw_norm is not present in older vector banks
                    m_vec = torch.tensor(v_data[mp_key], dtype=torch.float32)
                    w_norm = torch.norm(w_vec).item()
                    m_norm = torch.norm(m_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * m_norm
            layer_w[L] = w_vec

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

    print(f"=== Proj & Prior DLS Execution ===")
    print(f"  Axis  : {args.axis}")
    print(f"  Alpha : {args.alpha}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    layer_w_dev = {L: w.to(device) for L, w in layer_w.items()}
    results = []

    # Generate baseline once for prompts
    baselines = []
    print("Generating baseline texts...")
    for idx, (orig_idx, p_text) in enumerate(prompts):
        inputs = format_and_tokenize(tokenizer, p_text, device)
        with torch.no_grad():
            base_outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
        prompt_len = inputs.input_ids.shape[1]
        base_text = tokenizer.decode(base_outputs[0][prompt_len:], skip_special_tokens=True)
        base_ppl = calc_ppl(model, base_outputs[0])
        baselines.append((base_text, base_ppl))

    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts)):
        inputs = format_and_tokenize(tokenizer, p_text, device)
        base_text, base_ppl = baselines[idx]

        # Layer Selection
        best_layer, raw_scores, final_scores = select_layer_proj_prior(
            model, inputs.input_ids, layer_w_dev, args.direction, layer_priors, args.score_mode
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
            "raw_scores":   {str(L): float(v) for L, v in raw_scores.items()},
            "final_scores": {str(L): float(v) for L, v in final_scores.items()},
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
