#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 59_run_dynamic_layer_steering.py
#
# Dynamic Layer Selection (DLS) Steering
#
# For each prompt p and a given alpha:
#   1. Run baseline forward pass -> z_base (logits)
#   2. For each candidate layer L:
#        Apply small steering hook at L -> z_steered_L
#        nu(L) = ||z_steered_L - z_base||_2   (last-token logits)
#   3. L* = argmax nu(L)
#   4. Generate full text with alpha at L*
#
# Output JSONL per alpha:
#   dyn_Val{alpha}.jsonl

from __future__ import annotations

import argparse
import json
import torch
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
    """Return last-token logits for baseline (no steering)."""
    with torch.no_grad():
        out = model(input_ids)
    return out.logits[0, -1, :].float()  # (vocab_size,)


def get_steered_logits(model, input_ids, layer, w_dev, alpha):
    """Apply constant steering hook at a single layer and return last-token logits."""
    stack, _, _ = get_layer_stack(model)

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not torch.isfinite(hs).all():
            return out
        orig_dtype = hs.dtype
        hs_f32 = hs.to(torch.float32)
        add = w_dev.view(1, 1, -1)
        steered = hs_f32 + alpha * add
        if not torch.isfinite(steered).all():
            return out
        steered = steered.to(orig_dtype)
        return (steered, *out[1:]) if isinstance(out, tuple) else steered

    handle = stack[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0, -1, :].float()
    finally:
        handle.remove()
    return logits


def select_best_layer(model, input_ids, w_dev, alpha):
    """Compute delta-logit norms across all candidate layers and return argmax layer."""
    base_logits = get_base_logits(model, input_ids)
    norms = {}
    for L in LAYERS:
        steered_logits = get_steered_logits(model, input_ids, L, w_dev, alpha)
        delta = steered_logits - base_logits
        norms[L] = delta.norm().item()
    best_layer = max(norms, key=lambda L: norms[L])
    return best_layer, norms


def generate_with_steered_layer(model, tokenizer, prompt, w_dev, alpha, layer,
                                 max_new_tokens=150):
    """Generate text with constant steering at a specific layer."""
    device = _infer_main_device(model)
    inputs = format_and_tokenize(tokenizer, prompt, device)
    stack, _, _ = get_layer_stack(model)

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not torch.isfinite(hs).all():
            return out
        if hs.size(1) != 1:  # answer_only: skip prefill
            return out
        orig_dtype = hs.dtype
        hs_f32 = hs.to(torch.float32)
        add = w_dev.view(1, 1, -1)
        steered = hs_f32 + alpha * add
        if not torch.isfinite(steered).all():
            return out
        steered = steered.to(orig_dtype)
        return (steered, *out[1:]) if isinstance(out, tuple) else steered

    handle = stack[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
    finally:
        handle.remove()

    prompt_len = inputs.input_ids.shape[1]
    gen_ids = outputs[0]
    text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return text, gen_ids


@torch.no_grad()
def calc_ppl(model, ids):
    labels = ids.clone()
    out = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
    return torch.exp(out.loss).item()


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--axis", type=str, default="extraversion")
    ap.add_argument("--alpha", type=float, required=True, help="Steering strength")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high")
    args = ap.parse_args()

    direction_mult = 1.0 if args.direction == "high" else -1.0

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model_name")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"dyn_Val{args.alpha:g}.jsonl"
    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # Load vectors for all layers
    v_data = np.load(args.vector_bank)
    layer_w = {}
    for L in LAYERS:
        w_key = f"{L}|{args.axis}|w"
        if w_key not in v_data:
            print(f"[WARN] Key {w_key} not found, skipping layer {L}")
            continue
        layer_w[L] = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult

    if not layer_w:
        print("[ERROR] No layer vectors found. Exiting.")
        return

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
            except Exception:
                item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
    prompts = prompts[:10]

    print(f"=== 59_run_dynamic_layer_steering.py ===")
    print(f"  Model  : {model_name}")
    print(f"  Axis   : {args.axis}")
    print(f"  Alpha  : {args.alpha}")
    print(f"  Prompts: {len(prompts)}")

    model, tokenizer = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    # Move all w vectors to device
    layer_w_dev = {L: w.to(device) for L, w in layer_w.items()}

    results = []
    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts, desc=f"DLS alpha={args.alpha}")):
        inputs = format_and_tokenize(tokenizer, p_text, device)

        # 1. Baseline generation
        with torch.no_grad():
            base_outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        prompt_len = inputs.input_ids.shape[1]
        base_text = tokenizer.decode(base_outputs[0][prompt_len:], skip_special_tokens=True)
        base_ppl = calc_ppl(model, base_outputs[0])

        # 2. Select best layer via delta-logit norm
        # Use the full prompt token sequence for probing (prefill)
        best_layer, norms = select_best_layer(
            model, inputs.input_ids, layer_w_dev[LAYERS[0]], args.alpha
        )
        # Re-run with the per-layer w vector
        best_layer, norms = select_best_layer(
            model, inputs.input_ids,
            layer_w_dev.get(LAYERS[0], layer_w_dev[list(layer_w_dev.keys())[0]]),
            args.alpha
        )
        # Correct: probe each layer with its OWN w vector
        base_logits = get_base_logits(model, inputs.input_ids)
        norms = {}
        for L in LAYERS:
            if L not in layer_w_dev:
                continue
            steered_logits = get_steered_logits(model, inputs.input_ids, L, layer_w_dev[L], args.alpha)
            delta = steered_logits - base_logits
            norms[L] = delta.norm().item()
        best_layer = max(norms, key=lambda L: norms[L])

        # 3. Generate with best layer
        dyn_text, dyn_ids = generate_with_steered_layer(
            model, tokenizer, p_text, layer_w_dev[best_layer], args.alpha, best_layer
        )
        dyn_ppl = calc_ppl(model, dyn_ids)

        results.append({
            "idx": idx,
            "orig_idx": orig_idx,
            "prompt": p_text,
            "base_text": base_text,
            "base_ppl": base_ppl,
            "dyn_text": dyn_text,
            "dyn_ppl": dyn_ppl,
            "dyn_layer": best_layer,
            "delta_logit_norms": {str(L): float(v) for L, v in norms.items()},
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_base = sum(r["base_ppl"] for r in results) / len(results)
    avg_dyn  = sum(r["dyn_ppl"]  for r in results) / len(results)
    layer_counts = {}
    for r in results:
        l = r["dyn_layer"]
        layer_counts[l] = layer_counts.get(l, 0) + 1

    print(f"\n--- Summary (alpha={args.alpha}) ---")
    print(f"  Base PPL: {avg_base:.2f}")
    print(f"  DLS  PPL: {avg_dyn:.2f}")
    print(f"  Layer distribution: {dict(sorted(layer_counts.items()))}")
    print(f"  Saved to: {out_file}")


if __name__ == "__main__":
    main()
