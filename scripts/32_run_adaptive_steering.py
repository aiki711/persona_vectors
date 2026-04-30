#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 32_run_adaptive_steering.py
#
# Adaptive Steering Implementation based on SVM Decision Boundary.
#
# Intervention rule:
#   d(h) = w_norm · x + b_norm
#   if d(h) < tau:
#       alpha = alpha_base * max(0, tau - d(h))
#   else:
#       alpha = 0
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

from transformers import PreTrainedTokenizer

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    _is_bnb_quantized,
    get_layer_stack,
    _format_prompt
)

@dataclass
class AdaptiveSteerer:
    """
    Adaptive steering hook that computes the signed distance from the 
    decision boundary at each forward pass, and dynamically scales the intervention.
    """
    model: torch.nn.Module
    layer: int
    w: np.ndarray      # Normal vector of the hyperplane (H,)
    b: float           # Bias of the hyperplane
    tau: float         # Target margin
    max_alpha: float   # Max intervention strength to prevent explosion
    answer_only: bool = True

    def __post_init__(self):
        self.handle = None

    def __enter__(self):
        w_t = torch.tensor(self.w, dtype=torch.float32)
        b_t = float(self.b)

        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            
            if not torch.isfinite(hs).all():
                return out

            if self.answer_only and hs.size(1) != 1:
                return out

            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            device = hs.device
            
            w_dev = w_t.to(device)

            # d(h) = w \cdot h + b
            # hs_f32 is (B, 1, H) -> (B, 1) -> scalar
            # We assume B=1 for simplicity in this script
            dot_product = (hs_f32 * w_dev).sum(dim=-1) # (B, 1)
            dist = dot_product + b_t # (B, 1)
            
            # alpha = max(0, tau - dist)
            # Clip to max_alpha
            alpha = torch.clamp((self.tau - dist), min=0.0, max=self.max_alpha) # (B, 1)
            
            # Add to hidden states
            add = w_dev.view(1, 1, -1)
            steered_hs_f32 = hs_f32 + alpha.unsqueeze(-1) * add
            
            if not torch.isfinite(steered_hs_f32).all():
                return out

            steered_hs = steered_hs_f32.to(orig_dtype)

            if isinstance(out, tuple):
                return (steered_hs, *out[1:])
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
    
    # Simple generation for demonstration
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    
    # Calculate Perplexity (PPL) for the generated sequence
    # For PPL, we need the logits of the generated sequence.
    # To save memory, we can do a forward pass with the full sequence.
    # We will compute it exactly as `14_calc_personality_score_llm.py` does or simple causal loss.
    
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = outputs[0][prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    return gen_text, outputs[0]

@torch.no_grad()
def calc_sequence_ppl(model, input_ids):
    """
    Calculate Perplexity of the sequence (ignoring the prefix if needed, 
    but here we calculate for the whole sequence for simplicity, or just the generated part).
    Let's calculate for the whole sequence for comparative measure.
    """
    labels = input_ids.clone()
    # Simple PPL over the whole sequence
    outputs = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
    loss = outputs.loss
    return torch.exp(loss).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML config path")
    ap.add_argument("--boundary_bank", required=True, help="Path to boundary_vectors.npz")
    ap.add_argument("--prompts", required=True, help="Input prompts JSONL (e.g. from util_extract_prompts)")
    ap.add_argument("--out_dir", required=True, help="Output directory for results")
    ap.add_argument("--axis", type=str, default="extraversion")
    ap.add_argument("--layer", type=str, default="15", help="Comma-separated layers (e.g. '10,15,20')")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high", help="Direction to steer towards")
    ap.add_argument("--tau", type=float, default=2.0, help="Target margin for adaptive steering")
    ap.add_argument("--max_alpha", type=float, default=5.0, help="Max intervention scale")
    ap.add_argument("--constant_alpha", type=float, default=5.0, help="Alpha for constant steering comparison")
    ap.add_argument("--tag", type=str, default=None, help="Custom tag for output filename")
    
    args = ap.parse_args()
    
    layers = [int(l.strip()) for l in args.layer.split(",")]
    direction_mult = 1.0 if args.direction == "high" else -1.0

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model_name")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== 32_run_adaptive_steering.py ===")
    print(f"  Model       : {model_name}")
    print(f"  Axis/Layers : {args.axis} / {layers}")
    print(f"  Direction   : {args.direction.upper()}")
    print(f"  Adaptive Tau: {args.tau}, Max Alpha: {args.max_alpha}")
    print(f"  Constant Alp: {args.constant_alpha}")

    # 1. Load Boundaries for all specified layers
    b_data = np.load(args.boundary_bank)
    layer_w = {}
    layer_b = {}
    for l in layers:
        w_key = f"{l}|{args.axis}|w"
        b_key = f"{l}|{args.axis}|b"
        if w_key not in b_data:
            print(f"Error: Boundary for layer {l} / axis {args.axis} not found!")
            return
        # Multiply w by direction_mult so that intervention is correctly oriented
        layer_w[l] = b_data[w_key].astype(np.float32) * direction_mult
        layer_b[l] = float(b_data[b_key][0]) * direction_mult # Also flip the bias so d(h) logic remains the same relative to the new target
        
    # Example stats for the first layer in the list
    first_l = layers[0]
    print(f"  Boundary (L{first_l}) -> ||w||: {np.linalg.norm(layer_w[first_l]):.4f}, b: {layer_b[first_l]:.4f}")
    
    # 2. Load Prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            # Remove trailing comma if present (from array sampling)
            if line.endswith(","):
                line = line[:-1]
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # Fallback if it's not valid JSON, just use the raw string
                item = line.strip('"')
                
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
        
        # Limit to 50 prompts if more exist
        prompts = prompts[:50]

    # 3. Load Model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    # We will borrow ResidualSteerer for constant steering comparison
    from persona_vectors.live_axes import ResidualSteerer

    results = []

    print(f"\nProcessing {len(prompts)} prompts...")
    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts)):
        
        # --- 1. Baseline (No Steering) ---
        base_text, base_ids = generate_text(model, tokenizer, p_text)
        base_ppl = calc_sequence_ppl(model, base_ids)
        
        # --- 2. Constant Steering ---
        # We need to nest context managers for multiple layers.
        # Instead of deep nesting, we can collect them in a list and enter them.
        import contextlib
        with contextlib.ExitStack() as stack:
            for l in layers:
                stack.enter_context(ResidualSteerer(model, l, layer_w[l], args.constant_alpha, answer_only=True))
            const_text, const_ids = generate_text(model, tokenizer, p_text)
            const_ppl = calc_sequence_ppl(model, const_ids)
            
        # --- 3. Adaptive Steering ---
        with contextlib.ExitStack() as stack:
            for l in layers:
                stack.enter_context(AdaptiveSteerer(model, l, layer_w[l], layer_b[l], tau=args.tau, max_alpha=args.max_alpha, answer_only=True))
            adapt_text, adapt_ids = generate_text(model, tokenizer, p_text)
            adapt_ppl = calc_sequence_ppl(model, adapt_ids)
            
        results.append({
            "idx": idx,
            "orig_idx": orig_idx,
            "prompt": p_text,
            "base_text": base_text,
            "base_ppl": base_ppl,
            "const_text": const_text,
            "const_ppl": const_ppl,
            "adapt_text": adapt_text,
            "adapt_ppl": adapt_ppl
        })

    # Save Results
    # Output file name includes direction and layers
    layers_str = "_".join(map(str, layers))
    tag_str = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"adaptive_{args.axis}_{args.direction}_L{layers_str}{tag_str}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Print Summary
    avg_base_ppl = sum(r["base_ppl"] for r in results) / len(results)
    avg_const_ppl = sum(r["const_ppl"] for r in results) / len(results)
    avg_adapt_ppl = sum(r["adapt_ppl"] for r in results) / len(results)
    
    print("\n--- Summary ---")
    print(f"Avg PPL - Base    : {avg_base_ppl:.2f}")
    print(f"Avg PPL - Constant: {avg_const_ppl:.2f}")
    print(f"Avg PPL - Adaptive: {avg_adapt_ppl:.2f}")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
