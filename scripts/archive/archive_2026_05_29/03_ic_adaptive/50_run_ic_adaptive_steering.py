#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 50_run_ic_adaptive_steering.py
#
# Information Content (IC) Weighted Adaptive Steering.
# Scales steering intensity alpha based on the surprisal (-log P) of the tokens.
#

import argparse
import json
import torch
import torch.nn.functional as F
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

# ==================== IC-Adaptive Steerer ====================

@dataclass
class ICAdaptiveSteerer:
    """
    IC-Adaptive steering: Scales alpha by the Information Content (Surprisal).
    Alpha_final = max(0, tau - dist) * w * IC_factor
    """
    model: torch.nn.Module
    layer: int
    w: torch.Tensor
    b: float
    tau: float
    ic_scale: float = 1.0  # Overall scaling factor for IC
    answer_only: bool = True

    def __post_init__(self):
        self.handle = None
        self.last_prob = 1.0  # Probability of the last token (default 1.0 -> IC 0)
        self.ic_factor = 0.0

    def update_ic(self, logits):
        """Update IC factor based on the latest logits."""
        # logits shape: [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        max_prob = probs.max().item()
        
        # Surprisal (base 2)
        surprisal = -np.log2(max_prob + 1e-10)
        
        # Simplified and corrected heuristic: use surprisal directly to scale alpha,
        # capped at 3.0 to prevent explosion on extremely uncertain tokens.
        self.ic_factor = min(surprisal, 3.0)
        self.ic_factor *= self.ic_scale

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
            
            # Base Adaptive Alpha
            base_alpha = torch.clamp((self.tau - dist), min=0.0)
            
            # Apply IC Weight
            final_alpha = base_alpha * self.ic_factor
            
            add = w_dev.view(1, 1, -1)
            steered = hs_f32 + final_alpha.unsqueeze(-1) * add

            if not torch.isfinite(steered).all(): return out
            steered = steered.to(orig_dtype)
            if isinstance(out, tuple): return (steered, *out[1:])
            return steered

        stack, _, _ = get_layer_stack(self.model)
        self.handle = stack[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle: self.handle.remove(); self.handle = None

# ==================== Custom Generation Loop ====================

def generate_with_ic_steering(model, tokenizer, prompt, steerer, max_new_tokens=100):
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    gen_ids = inputs.input_ids
    
    with steerer:
        for _ in range(max_new_tokens):
            # 1. Forward pass to get logits and apply steering (using last_ic)
            outputs = model(gen_ids)
            logits = outputs.logits
            
            # 2. Update IC factor for the NEXT token using the logits just produced
            steerer.update_ic(logits)
            
            # 3. Sample next token
            next_token_logits = logits[:, -1, :] / 0.7 # Temp 0.7
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            gen_ids = torch.cat([gen_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    prompt_len = inputs.input_ids.shape[1]
    decoded = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    return decoded, gen_ids[0]

# ==================== Main ====================

def calc_ppl(model, ids):
    with torch.no_grad():
        labels = ids.clone()
        outputs = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
        return torch.exp(outputs.loss).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--vector_bank", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--axis", type=str, default="extraversion")
    parser.add_argument("--target_layer", type=int, default=24)
    parser.add_argument("--tau", type=float, default=25.0)
    parser.add_argument("--ic_scale", type=float, default=1.5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Vectors
    v_data = np.load(args.vector_bank)
    w = torch.tensor(v_data[f"{args.target_layer}|{args.axis}|w"], dtype=torch.float32)
    b = float(v_data[f"{args.target_layer}|{args.axis}|b"][0])
    
    # Load Model
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], quant=cfg.get("quant", "auto"))
    model.eval()
    
    # Load Prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"): continue
            if line.endswith(","): line = line[:-1]
            try:
                item = json.loads(line)
            except:
                item = line.strip('"')
            
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
    prompts = prompts[:10]

    results = []
    steerer = ICAdaptiveSteerer(model, args.target_layer, w, b, tau=args.tau, ic_scale=args.ic_scale)
    
    print(f"Starting IC-Adaptive Sweep: Layer={args.target_layer}, Tau={args.tau}, IC_Scale={args.ic_scale}")
    
    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts)):
        text, ids = generate_with_ic_steering(model, tokenizer, p_text, steerer)
        ppl = calc_ppl(model, ids)
        
        results.append({
            "idx": idx,
            "orig_idx": orig_idx,
            "prompt": p_text,
            "ic_adapt_text": text,
            "ic_adapt_ppl": ppl
        })
    
    # Save
    out_file = out_dir / f"ic_adapt_layer{args.target_layer}_Tau{args.tau}_S{args.ic_scale}.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    avg_ppl = sum(r["ic_adapt_ppl"] for r in results) / len(results)
    print(f"Average IC-Adaptive PPL: {avg_ppl:.2f}")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
