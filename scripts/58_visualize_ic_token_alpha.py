#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt
)

class TrackingICAdaptiveSteerer:
    """
    IC-Adaptive steering with detailed tracking of surprisal and alpha per token.
    """
    def __init__(self, model, layer, w, b, tau, ic_scale=1.0):
        self.model = model
        self.layer = layer
        self.w = w
        self.b = b
        self.tau = tau
        self.ic_scale = ic_scale
        
        self.handle = None
        self.ic_factor = 0.0
        self.surprisal = 0.0
        self.max_prob = 1.0
        
        # Tracking logs
        self.logs = []
        self.current_token_idx = 0

    def update_ic(self, logits):
        """Update IC factor based on the latest logits (before sampling next token)."""
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        max_prob = probs.max().item()
        
        surprisal = -np.log2(max_prob + 1e-10)
        
        # Current heuristic (same as 50_run_ic_adaptive_steering.py to debug it)
        ic_factor = max(0.0, surprisal - 3.0) / 10.0
        ic_factor *= self.ic_scale
        
        self.max_prob = max_prob
        self.surprisal = surprisal
        self.ic_factor = ic_factor

    def __enter__(self):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all(): return out
            if hs.size(1) != 1: return out # Only apply to generated tokens

            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            device = hs.device
            w_dev = self.w.to(device)

            dot_product = (hs_f32 * w_dev).sum(dim=-1)
            dist = dot_product + self.b
            
            # Base Adaptive Alpha
            base_alpha = torch.clamp((self.tau - dist), min=0.0).item()
            
            # Apply IC Weight
            final_alpha = base_alpha * self.ic_factor
            
            # Log the step
            self.logs.append({
                "step": self.current_token_idx,
                "dist": dist.item(),
                "base_alpha": base_alpha,
                "max_prob": self.max_prob,
                "surprisal": self.surprisal,
                "ic_factor": self.ic_factor,
                "final_alpha": final_alpha
            })
            self.current_token_idx += 1
            
            add = w_dev.view(1, 1, -1)
            steered = hs_f32 + final_alpha * add

            if not torch.isfinite(steered).all(): return out
            steered = steered.to(orig_dtype)
            if isinstance(out, tuple): return (steered, *out[1:])
            return steered

        stack, _, _ = get_layer_stack(self.model)
        self.handle = stack[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle: self.handle.remove(); self.handle = None

def generate_and_track(model, tokenizer, prompt, steerer, max_new_tokens=50):
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    gen_ids = inputs.input_ids
    generated_tokens = []
    
    with steerer:
        for _ in range(max_new_tokens):
            outputs = model(gen_ids)
            logits = outputs.logits
            
            # 1. Update IC factor for the NEXT token using current logits
            steerer.update_ic(logits)
            
            # 2. Greedy Sample next token
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
            
            token_str = tokenizer.decode(next_token[0])
            generated_tokens.append(token_str)
            
            gen_ids = torch.cat([gen_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    prompt_len = inputs.input_ids.shape[1]
    decoded = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    return decoded, generated_tokens, steerer.logs

def plot_token_tracking(df: pd.DataFrame, out_dir: Path, title_suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    x = np.arange(len(df))
    labels = df['token'].tolist()
    
    # Plot 1: Alphas
    ax1.plot(x, df['base_alpha'], label='Base Alpha (tau - dist)', marker='o', color='gray', linestyle='--')
    ax1.plot(x, df['final_alpha'], label='Final Alpha (IC-Weighted)', marker='s', color='red')
    ax1.set_ylabel("Alpha")
    ax1.set_title("Intervention Magnitude over Tokens")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Surprisal vs Threshold
    ax2.bar(x, df['surprisal'], color='royalblue', alpha=0.7, label='Surprisal (-log2 P)')
    ax2.axhline(3.0, color='red', linestyle='--', label='Current Threshold (3.0)')
    ax2.set_ylabel("Surprisal")
    ax2.set_title("Token Surprisal Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Max Prob
    ax3.plot(x, df['max_prob'], color='green', marker='d', label='Max Prob')
    ax3.set_ylabel("Probability")
    ax3.set_title("Argmax Token Probability")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=90, fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / f"ic_token_tracking_{title_suffix}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved token tracking plot: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--vector_bank", default="vectors.npz")
    parser.add_argument("--axis", type=str, default="extraversion")
    parser.add_argument("--target_layer", type=int, default=15)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--out_dir", default="exp_steering_ic_adaptive/figures/debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    v_data = np.load(args.vector_bank)
    w = torch.tensor(v_data[f"{args.target_layer}|{args.axis}|w"], dtype=torch.float32)
    b = float(v_data[f"{args.target_layer}|{args.axis}|b"][0])
    
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], quant=cfg.get("quant", "auto"))
    model.eval()

    test_prompt = "Zendaya, I think we've had enough sun for today. We're both looking like lobsters! Let's grab some aloe vera and head back to the shade. I don't want to be peeling for the next week."
    
    steerer = TrackingICAdaptiveSteerer(model, args.target_layer, w, b, tau=args.tau)
    
    print(f"Generating with Greedy Decoding... Layer={args.target_layer}, Tau={args.tau}")
    decoded, tokens, logs = generate_and_track(model, tokenizer, test_prompt, steerer)
    
    # Ensure logs match tokens (hook runs BEFORE outputting the token)
    # Actually, hook runs during the forward pass.
    # We collect log in hook (which is applied to the token being generated).
    
    df = pd.DataFrame(logs)
    if len(tokens) == len(df):
        df['token'] = [t.replace('\n', '\\n') for t in tokens]
    else:
        # Pad or truncate
        min_len = min(len(tokens), len(df))
        df = df.iloc[:min_len].copy()
        df['token'] = [t.replace('\n', '\\n') for t in tokens[:min_len]]
        
    csv_path = out_dir / f"ic_tracking_{args.axis}_L{args.target_layer}_Tau{args.tau}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    plot_token_tracking(df, out_dir, f"{args.axis}_L{args.target_layer}_Tau{args.tau}")
    
    print("\nGenerated Text:")
    print(decoded)

if __name__ == "__main__":
    main()
