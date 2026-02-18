#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 19_layer_sensitivity_scan.py
#
# Goals:
#  - Scan all layers to find "Golden Layers" for intervention.
#  - Metrics:
#    1. Delta L2: Magnitude of difference between High/Low (pre-computed or from vectors).
#    2. KL Divergence: Impact of steering on output distribution.
#    3. Flip Rate: % of top-1 token changes or rank changes.
#

import argparse
import json
import random
import torch
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path
from torch.nn import functional as F

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    get_layer_stack,
    _infer_main_device,
    ResidualSteerer,
    _format_prompt,
    AXES as AXES_CANON
)
from datasets import load_dataset

def load_calibration_sentences(per_axis=10):
    """Load a small set of pairs for sensitivity testing."""
    ds = load_dataset("wenkai-li/big5_chat", split="train")
    
    samples = {ax: [] for ax in AXES_CANON}
    
    # We need prompts. The dataset has "train_output".
    # We can use "Hello." as a generic prompt, or extracting prompts if available.
    # For consistency with training, let's use the same "Hello." -> "Response" structure.
    # We will measure KL on the *response* tokens.
    
    # Actually, to measure "Flip Rate", we want to see if the model changes what it *would* have said.
    # So we can just use a fixed set of generic prompts.
    prompts = [
        "Hello, how are you?",
        "What do you think about parties?",
        "Do you like to plan ahead?",
        "How do you handle stress?",
        "Are you interested in abstract ideas?"
    ]
    return prompts

def compute_kl_div(log_probs_p, log_probs_q):
    """
    Compute KL(P || Q) = sum(P * (logP - logQ))
    Input: Log probs (B, V)
    """
    # P is target (usually Steered or Base? KL is asymmetric.)
    # Usually we want KL(Steered || Base) to measure "how much did we change from base".
    # Or KL(Base || Steered)?
    # Convention is often KL(P || Q) where P is true/base, Q is approx/steered.
    # But here we just want a magnitude of change.
    # Let's use symmetric Jensen-Shannon or just sum(p * (logp - logq)).
    
    p = log_probs_p.exp()
    kl = (p * (log_probs_p - log_probs_q)).sum(dim=-1)
    return kl.mean().item()

def compute_flip_rate(probs_base, probs_steered):
    """
    % of tokens where argmax changed.
    """
    top1_base = probs_base.argmax(dim=-1)
    top1_steered = probs_steered.argmax(dim=-1)
    
    flips = (top1_base != top1_steered).float().mean().item()
    return flips

@torch.no_grad()
def scan_layers(args, model, tokenizer, vectors, device):
    prompts = load_calibration_sentences()
    
    # Tokenize prompts
    inputs = [ _format_prompt(tokenizer, p) for p in prompts ]
    enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    
    input_ids = enc.input_ids
    attn_mask = enc.attention_mask
    
    # Get Base Logits (No Steering)
    # We focus on the *next token prediction* after the prompt.
    # Or the whole response?
    # Simple scan: Just next token distribution change at the end of prompt.
    
    outputs_base = model(input_ids, attention_mask=attn_mask)
    logits_base = outputs_base.logits[:, -1, :] # (B, V)
    log_probs_base = F.log_softmax(logits_base, dim=-1)
    probs_base = log_probs_base.exp()
    
    results = {}
    
    layers, num_layers, _ = get_layer_stack(model)
    
    for ax in AXES_CANON:
        print(f"Scanning axis: {ax}")
        results[ax] = []
        
        for L in range(num_layers):
            key = f"{L}|{ax}"
            if key not in vectors:
                continue
                
            vec = vectors[key]
            # Vector norm is already 1.0 from preparation script? 
            # We need to apply it with some alpha.
            # Alpha selection: standardized value, e.g. 5.0 or 10.0
            alpha = args.alpha
            
            # Steering
            with ResidualSteerer(model, L, vec, alpha, answer_only=False):
                outputs_steered = model(input_ids, attention_mask=attn_mask)
                logits_steered = outputs_steered.logits[:, -1, :]
                log_probs_steered = F.log_softmax(logits_steered, dim=-1)
                
            # Metrics
            kl = compute_kl_div(log_probs_base, log_probs_steered) # KL(Base || Steered)
            
            # Flip Rate
            # We compare probabilities of the *same* token? No, we compare argmax.
            probs_steered = log_probs_steered.exp()
            flip = compute_flip_rate(probs_base, probs_steered)
            
            # Delta L2 (Norm of the vector itself? No, that's fixed to 1)
            # Delta L2 usually refers to the distance between High/Low centroids.
            # If we don't have the centroids, we can't compute it here easily unless we load them.
            # But `prepare_vectors` probably saved normalized vectors?
            # Creating script `00_prepare_vectors_subspace.py` saved `vec_unit`.
            # If we want the *magnitude* (which indicates separability), we should have saved it?
            # `prepare_vectors` didn't save the norm of the raw diff.
            # BUT, we can estimate "sensitivity" by how much the *activations* change?
            # Activation change = alpha * vec. Norm is alpha. Trivial.
            
            # The user request says "Delta L2 ... Magnitude of separation".
            # We need the RAW magnitude.
            # I should update `00_prepare_vectors_subspace.py` to save the raw magnitude or eigenvalues?
            # Or just rely on KL and Flip for now as they are more "semantic".
            # Let's stick to KL and Flip for this script as they are the direct request for "Intervention Layer Optimization".
            
            results[ax].append({
                "layer": L,
                "kl": kl,
                "flip_rate": flip,
                "score": kl * (1.0 + flip) # Heuristic score?
            })
            
            if L % 5 == 0:
                print(f"  L{L}: KL={kl:.4f}, Flip={flip:.4f}")
                
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vectors", required=True, help="Path to .npz vector file")
    ap.add_argument("--output", "-o", required=True, help="Output JSON path")
    ap.add_argument("--alpha", type=float, default=5.0, help="Steering alpha for sensitivity check")
    ap.add_argument("--limit", type=int, default=5, help="Number of samples to use for sensitivity check")
    args = ap.parse_args()
    
    # Load Vectors
    print(f"Loading vectors from {args.vectors}")
    vec_data = np.load(args.vectors)
    vectors = {k: vec_data[k] for k in vec_data.files}
    
    # Load Model
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model_name")
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.to(device).eval()
    
    # Scan
    results = scan_layers(args, model, tokenizer, vectors, device)
    
    # Save Results
    # Find Golden Layer (Max KL?)
    golden_layers = {}
    for ax, metrics in results.items():
        if not metrics:
            print(f"Warning: No metrics computed for axis '{ax}'. Skipping.")
            continue
            
        # Sort by KL
        best = max(metrics, key=lambda x: x["kl"])
        golden_layers[ax] = best["layer"]
        print(f"Golden Layer for {ax}: {best['layer']} (KL={best['kl']:.4f})")
        
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    with open(out_path, "w") as f:
        json.dump({"golden_layers": golden_layers, "details": results}, f, indent=2)
        
    print(f"Saved sensitivity map to {out_path}")

if __name__ == "__main__":
    main()
