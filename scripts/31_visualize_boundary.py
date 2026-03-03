#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 31_visualize_boundary.py
#
# Goals:
#  - Extract High/Low pairs (similar to training script).
#  - Load the trained SVM boundary vectors (w, b).
#  - Project High/Low hidden states onto the normal vector (w) and the first principal component orthogonal to w to visualize the "clouds" and decision boundary.
#

from __future__ import annotations

import argparse
import random
import os
import numpy as np
import torch
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from datasets import load_dataset

from persona_vectors.live_axes import (
    AXES as AXES_CANON,
    load_model_and_tokenizer,
    _infer_main_device,
    _is_bnb_quantized,
    get_layer_stack,
)

def extract_big5_pairs_from_hf(per_axis: int = 100) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract (high, low) response pairs from Big5Chat dataset.
    Using a smaller subset for visualization to keep it clear.
    """
    ds_all = load_dataset("wenkai-li/big5_chat")
    if isinstance(ds_all, dict):
        split_name = next(iter(ds_all.keys()))
        ds = ds_all[split_name]
    else:
        ds = ds_all

    buckets: Dict[tuple, Dict[str, List[str]]] = defaultdict(
        lambda: {"high": [], "low": []}
    )

    for ex in ds:
        tr_raw = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr_raw not in AXES_CANON or lv not in {"high", "low"}:
            continue

        orig_idx = ex.get("original_index")
        if orig_idx is None:
            continue

        to = (ex.get("train_output") or "").strip()
        if not to:
            continue

        buckets[(tr_raw, orig_idx)][lv].append(to)

    PAIRS: Dict[str, List[Tuple[str, str]]] = {ax: [] for ax in AXES_CANON}
    for (tr, orig_idx), d in buckets.items():
        highs = d["high"]
        lows = d["low"]
        if not highs or not lows:
            continue
        PAIRS[tr].append((highs[0], lows[0]))

    for ax in AXES_CANON:
        random.shuffle(PAIRS[ax])
        if per_axis > 0:
            PAIRS[ax] = PAIRS[ax][:per_axis]
        print(f"[big5chat-pairs] {ax}: {len(PAIRS[ax])} pairs")

    return PAIRS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML config path")
    ap.add_argument("--boundary_bank", required=True, help="Path to boundary_vectors.npz")
    ap.add_argument("--out_dir", required=True, help="Output directory for figures")
    ap.add_argument("--model_name", "-m", help="Override config model_name")
    ap.add_argument("--axis", type=str, default="extraversion", help="Process only specific axis")
    ap.add_argument("--layer", type=int, default=15, help="Layer to visualize")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg.get("model_name")
    if not model_name:
        raise ValueError("config requires model_name")

    quant = cfg.get("quant", "auto")
    per_axis = 150 # Enough points to see a distribution
    batch_size = int(cfg.get("batch_size", 4))
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axis = args.axis
    layer = args.layer

    print("=== 31_visualize_boundary.py ===")
    print(f"  Model        : {model_name}")
    print(f"  Boundary     : {args.boundary_bank}")
    print(f"  Output Dir   : {out_dir}")
    print(f"  Target Axis  : {axis}")
    print(f"  Target Layer : {layer}")

    seed = int(cfg.get("seed", 2025))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load Boundary Vectors
    boundary_data = np.load(args.boundary_bank)
    w_key = f"{layer}|{axis}|w"
    b_key = f"{layer}|{axis}|b"
    
    if w_key not in boundary_data or b_key not in boundary_data:
        print(f"Error: Boundary vectors for Layer {layer} Axis {axis} not found in {args.boundary_bank}")
        return
        
    w = boundary_data[w_key].astype(np.float32) # (H,)
    b = boundary_data[b_key][0].astype(np.float32) # scalar
    
    print(f"Loaded boundary: ||w||={np.linalg.norm(w):.4f}, b={b:.4f}")

    # 2. Get Hidden States
    print("\n[Step 1] Loading model/tokenizer...")
    model, tok = load_model_and_tokenizer(model_name, quant=quant)
    
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        
    if not tok.chat_template:
        tok.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

    device = _infer_main_device(model)
    
    if _is_bnb_quantized(model):
        model.eval()
    else:
        model.to(device).eval()

    print("\n[Step 2] Loading Big5Chat pairs...")
    PAIRS = extract_big5_pairs_from_hf(per_axis=per_axis)

    @torch.no_grad()
    def get_assistant_hidden_states_for_layer(texts: List[str], target_layer: int) -> torch.Tensor:
        msgs_prefix = [{"role": "user", "content": "Hello."}]
        prefix_ids = tok.apply_chat_template(msgs_prefix, add_generation_prompt=True, tokenize=True)
        len_prefix = len(prefix_ids)
        
        full_inputs = []
        for t in texts:
            msgs = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": t}]
            full_ids = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
            full_inputs.append(torch.tensor(full_ids))
            
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(full_inputs, batch_first=True, padding_value=tok.pad_token_id).to(device)
        attn_mask = (input_ids != tok.pad_token_id).long()
        
        out = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        results = []
        for b_idx in range(len(texts)):
            start_idx = len_prefix
            end_idx = attn_mask[b_idx].sum().item()
            if start_idx >= end_idx:
                start_idx = end_idx - 1
            
            hidden = out.hidden_states[target_layer][b_idx]
            valid_hidden = hidden[start_idx:end_idx]
            pooled = valid_hidden.mean(dim=0)
            results.append(pooled)
            
        return torch.stack(results)

    pairs = PAIRS[axis]
    highs = []
    lows = []
    
    print("  Computing hidden states (Batch)...")
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_highs = [p[0] for p in batch]
        batch_lows = [p[1] for p in batch]
        
        h_out = get_assistant_hidden_states_for_layer(batch_highs, layer)
        l_out = get_assistant_hidden_states_for_layer(batch_lows, layer)
        
        highs.append(h_out.cpu())
        lows.append(l_out.cpu())

    H_all = torch.cat(highs, dim=0).numpy() # (N, H)
    L_all = torch.cat(lows, dim=0).numpy()  # (N, H)
    
    # 3. Projection and Plotting
    # X-axis will be the distance relative to the boundary: d(x) = w·x + b
    # High is supposed to be > 0.
    
    # Calculate Distances (Projection onto normal vector + bias)
    dist_H = np.dot(H_all, w) + b
    dist_L = np.dot(L_all, w) + b
    
    # To plot a 2D cloud, we need a Y-axis. 
    # Let's find the first principal component of the entire dataset OR orthogonalize wrt w.
    X_all = np.vstack([H_all, L_all])
    
    # Remove the w component from X_all
    # X_perp = X - proj_w(X) = X - (X·w) * w (since w is unit vector)
    projections = np.dot(X_all, w)[:, np.newaxis]
    X_perp = X_all - projections * w

    # Run PCA on the orthogonal subspace to find the direction of maximum variance
    pca = PCA(n_components=1)
    pca.fit(X_perp)
    y_axis_vec = pca.components_[0] # This guarantees to be orthogonal to w
    
    # Project onto Y-axis
    y_H = np.dot(H_all, y_axis_vec)
    y_L = np.dot(L_all, y_axis_vec)

    # 4. Create Plot
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plt.scatter(dist_H, y_H, c='blue', alpha=0.6, label='High (Target)', edgecolors='w', s=50)
    plt.scatter(dist_L, y_L, c='red', alpha=0.6, label='Low (Anti-Target)', edgecolors='w', s=50)
    
    # Plot Decision Boundary (x=0)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary ($w \\cdot h + b = 0$)')
    
    # Plot Margins (Optional: e.g. target margin tau = 1.0)
    tau = 2.0
    plt.axvline(x=tau, color='green', linestyle=':', linewidth=2, label=f'Target Margin ($\\tau={tau}$)')
    
    # Formatting
    plt.title(f'Activation Distribution and SVM Boundary\nAxis: {axis.capitalize()} | Layer: {layer}', fontsize=14)
    plt.xlabel('Signed Distance from Boundary ($w \\cdot h + b$)', fontsize=12)
    plt.ylabel('Orthogonal Variance (1st PC of $X_{\perp}$)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for steering concept
    plt.annotate('', xy=(tau + 0.5, 0), xytext=(0, 0),
            arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
            horizontalalignment='center', verticalalignment='top')
    plt.text(tau/2, 1, 'Adaptive Steering\nPush until threshold', color='green', fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    
    out_file = out_dir / f"boundary_{axis}_L{layer}.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nSaved visualization to {out_file}")

if __name__ == "__main__":
    main()
