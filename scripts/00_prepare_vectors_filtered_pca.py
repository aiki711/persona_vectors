#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 00_prepare_vectors_filtered_pca.py
#
# Goals:
#  - Extract High-Low pairs from Big5Chat.
#  - [New] Pre-normalize hidden states (L2) to focus on direction quality, minimizing "high activation" bias.
#  - [New] Filter pairs by Difference Norm (select top K strongest directional differences).
#  - [New] Apply PCA on the filtered subset to extract the primary steering direction.
#  - Avoid overfitting by setting a minimum threshold for K.

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import os
import numpy as np
import torch
import yaml
from datasets import load_dataset
from sklearn.decomposition import PCA
from torch.nn.functional import normalize

# Helper imports from live_axes_and_hook
from persona_vectors.live_axes import (
    AXES as AXES_CANON,
    load_model_and_tokenizer,
    _infer_main_device,
    _is_bnb_quantized,
    get_layer_stack,
)

# ==============================================
#  Big5Chat Data Extraction
# ==============================================

def extract_big5_pairs_from_hf(per_axis: int = 1000) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract (high, low) response pairs from Big5Chat dataset.
    Returns: PAIRS[axis] = [(text_high, text_low), ...]
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
        # Use first pair
        PAIRS[tr].append((highs[0], lows[0]))

    for ax in AXES_CANON:
        random.shuffle(PAIRS[ax])
        if per_axis > 0:
            PAIRS[ax] = PAIRS[ax][:per_axis]
        print(f"[big5chat-pairs] {ax}: {len(PAIRS[ax])} pairs")

    return PAIRS


# ==============================================
#  Main Logic: Filtered PCA
# ==============================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML config path")
    ap.add_argument("--bank_path", "-b", required=True, help="Axes bank path (output)")
    ap.add_argument("--model_name", "-m", help="Override config model_name")
    ap.add_argument("--filter_ratio", type=float, default=0.25, help="Ratio of top pairs to keep (e.g. 0.25 for top 25%)")
    ap.add_argument("--min_pairs", type=int, default=50, help="Minimum number of pairs to keep to avoid overfitting")
    ap.add_argument("--normalize", action="store_true", default=True, help="Apply L2 normalization to hidden states before differencing")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("config requires model_name")

    quant = cfg.get("quant", "auto")
    per_axis = int(cfg.get("per_axis", 2000)) # Default increased to get better filtering pool
    batch_size = int(cfg.get("batch_size", 4))
    
    bank_path_str = args.bank_path
    
    print("=== 00_prepare_vectors_filtered_pca.py ===")
    print(f"  model_name   : {model_name}")
    print(f"  filter_ratio : {args.filter_ratio}")
    print(f"  min_pairs    : {args.min_pairs}")
    print(f"  normalize    : {args.normalize}")
    print(f"  per_axis     : {per_axis}")
    print(f"  bank_path    : {bank_path_str}")

    seed = int(cfg.get("seed", 2025))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load Model
    print("\n[Step 1] Loading model/tokenizer...")
    model, tok = load_model_and_tokenizer(model_name, quant=quant)
    
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        
    if not tok.chat_template:
        print("Warning: No chat template found. Using default ChatML-like template.")
        tok.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    layers_stack, N_layers, kind = get_layer_stack(model)
    layer_indices = list(range(N_layers))
    H_dim = model.config.hidden_size
    device = _infer_main_device(model)
    
    # Enable eval
    if _is_bnb_quantized(model):
        model.eval()
    else:
        model.to(device).eval()

    # 2. Load Data
    print("\n[Step 2] Loading Big5Chat pairs...")
    PAIRS = extract_big5_pairs_from_hf(per_axis=per_axis)

    # 3. Helper Hidden States
    @torch.no_grad()
    def get_assistant_hidden_states(texts: List[str]) -> Dict[int, torch.Tensor]:
        """
        Process a batch of assistant response texts.
        Returns: {L: Tensor(B, H)}
        """
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
        
        results = {L: [] for L in layer_indices}
        
        for b in range(len(texts)):
            start_idx = len_prefix
            end_idx = attn_mask[b].sum().item()
            if start_idx >= end_idx:
                start_idx = end_idx - 1
            
            for L in layer_indices:
                hidden = out.hidden_states[L][b] # (T, H)
                valid_hidden = hidden[start_idx:end_idx]
                pooled = valid_hidden.mean(dim=0) # (H,)
                results[L].append(pooled)
                
        for L in layer_indices:
            results[L] = torch.stack(results[L]) # (B, H)
            
        return results

    # 4. Compute Vectors
    final_axes: Dict[Tuple[int, str], np.ndarray] = {}
    
    for ax in AXES_CANON:
        print(f"\nProcessing Axis: {ax}")
        pairs = PAIRS[ax]
        
        # Store all raw diffs or hidden states?
        # We need to filter based on layer-specific norms.
        # So we collect all (High, Low) tensors first.
        
        highs_list = []
        lows_list = []
        
        # Batch processing
        print("  Computing hidden states...")
        high_states_agg = {L: [] for L in layer_indices}
        low_states_agg = {L: [] for L in layer_indices}
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_highs = [p[0] for p in batch]
            batch_lows = [p[1] for p in batch]
            
            # Combine to save inference calls if memory fits, or do separate
            # Let's do all texts together to batch efficiently? 
            # Actually separate batching of High and Low is easier for logic, just 2 calls.
            
            h_out = get_assistant_hidden_states(batch_highs)
            l_out = get_assistant_hidden_states(batch_lows)
            
            for L in layer_indices:
                high_states_agg[L].append(h_out[L].cpu())
                low_states_agg[L].append(l_out[L].cpu())
        
        # Concatenate all
        for L in layer_indices:
            try:
                H_all = torch.cat(high_states_agg[L], dim=0) # (N, H)
                L_all = torch.cat(low_states_agg[L], dim=0)  # (N, H)
            except RuntimeError:
                print(f"Error concatenating layer {L}")
                continue

            # --- Pre-normalization (User Request) ---
            if args.normalize:
                # L2 Normalize feature vectors
                # Avoid division by zero
                H_all = normalize(H_all, p=2.0, dim=1)
                L_all = normalize(L_all, p=2.0, dim=1)
                
            # Difference
            Diffs = H_all - L_all # (N, H)
            
            # --- Filtering Logic ---
            # Calculate Norms of difference vectors
            norms = torch.norm(Diffs, p=2, dim=1) # (N,)
            
            # Determine threshold
            N = Diffs.shape[0]
            k = int(N * args.filter_ratio)
            if k < args.min_pairs:
                k = min(N, args.min_pairs)
            
            # Sort and selecting top K
            # torch.topk returns values, indices
            if k >= N:
                # Keep all
                indices = torch.arange(N)
                print(f"    L{L}: Keep all {N} (k={k})")
            else:
                _, indices = torch.topk(norms, k)
                # print(f"    L{L}: Filtering top {k}/{N} pairs (Threshold: {_.min():.4f})")
                
            filtered_diffs = Diffs[indices].numpy() # (K, H)
            
            # --- PCA ---
            if filtered_diffs.shape[0] < 2:
                # Fallback to mean if too few
                vec = np.mean(filtered_diffs, axis=0)
            else:
                pca = PCA(n_components=1)
                pca.fit(filtered_diffs)
                vec = pca.components_[0]
                
            # Sign correction (align with mean diff direction)
            mean_vec = np.mean(filtered_diffs, axis=0)
            if np.dot(vec, mean_vec) < 0:
                vec = -vec
                
            # Normalize final vector (unit vector)
            norm = np.linalg.norm(vec) + 1e-12
            vec_unit = vec / norm
            
            final_axes[(L, ax)] = vec_unit.astype(np.float32)

    # 5. Save
    bank_path = Path(bank_path_str)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    
    npz_dict = {
        f"{L}|{ax}": vec
        for (L, ax), vec in final_axes.items()
    }
    
    np.savez_compressed(bank_path, **npz_dict)
    print(f"\n[Done] Saved Filtered PCA axes bank to: {bank_path}")
    print(f"  Total: {len(npz_dict)} vectors")

if __name__ == "__main__":
    main()
