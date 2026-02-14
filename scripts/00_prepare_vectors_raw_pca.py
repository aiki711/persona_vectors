#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 00_prepare_vectors_raw_pca.py
#
# Goals:
#  - Extract High-Low pairs from Big5Chat.
#  - [Raw PCA] NO pre-normalization (preserve magnitude info).
#  - [Raw PCA] NO filtering based on norm (use all data to satisfy Linear Probability).
#  - Memory Optimization: Compute difference on the fly and offload to CPU.
#

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
#  Main Logic: Raw PCA
# ==============================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML config path")
    ap.add_argument("--bank_path", "-b", required=True, help="Axes bank path (output)")
    ap.add_argument("--model_name", "-m", help="Override config model_name")
    # Removed filter_ratio / normalize args since they are fixed to False/1.0 for Raw PCA
    # but keeping them as optional to avoid breaking if someone passes them
    ap.add_argument("--normalize", action="store_true", help="Included for compatibility, but ignored/warned")
    args = ap.parse_args()

    if args.normalize:
        print("WARNING: --normalize flag passed but Raw PCA script forces NO normalization.")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("config requires model_name")

    quant = cfg.get("quant", "auto")
    per_axis = int(cfg.get("per_axis", 2000)) 
    batch_size = int(cfg.get("batch_size", 4))
    
    bank_path_str = args.bank_path
    
    print("=== 00_prepare_vectors_raw_pca.py ===")
    print(f"  model_name   : {model_name}")
    print(f"  per_axis     : {per_axis}")
    print(f"  bank_path    : {bank_path_str}")
    print("  Normalization: OFF (Raw)")
    print("  Filtering    : OFF (All Pairs)")

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

    # 4. Compute Vectors (Memory Optimized)
    final_axes: Dict[Tuple[int, str], np.ndarray] = {}
    
    for ax in AXES_CANON:
        print(f"\nProcessing Axis: {ax}")
        pairs = PAIRS[ax]
        
        # Accumulate DIFFERENCES on CPU
        diffs_agg = {L: [] for L in layer_indices}
        
        print("  Computing hidden states (Batch & Diff)...")
        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_highs = [p[0] for p in batch]
            batch_lows = [p[1] for p in batch]
            
            # Forward pass
            h_out = get_assistant_hidden_states(batch_highs)
            l_out = get_assistant_hidden_states(batch_lows)
            
            # Compute diff and move to CPU immediately
            for L in layer_indices:
                H = h_out[L] # (B, H) on GPU
                L_vec = l_out[L] # (B, H) on GPU
                
                # Raw Difference (No Normalization)
                D = H - L_vec # (B, H)
                
                # Move to CPU list
                diffs_agg[L].append(D.cpu())
            
            # Explicit delete to help cache clearing?
            del h_out
            del l_out
            # torch.cuda.empty_cache() # Calling this too often slows down, let allocator handle if possible
        
        # Concatenate and PCA per layer
        print("  Running PCA on accumulated differences...")
        for L in layer_indices:
            try:
                # Cat all batches: (N, H)
                Diffs_all = torch.cat(diffs_agg[L], dim=0).to(torch.float32).numpy()
            except Exception as e:
                print(f"    Error concatenating layer {L}: {e}")
                continue

            # Filtering: OFF
            # Normalization: OFF
            
            # PCA
            if Diffs_all.shape[0] < 2:
                vec = np.mean(Diffs_all, axis=0)
            else:
                pca = PCA(n_components=1)
                pca.fit(Diffs_all)
                vec = pca.components_[0]
            
            # Sign correction (align with mean diff direction)
            mean_vec = np.mean(Diffs_all, axis=0)
            if np.dot(vec, mean_vec) < 0:
                vec = -vec
                
            # Normalize ONLY the final steering vector to be unit length
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
    print(f"\n[Done] Saved Raw PCA axes bank to: {bank_path}")
    print(f"  Total: {len(npz_dict)} vectors")

if __name__ == "__main__":
    main()
