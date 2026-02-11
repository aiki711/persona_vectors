#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 00_prepare_vectors_pca.py (Big5Chat Pairwise PCA Edition)
#
# Goals:
#  - Use Big5Chat dataset to get (High - Low) difference vectors.
#  - Use PCA (First Component) instead of Mean to extract the steering direction.
#  - Use robust chat template handling to identify assistant tokens, avoiding <asst> dependency.
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
    Note: Texts here are raw "train_output" (the assistant's response).
    We will format them with chat templates later.
    """
    ds_all = load_dataset("wenkai-li/big5_chat")
    if isinstance(ds_all, dict):
        # usually 'train'
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

        # Store raw output, do not prepend <asst> here
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
#  Main Logic: PCA on Pairwise Differences
# ==============================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML config path")
    ap.add_argument("--bank_path", "-b", required=True, help="Axes bank path (output)")
    ap.add_argument("--model_name", "-m", help="Override config model_name")
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
    per_axis = int(cfg.get("per_axis", 1000))
    batch_size = int(cfg.get("batch_size", 4)) # slightly smaller due to full chat context
    # pooling strategy: "last", "mean", "pca_all" ?
    # For this script we will use "mean" of assistant tokens
    pooling_type = "mean" 
    
    bank_path_str = args.bank_path
    
    # Generic user prompt for context
    # Typically Big5Chat doesn't have the user prompt in the dataset easily accessible in 'train_output'
    # The dataset has 'instruction' but it varies.
    # To reduce variance, we can use a generic prompt or try to use the dataset's instruction if available.
    # The dataset has 'instruction' field. Let's load it too.
    # Re-writing extract to include instruction would be better, but for now let's use a fixed generic prompt
    # to ensure the steering vector captures the *response style* difference, not input difference.
    # Actually, if we use fixed prompt "Hello", the model might not generate the high/low text naturally.
    # But we are forcing the model to generate (or rather, evaluating the hidden states of) the specific high/low text.
    # So the input prompt serves as context. A neutral prompt is best.
    generic_prompt = "Act as a sophisticated AI assistant. Please respond to the following user message: 'Hello, how are you?'"
    
    print("=== 00_prepare_vectors_pca.py (Big5Chat Pairwise PCA) ===")
    print(f"  model_name : {model_name}")
    print(f"  quant      : {quant}")
    print(f"  per_axis   : {per_axis}")
    print(f"  batch_size : {batch_size}")
    print(f"  bank_path  : {bank_path_str}")

    seed = int(cfg.get("seed", 2025))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load Model
    print("\n[Step 1] Loading model/tokenizer...")
    model, tok = load_model_and_tokenizer(model_name, quant=quant)
    
    # Ensure pad token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        
    # Ensure chat template exists (for Base models)
    if not tok.chat_template:
        print("Warning: No chat template found. Using default ChatML-like template.")
        tok.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    layers_stack, N_layers, kind = get_layer_stack(model)
    layer_indices = list(range(N_layers))
    H_dim = model.config.hidden_size
    device = _infer_main_device(model)
    if _is_bnb_quantized(model):
        model.eval()
    else:
        model.to(device).eval()

    # 2. Load Data
    print("\n[Step 2] Loading Big5Chat pairs...")
    PAIRS = extract_big5_pairs_from_hf(per_axis=per_axis)

    # 3. Helper to format and compute hidden states
    @torch.no_grad()
    def get_assistant_hidden_states(texts: List[str]) -> Dict[int, torch.Tensor]:
        """
        Process a batch of assistant response texts.
        We pretend the user said something generic, and the assistant says 'text'.
        We format this using chat template, identify the assistant part, and pool it.
        Returns: {L: Tensor(B, H)}
        """
        # Prepare inputs
        # We need to find where the assistant text starts.
        # Strategy:
        # 1. Tokenize "User: ..." (prefix)
        # 2. Tokenize "User: ... \n Assistant: <text>" (full)
        # 3. Diff is assistant part.
        
        # Note: apply_chat_template handles special tokens.
        # But we need to do this per sample if lengths vary wildly, but here we batch.
        # For batching, tricky if prefix length varies (it shouldn't if we use fixed prompt).
        
        msgs_prefix = [{"role": "user", "content": "Hello."}]
        # We'll use a very short prompt to save context.
        
        # Encode prefix once to get length (assuming deterministic length)
        # Note: Some tokenizers add BOS at start.
        prefix_ids = tok.apply_chat_template(msgs_prefix, add_generation_prompt=True, tokenize=True)
        len_prefix = len(prefix_ids)
        
        # Prepare batch input
        full_inputs = []
        for t in texts:
            msgs = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": t}]
            # tokenize=False then tok(..., padding=True) is safer for batching
            # BUT apply_chat_template(tokenize=False) returns string, which we then need to tokenize.
            # If we do that, we rely on string matching to find the split? No, dangerous.
            
            # Better: apply_chat_template(tokenize=True) for each, then pad manually?
            # Or just use the string text and rely on the fact that we know the prefix string.
            # Let's use the ID length method per sample.
            full_ids = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
            full_inputs.append(torch.tensor(full_ids))
            
        # Pad
        # pad direction? usually right for causal LM if we want position IDs to align?
        # Actually left padding is common for generation, but here we just process.
        # Right padding is easiest for indexing if we mask the pad.
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(full_inputs, batch_first=True, padding_value=tok.pad_token_id).to(device)
        attn_mask = (input_ids != tok.pad_token_id).long()
        
        # Forward
        out = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        results = {L: [] for L in layer_indices}
        
        # Pool per sample
        for b in range(len(texts)):
            # Find valid tokens for this sample
            # The assistant tokens start after 'len_prefix' ?
            # Warn: apply_chat_template might change prefix tokens if followed by assistant?
            # Usually strict standard templates don't interact across boundaries much.
            # We will assume start index is len_prefix.
            
            # Cap start index to avoid error if response is empty (unlikely)
            start_idx = len_prefix
            end_idx = attn_mask[b].sum().item() # non-pad length
            
            if start_idx >= end_idx:
                # Fallback: use last token
                start_idx = end_idx - 1
            
            # Extract hidden states for relevant layers
            # We want [start_idx : end_idx]
            for L in layer_indices:
                hidden = out.hidden_states[L][b] # (T, H)
                valid_hidden = hidden[start_idx:end_idx]
                
                # Mean pooling
                pooled = valid_hidden.mean(dim=0) # (H,)
                results[L].append(pooled)
                
        # Stack
        for L in layer_indices:
            results[L] = torch.stack(results[L]) # (B, H)
            
        return results


    # 4. Compute Vectors per Layer/Trait
    final_axes: Dict[Tuple[int, str], np.ndarray] = {}
    
    for ax in AXES_CANON:
        print(f"\nProcessing Axis: {ax}")
        pairs = PAIRS[ax]
        
        # Store all difference vectors per layer
        # {L: [ (H,), (H,), ... ]}
        diff_vecs = {L: [] for L in layer_indices}
        
        # Batch loop
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            highs = [p[0] for p in batch]
            lows = [p[1] for p in batch]
            
            # Forward High
            # NOTE: We can optimize by batching high/low together, but careful with memory.
            # Let's do high batch then low batch, or mix (2*B).
            # Mix is better to save forward passes if memory allows.
            
            all_texts = []
            for h, l in zip(highs, lows):
                all_texts.append(h)
                all_texts.append(l)
                
            # Forward
            # This calls model(). might OOM if B is large. B=4 -> 8 seqs. OK.
            states = get_assistant_hidden_states(all_texts)
            
            # Compute diffs
            for L in layer_indices:
                layer_hs = states[L] # (2B, H)
                # Reshape to (B, 2, H) -> (B, high, H) - (B, low, H)
                layer_hs = layer_hs.view(len(batch), 2, H_dim)
                diffs = layer_hs[:, 0, :] - layer_hs[:, 1, :] # (B, H)
                
                # Accumulate
                # cpu to save gpu mem?
                diff_vecs[L].append(diffs.cpu())

        # PCA per Layer
        print(f"  Computing PCA for {ax}...")
        for L in layer_indices:
            # Concat all diffs -> (N_pairs, H)
            if not diff_vecs[L]:
                print(f"Warning: No data for {ax} L{L}")
                continue
                
            X = torch.cat(diff_vecs[L], dim=0).float().numpy() # (N, H)
            
            # PCA logic
            # We want the first component.
            # Identify direction of max variance.
            # Note: The sign of PCA component is arbitrary.
            # For "mean", we did (High - Low), so direction is roughly High-wards.
            # PCA might flip it.
            # Solution: Enforce correlation with the Mean vector to correct sign.
            
            # 1. Compute Mean
            mean_vec = np.mean(X, axis=0) # (H,)
            
            # 2. PCA
            if X.shape[0] < 2:
                # fallback
                vec = mean_vec
            else:
                pca = PCA(n_components=1)
                pca.fit(X)
                vec = pca.components_[0] # (H,)
            
            # 3. Sign Flip Check
            # If dot(vec, mean_vec) < 0, flip vec
            if np.dot(vec, mean_vec) < 0:
                vec = -vec
                
            # Normalize (unit vector)
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
    print(f"\n[Done] Saved PCA axes bank to: {bank_path}")
    print(f"  Total: {len(npz_dict)} vectors")

if __name__ == "__main__":
    main()
