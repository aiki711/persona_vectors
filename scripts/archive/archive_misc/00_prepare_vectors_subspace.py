#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 00_prepare_vectors_subspace.py
#
# Goals:
#  - Extract High-Low pairs from Big5Chat.
#  - [Subspace PCA] Keep top-k components explaining >95% variance (or at least 1).
#  - [Polarity Calibration] Automatically check and flip sign if vector is anti-steering.
#  - Memory Optimized.
#

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import ExitStack

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
    ResidualSteerer,  # Import ResidualSteerer for polarity check
    _format_prompt, # Helper for formatting prompts
    _ensure_pad_token,
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
#  Polarity Calibration Logic
# ==============================================

@torch.no_grad()
def check_polarity(
    model, tokenizer, vector: np.ndarray, layer: int, axis: str, 
    sample_pairs: List[Tuple[str, str]], device, alpha: float = 5.0
) -> float:
    """
    Check if vector effectively steers towards 'High' trait.
    Returns a score > 0 if correct polarity, < 0 if flipped.
    
    Method:
    1. Apply vector with +alpha.
    2. Compute log prob of High response vs Low response.
    3. Score = (logP(High) - logP(Low))_steered - (logP(High) - logP(Low))_base
       If score > 0, steering increased relative preference for High.
    """
    # Prepare prompt (dummy prompt since we just want to measure response likelihood)
    # Actually, Big5Chat pairs are (Response_High, Response_Low).
    # We need a prompt to precede them. "User: Hello.\nAssistant:" is standard.
    
    prompt = "Hello."
    formatted_prompt = _format_prompt(tokenizer, prompt)
    
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    
    # Pre-compute target token IDs for High/Low responses
    # We only check the first few tokens to save time, or the whole sequence perplexity?
    # Whole sequence perplexity is better.
    
    def get_log_prob(target_text, past_key_values=None):
        # This is tricky with steering. Steering happens during generation or forward pass.
        # We want P(target | prompt) with steering on prompt+target? 
        # Or just steering on prompt?
        # Standard steering: steer on all tokens.
        
        # Concatenate prompt + target
        full_text = formatted_prompt + target_text
        tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        
        # Create labels (ignore prompt part)
        labels = tokens.clone()
        prompt_len = input_ids.shape[1]
        labels[:, :prompt_len] = -100
        
        # Forward pass
        # If steering, we need the hook active.
        outputs = model(tokens, labels=labels)
        return -outputs.loss.item() # negative cross entropy = log likelihood (approx)

    # Base score
    diff_base = 0.0
    diff_steered = 0.0
    
    # We use a small subset for calibration
    LIMIT = 2
    subset = sample_pairs[:LIMIT]
    
    # 1. Base (No Steering)
    for high_txt, low_txt in subset:
        lp_h = get_log_prob(high_txt)
        lp_l = get_log_prob(low_txt)
        diff_base += (lp_h - lp_l)
        
    # 2. Steered (+alpha)
    # Register hook
    # v_mix needs to be shaped right. ResidualSteerer expects dim=H
    # vector is (H,) from PCA
    
    # Since we are doing causal LM, we can just wrap the model forward with the hook
    with ResidualSteerer(model, layer, vector, alpha, answer_only=False):
        for high_txt, low_txt in subset:
            lp_h = get_log_prob(high_txt)
            lp_l = get_log_prob(low_txt)
            diff_steered += (lp_h - lp_l)
            
    # Result
    # Improvement = Steered_Diff - Base_Diff
    # If Improvement > 0, then +alpha made High more likely relative to Low compared to base.
    improvement = diff_steered - diff_base
    return improvement


# ==============================================
#  Main Logic: Subspace PCA
# ==============================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML config path")
    ap.add_argument("--bank_path", "-b", required=True, help="Axes bank path (output)")
    ap.add_argument("--model_name", "-m", help="Override config model_name")
    ap.add_argument("--variance_threshold", type=float, default=0.95, help="Variance threshold for subspace (default: 0.95)")
    ap.add_argument("--axis", type=str, default=None, help="Process only specific axis (e.g. extraversion)")
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
    per_axis = int(cfg.get("per_axis", 2000)) 
    batch_size = int(cfg.get("batch_size", 4))
    
    bank_path_str = args.bank_path
    
    # Filter Axes if requested
    target_axes = AXES_CANON
    if args.axis:
        if args.axis not in AXES_CANON:
             print(f"Warning: Axis '{args.axis}' not in canonical list. Using canonical list.")
        else:
             target_axes = [args.axis]
             
    print("=== 00_prepare_vectors_subspace.py ===")
    print(f"  model_name   : {model_name}")
    print(f"  per_axis     : {per_axis}")
    print(f"  bank_path    : {bank_path_str}")
    print(f"  Variance Thr : {args.variance_threshold}")
    print(f"  Target Axes  : {target_axes}")
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
            # Find the start of assistant response
            # Simple heuristic: prefix length. 
            # Note: prompts might vary if chat template varies or if left padding exists?
            # Here we left-padded logic? No, pad_sequence is right-padded by default usually for batch_first=True unless specified.
            # But wait, pad_sequence pads at the END.
            
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
    subspace_bases: Dict[Tuple[int, str], np.ndarray] = {} # For saving raw bases if needed
    
    bank_path = Path(bank_path_str)
    if bank_path.exists():
        print(f"Loading existing bank from {bank_path} to resume...")
        existing_data = np.load(bank_path)
        for k in existing_data.files:
            try:
                ls_str, axis = k.split("|")
                final_axes[(int(ls_str), axis)] = existing_data[k]
            except ValueError:
                continue
        print(f"  Loaded {len(final_axes)} existing vectors.")

    for ax in target_axes:
        if any(a == ax for (L, a) in final_axes.keys()):
            print(f"\nSkipping Axis: {ax} (Already exists in bank)")
            continue

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
            # torch.cuda.empty_cache() 
        
        # Concatenate and Subspace PCA per layer
        print("  Running Subspace PCA on accumulated differences...")
        for L in layer_indices:
            try:
                # Cat all batches: (N, H)
                Diffs_all = torch.cat(diffs_agg[L], dim=0).to(torch.float32).numpy()
            except Exception as e:
                print(f"    Error concatenating layer {L}: {e}")
                continue

            # Subspace PCA
            # We want to keep enough components to explain variance_threshold (e.g. 0.95)
            # But standard implementation: n_components < 1 means ratio variance.
            # If n_components >= 1, it means count.
            
            n_samples = Diffs_all.shape[0]
            if n_samples < 2:
                vec = np.mean(Diffs_all, axis=0)
                # Just one component
                norm = np.linalg.norm(vec) + 1e-12
                final_axes[(L, ax)] = (vec / norm).astype(np.float32)
                continue
                
            pca = PCA(n_components=args.variance_threshold) # e.g. 0.95
            pca.fit(Diffs_all)
            
            # Number of components kept
            n_comps = pca.n_components_
            
            # Purified Vector: Reconstruct or just take the main direction?
            # "Reconstruct the purified direction" often means the first principal component
            # represents the shared direction (signal), and later components are noise if we assume rank-1 hypothesis.
            # However, "Low-Rank Subspace" implies the signal might be multidimensional.
            # But for simple steering, we need 1 vector.
            # Strategy: Take the first component (PC1) as the primary steering vector.
            # Or: Take the weighted sum of components? No, PCA separates them. PC1 is the strongest direction.
            # We will save PC1 as the main vector.
            
            vec = pca.components_[0] # Shape (H,)
            
            # --- Polarity Calibration ---
            print(f"    [L{L}] Calibration... (PC1 explained var: {pca.explained_variance_ratio_[0]:.4f})")
            
            # Use top 5 pairs for quick check
            # We need to temporarily re-enable gradient maybe? No, 'check_polarity' is no_grad.
            
            polarity_score = check_polarity(
                model, tok, vec, L, ax,
                sample_pairs=pairs[:5], 
                device=device, 
                alpha=5.0
            )
            
            if polarity_score < 0:
                print(f"      -> Flipped sign (score: {polarity_score:.4f})")
                vec = -vec
            else:
                print(f"      -> Sign OK (score: {polarity_score:.4f})")
                
            # Normalize ONLY the final steering vector to be unit length
            norm = np.linalg.norm(vec) + 1e-12
            vec_unit = vec / norm
            
            final_axes[(L, ax)] = vec_unit.astype(np.float32)
            
            # Save all bases if needed later (e.g. for orthogonal projection)
            # shape (k, H)
            subspace_bases[(L, ax)] = pca.components_.astype(np.float32)

        # Save incrementally
        # Actually it's simpler to just save all current final_axes
        npz_current = { f"{L}|{a}": v for (L, a), v in final_axes.items() }
        np.savez_compressed(bank_path_str, **npz_current)
        print(f"  [Progress] Saved current bank to: {bank_path_str}")

    print(f"\n[Done] All axes processed. Final total: {len(final_axes)} vectors.")
    
    # Save subspace
    subspace_path = bank_path.with_name(bank_path.stem + "_subspace.npz")
    subspace_dict = {
        f"{L}|{ax}": bases
        for (L, ax), bases in subspace_bases.items()
    }
    np.savez_compressed(subspace_path, **subspace_dict)
    print(f"  Saved subspace bases to: {subspace_path}")

if __name__ == "__main__":
    main()
