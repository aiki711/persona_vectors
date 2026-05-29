#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 30_train_boundary.py
#
# Goals:
#  - Extract High/Low pairs from Big5Chat (similar to 00_prepare_vectors_subspace.py)
#  - For each layer, train a Linear SVM (or Logistic Regression) to classify High vs Low.
#  - Save the learned normal vectors (w) and biases (b).
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

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

from persona_vectors.live_axes import (
    AXES as AXES_CANON,
    load_model_and_tokenizer,
    _infer_main_device,
    _is_bnb_quantized,
    get_layer_stack,
)

def extract_big5_pairs_from_hf(per_axis: int = 1000) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract (high, low) response pairs from Big5Chat dataset.
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
    ap.add_argument("--out_dir", required=True, help="Output directory for vectors")
    ap.add_argument("--model_name", "-m", help="Override config model_name")
    ap.add_argument("--axis", type=str, default=None, help="Process only specific axis (e.g. extraversion)")
    ap.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg.get("model_name")
    if not model_name:
        raise ValueError("config requires model_name")

    quant = cfg.get("quant", "auto")
    per_axis = int(cfg.get("per_axis", 2000)) 
    batch_size = int(cfg.get("batch_size", 4))
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "boundary_vectors.npz"

    target_axes = AXES_CANON
    if args.axis:
        if args.axis not in AXES_CANON:
             print(f"Warning: Axis '{args.axis}' not in canonical list. Using canonical.")
        else:
             target_axes = [args.axis]

    print("=== 30_train_boundary.py ===")
    print(f"  Model        : {model_name}")
    print(f"  Pairs/Axis   : {per_axis}")
    print(f"  Output       : {bank_path}")
    print(f"  Target Axes  : {target_axes}")
    print(f"  SVM C param  : {args.C}")

    seed = int(cfg.get("seed", 2025))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n[Step 1] Loading model/tokenizer...")
    model, tok = load_model_and_tokenizer(model_name, quant=quant)
    
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        
    if not tok.chat_template:
        tok.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

    layers_stack, N_layers, kind = get_layer_stack(model)
    layer_indices = list(range(N_layers))
    device = _infer_main_device(model)
    
    model.eval()

    print("\n[Step 2] Loading Big5Chat pairs...")
    PAIRS = extract_big5_pairs_from_hf(per_axis=per_axis)

    @torch.no_grad()
    def get_assistant_hidden_states(texts: List[str]) -> Dict[int, torch.Tensor]:
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
                hidden = out.hidden_states[L][b]
                valid_hidden = hidden[start_idx:end_idx]
                pooled = valid_hidden.mean(dim=0)
                results[L].append(pooled)
                
        for L in layer_indices:
            results[L] = torch.stack(results[L])
            
        return results

    final_boundary: Dict[str, np.ndarray] = {}
    
    if bank_path.exists():
        print(f"Loading existing bank from {bank_path} to resume...")
        existing_data = np.load(bank_path)
        for k in existing_data.files:
            final_boundary[k] = existing_data[k]
        print(f"  Loaded existing data keys.")

    for ax in target_axes:
        # Check if we already have it
        if any(k.endswith(f"|{ax}|w") for k in final_boundary.keys()):
            print(f"\nSkipping Axis: {ax} (Already exists)")
            continue

        print(f"\nProcessing Axis: {ax}")
        pairs = PAIRS[ax]
        
        h_agg = {L: [] for L in layer_indices} # High
        l_agg = {L: [] for L in layer_indices} # Low
        
        print("  Computing hidden states (Batch)...")
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_highs = [p[0] for p in batch]
            batch_lows = [p[1] for p in batch]
            
            h_out = get_assistant_hidden_states(batch_highs)
            l_out = get_assistant_hidden_states(batch_lows)
            
            for L in layer_indices:
                h_agg[L].append(h_out[L].cpu())
                l_agg[L].append(l_out[L].cpu())

        print("  Training Linear SVM per layer...")
        for L in layer_indices:
            try:
                H_all = torch.cat(h_agg[L], dim=0).to(torch.float32).numpy()
                L_all = torch.cat(l_agg[L], dim=0).to(torch.float32).numpy()
            except Exception as e:
                print(f"    Error concatenating layer {L}: {e}")
                continue

            # X features shape (2N, H), y labels shape (2N,)
            # High = 1, Low = 0
            X = np.vstack([H_all, L_all])
            y = np.concatenate([np.ones(H_all.shape[0]), np.zeros(L_all.shape[0])])

            # SVM requires scaled data for good performance (optional but recommended)
            # However, for steering we usually want the exact decision boundary in the 
            # original space. So we will not scale, or we scale and then unscale the weights.
            # We'll stick to raw activations for simplicity of the distance calculation during inference.
            
            # Use dual=False when n_samples > n_features, but here n_features (4096) > n_samples (e.g. 2000)
            clf = LinearSVC(C=args.C, dual="auto", max_iter=2000, random_state=seed)
            clf.fit(X, y)
            
            acc = clf.score(X, y)
            
            w = clf.coef_[0] # (H,)
            b = clf.intercept_[0] # scalar
            
            # The distance from hyperplane is w·x + b.
            # SVM predicts 1 (High) if w·x + b > 0.
            # We want to steer towards High, so the vector is w.
            # Let's normalize w to make 1-unit step size consistent across axes/layers.
            norm = np.linalg.norm(w) + 1e-12
            w_norm = w / norm
            
            # Adjust b accordingly so distance is preserved? 
            # Original distance: D = w·x + b
            # Normalized distance: D' = w_norm·x + b_norm
            # b_norm = b / norm
            b_norm = b / norm
            
            final_boundary[f"{L}|{ax}|w"] = w_norm.astype(np.float32)
            final_boundary[f"{L}|{ax}|b"] = np.array([b_norm], dtype=np.float32)
            
            if L % 5 == 0 or L == N_layers - 1:
                print(f"    [L{L:02d}] SVM Train Acc: {acc:.4f} | ||w||={norm:.4f}")

        # SaveINCREMENTALLY
        np.savez_compressed(bank_path, **final_boundary)
        print(f"  [Progress] Saved current bank.")

    print(f"\n[Done] Saved boundary vectors to {bank_path}.")

if __name__ == "__main__":
    main()
