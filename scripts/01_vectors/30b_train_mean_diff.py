#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 30b_train_mean_diff.py
#
# Goals:
#  - Extract High/Low pairs from Big5Chat
#  - For each layer, compute Mean(High) and Mean(Low).
#  - Vector w = Mean(High) - Mean(Low).
#  - Midpoint m = (Mean(High) + Mean(Low)) / 2.
#  - Save normalized w and bias b = -w_norm · m.
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

from datasets import load_dataset

from persona_vectors.live_axes import (
    AXES as AXES_CANON,
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
)

def extract_big5_pairs_from_hf(per_axis: int = 1000) -> Dict[str, List[Tuple[str, str]]]:
    ds_all = load_dataset("wenkai-li/big5_chat")
    if isinstance(ds_all, dict):
        split_name = next(iter(ds_all.keys()))
        ds = ds_all[split_name]
    else:
        ds = ds_all

    buckets = defaultdict(lambda: {"high": [], "low": []})
    for ex in ds:
        tr_raw = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr_raw not in AXES_CANON or lv not in {"high", "low"}:
            continue
        orig_idx = ex.get("original_index")
        if orig_idx is None: continue
        to = (ex.get("train_output") or "").strip()
        if not to: continue
        buckets[(tr_raw, orig_idx)][lv].append(to)

    PAIRS = {ax: [] for ax in AXES_CANON}
    for (tr, orig_idx), d in buckets.items():
        if d["high"] and d["low"]:
            PAIRS[tr].append((d["high"][0], d["low"][0]))

    for ax in AXES_CANON:
        random.shuffle(PAIRS[ax])
        if per_axis > 0:
            PAIRS[ax] = PAIRS[ax][:per_axis]
        print(f"[big5chat-pairs] {ax}: {len(PAIRS[ax])} pairs")
    return PAIRS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--axis", type=str, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model_name")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "mean_diff_vectors.npz"

    target_axes = AXES_CANON if not args.axis else [args.axis]

    print("=== 30b_train_mean_diff.py ===")
    print(f"  Model : {model_name}")
    print(f"  Output: {bank_path}")

    model, tok = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    layers_stack, N_layers, _ = get_layer_stack(model)
    layer_indices = list(range(N_layers))
    device = _infer_main_device(model)
    model.eval()

    PAIRS = extract_big5_pairs_from_hf(per_axis=int(cfg.get("per_axis", 1000)))

    @torch.no_grad()
    def get_hidden_states(texts: List[str]):
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
            s_idx, e_idx = len_prefix, attn_mask[b].sum().item()
            if s_idx >= e_idx: s_idx = e_idx - 1
            for L in layer_indices:
                results[L].append(out.hidden_states[L][b][s_idx:e_idx].mean(dim=0))
        return {L: torch.stack(v) for L, v in results.items()}

    final_data: Dict[str, np.ndarray] = {}
    if bank_path.exists():
        existing = np.load(bank_path)
        final_data.update({k: existing[k] for k in existing.files})

    for ax in target_axes:
        print(f"\nProcessing Axis: {ax}")
        pairs = PAIRS[ax]
        batch_size = 4
        h_all = {L: [] for L in layer_indices}
        l_all = {L: [] for L in layer_indices}
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            h_out = get_hidden_states([p[0] for p in batch])
            l_out = get_hidden_states([p[1] for p in batch])
            for L in layer_indices:
                h_all[L].append(h_out[L].cpu())
                l_all[L].append(l_out[L].cpu())

        print("  Calculating Mean-Difference per layer...")
        for L in layer_indices:
            H = torch.cat(h_all[L], dim=0).to(torch.float32).numpy()
            L_ = torch.cat(l_all[L], dim=0).to(torch.float32).numpy()
            
            m_high = H.mean(axis=0)
            m_low = L_.mean(axis=0)
            midpoint = (m_high + m_low) / 2.0
            
            w = m_high - m_low
            norm = np.linalg.norm(w) + 1e-12
            w_norm = w / norm
            
            # Decision function: d(h) = w_norm · (h - midpoint) = w_norm · h - w_norm · midpoint
            b_norm = -np.dot(w_norm, midpoint)
            
            final_data[f"{L}|{ax}|w"] = w_norm.astype(np.float32)
            final_data[f"{L}|{ax}|b"] = np.array([b_norm], dtype=np.float32)
            final_data[f"{L}|{ax}|midpoint"] = midpoint.astype(np.float32)
            
        np.savez_compressed(bank_path, **final_data)
    print(f"\n[Done] Saved mean-diff vectors to {bank_path}.")

if __name__ == "__main__":
    main()
