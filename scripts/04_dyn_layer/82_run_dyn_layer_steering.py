#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 82_run_dyn_layer_proj_prior.py
#
# Dynamic Layer Selection using:
#   1. Dot Product Projection (Approach 1)
#   2. Data-Driven Layer Priors (Approach 3)
#
# Real-time selection during generation without requiring extra forward passes or z-score stats.
#

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

LAYERS = list(range(32))
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)

def extract_positive_texts(axis, limit=30):
    print("Loading Big5Chat dataset...")
    ds_all = load_dataset("wenkai-li/big5_chat")
    split_name = next(iter(ds_all.keys()))
    ds = ds_all[split_name]
    
    texts = []
    for ex in ds:
        tr = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr == axis and lv == "high":
            to = (ex.get("train_output") or "").strip()
            if to:
                texts.append(to)
                if len(texts) >= limit:
                    break
    print(f"Extracted {len(texts)} positive texts for {axis}.")
    return texts


def select_layer_proj_prior(model, input_ids, layer_w_dev, target_direction, layer_priors, score_mode="cosine", h_pos_dict=None, sims_ref_dict=None, alpha=1.0, layer_midpoint_dev=None, norm_mode="raw_norm", probe_masks=None):
    stack, _, _ = get_layer_stack(model)

    if score_mode == "logit_diff":
        # Bhandari et al. Method: Maximize change in last-token logits.
        with torch.no_grad():
            out = model(input_ids)
        base_logits = out.logits[0, -1, :].float()

        raw_scores = {}
        for L, w_dev in layer_w_dev.items():
            def hook(mod, inp, out_val):
                hs = out_val[0] if isinstance(out_val, tuple) else out_val
                if not torch.isfinite(hs).all(): return out_val
                hs_f32 = hs.to(torch.float32)
                if norm_mode == "relative":
                    h_norm = torch.norm(hs_f32, p=2, dim=-1, keepdim=True)
                    steered = hs_f32 + alpha * w_dev.view(1, 1, -1) * h_norm
                else:
                    steered = hs_f32 + alpha * w_dev.view(1, 1, -1)
                return (steered.to(hs.dtype), *out_val[1:]) if isinstance(out_val, tuple) else steered.to(hs.dtype)

            handle = stack[L].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    out_steered = model(input_ids)
                steered_logits = out_steered.logits[0, -1, :].float()
                raw_scores[L] = (steered_logits - base_logits).norm().item()
            finally:
                handle.remove()
    else:
        saved_h = {}
        handles = []
        for L in layer_w_dev.keys():
            def get_hook(L_idx):
                def hook(mod, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    saved_h[L_idx] = hs[0, -1, :].detach().float()
                return hook
            handles.append(stack[L].register_forward_hook(get_hook(L)))

        try:
            with torch.no_grad():
                _ = model(input_ids)
        finally:
            for h in handles:
                h.remove()

        raw_scores = {}
        for L, w_dev in layer_w_dev.items():
            h = saved_h[L]
            m = layer_midpoint_dev.get(L, None) if layer_midpoint_dev is not None else None
            mask = probe_masks.get(L, None) if probe_masks is not None else None

            if score_mode == "local_proj_rank":
                if h_pos_dict is not None and "H_pos" in h_pos_dict and L in h_pos_dict["H_pos"] and m is not None:
                    # Keep variables on GPU device of h
                    H_pos = torch.tensor(h_pos_dict["H_pos"][L], dtype=torch.float32, device=h.device) # [1000, dim]
                    
                    if mask is not None:
                        if mask.dtype == torch.bool:
                            H_pos = H_pos[:, mask]
                            m_val = m[mask]
                            h_val = h[mask]
                        else:
                            H_pos = H_pos * mask.unsqueeze(0)
                            m_val = m * mask
                            h_val = h * mask
                    else:
                        m_val = m
                        h_val = h
                        
                    # Calculate individual steering vectors and normalize them
                    W_dir = H_pos - m_val.unsqueeze(0) # [1000, dim]
                    W_norm = torch.norm(W_dir, p=2, dim=-1, keepdim=True) # [1000, 1]
                    W_unit = W_dir / (W_norm + 1e-10) # [1000, dim]
                    p_pos = W_norm.squeeze(-1) # [1000]
                    
                    # Calculate p_h
                    h_dev = h_val - m_val # [dim]
                    p_h = torch.matmul(W_unit, h_dev) # [1000]
                    
                    # Calculate score
                    score = 1.0 - (p_h >= p_pos).float().mean().item()
                else:
                    if mask is not None:
                        h_masked = h * mask
                        w_dev_masked = w_dev * mask
                        h_unit = h_masked / (torch.norm(h_masked) + 1e-10)
                        w_unit = w_dev_masked / (torch.norm(w_dev_masked) + 1e-10)
                    else:
                        h_unit = h / (torch.norm(h) + 1e-10)
                        w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
            elif score_mode == "proj_rank":
                if h_pos_dict is not None and "H_pos" in h_pos_dict and L in h_pos_dict["H_pos"] and m is not None:
                    # Keep variables on GPU device of h
                    H_pos = torch.tensor(h_pos_dict["H_pos"][L], dtype=torch.float32, device=h.device) # [1000, dim]
                    
                    if mask is not None:
                        if mask.dtype == torch.bool:
                            H_pos = H_pos[:, mask]
                            m_val = m[mask]
                            h_val = h[mask]
                        else:
                            H_pos = H_pos * mask.unsqueeze(0)
                            m_val = m * mask
                            h_val = h * mask
                    else:
                        m_val = m
                        h_val = h
                        
                    # Deviations from midpoint
                    d_pos = H_pos - m_val.unsqueeze(0) # [1000, dim]
                    w_avg = d_pos.mean(dim=0) # [dim]
                    d_h = h_val - m_val # [dim]
                    
                    # Normalize
                    d_pos_norm = d_pos / (torch.norm(d_pos, p=2, dim=-1, keepdim=True) + 1e-10) # [1000, dim]
                    w_avg_norm = w_avg / (torch.norm(w_avg, p=2, dim=-1) + 1e-10) # [dim]
                    d_h_norm = d_h / (torch.norm(d_h, p=2, dim=-1) + 1e-10) # [dim]
                    
                    # Calculate similarities
                    S_i = torch.matmul(d_pos_norm, d_h_norm) # [1000]
                    S_center = torch.dot(w_avg_norm, d_h_norm).item() # scalar
                    
                    # Calculate score
                    percentile = (S_center >= S_i).float().mean().item()
                    score = 1.0 - percentile
                else:
                    if mask is not None:
                        h_masked = h * mask
                        w_dev_masked = w_dev * mask
                        h_unit = h_masked / (torch.norm(h_masked) + 1e-10)
                        w_unit = w_dev_masked / (torch.norm(w_dev_masked) + 1e-10)
                    else:
                        h_unit = h / (torch.norm(h) + 1e-10)
                        w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
            elif score_mode == "proj_cosine":
                if m is not None:
                    if mask is not None:
                        h_dev = (h - m) * mask
                        w_unit = w_dev * mask
                        w_unit = w_unit / (torch.norm(w_unit) + 1e-10)
                    else:
                        h_dev = h - m
                        w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
                    h_dev_unit = h_dev / (torch.norm(h_dev) + 1e-10)
                    score = -torch.dot(h_dev_unit, w_unit).item()
                else:
                    if mask is not None:
                        h_masked = h * mask
                        w_dev_masked = w_dev * mask
                        h_unit = h_masked / (torch.norm(h_masked) + 1e-10)
                        w_unit = w_dev_masked / (torch.norm(w_dev_masked) + 1e-10)
                    else:
                        h_unit = h / (torch.norm(h) + 1e-10)
                        w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
            elif score_mode == "rank":
                if h_pos_dict is not None and "H_pos" in h_pos_dict and L in h_pos_dict["H_pos"] and m is not None:
                    # Keep variables on GPU device of h
                    H_pos = torch.tensor(h_pos_dict["H_pos"][L], dtype=torch.float32, device=h.device) # [1000, dim]
                    c = H_pos.mean(dim=0) # [dim]
                    
                    if mask is not None:
                        if mask.dtype == torch.bool:
                            H_pos = H_pos[:, mask]
                            c = c[mask]
                            h_val = h[mask]
                        else:
                            H_pos = H_pos * mask.unsqueeze(0)
                            c = c * mask
                            h_val = h * mask
                    else:
                        h_val = h
                        
                    # Normalize
                    H_pos_norm = H_pos / (torch.norm(H_pos, p=2, dim=-1, keepdim=True) + 1e-10) # [1000, dim]
                    c_norm = c / (torch.norm(c, p=2, dim=-1) + 1e-10) # [dim]
                    h_norm = h_val / (torch.norm(h_val, p=2, dim=-1) + 1e-10) # [dim]
                    
                    # Calculate similarities
                    S_i = torch.matmul(H_pos_norm, h_norm) # [1000]
                    S_center = torch.dot(c_norm, h_norm).item() # scalar
                    
                    # Calculate score
                    percentile = (S_center >= S_i).float().mean().item()
                    score = 1.0 - percentile
                else:
                    if mask is not None:
                        h_masked = h * mask
                        w_dev_masked = w_dev * mask
                        h_unit = h_masked / (torch.norm(h_masked) + 1e-10)
                        w_unit = w_dev_masked / (torch.norm(w_dev_masked) + 1e-10)
                    else:
                        h_unit = h / (torch.norm(h) + 1e-10)
                        w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
            else: # cosine
                if m is not None:
                    if mask is not None:
                        h_masked = h * mask
                        m_masked = m * mask
                        h_unit = h_masked / (torch.norm(h_masked) + 1e-10)
                        m_unit = m_masked / (torch.norm(m_masked) + 1e-10)
                    else:
                        h_unit = h / (torch.norm(h) + 1e-10)
                        m_unit = m / (torch.norm(m) + 1e-10)
                    score = torch.dot(h_unit, m_unit).item()
                else:
                    if mask is not None:
                        h_masked = h * mask
                        w_dev_masked = w_dev * mask
                        h_unit = h_masked / (torch.norm(h_masked) + 1e-10)
                        w_unit = w_dev_masked / (torch.norm(w_dev_masked) + 1e-10)
                    else:
                        h_unit = h / (torch.norm(h) + 1e-10)
                        w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()

            if target_direction == "high":
                if score_mode in ["rank", "proj_rank", "proj_cosine", "local_proj_rank"]:
                    raw_scores[L] = score
                else:
                    raw_scores[L] = score # Maximize cosine similarity with midpoint
            else:
                if score_mode in ["rank", "proj_rank", "proj_cosine", "local_proj_rank"]:
                    raw_scores[L] = -score
                else:
                    raw_scores[L] = -score

    final_scores = {}
    for L in layer_w_dev.keys():
        w_prior = layer_priors.get(L, 0.0)
        if w_prior > 1e-5:
            final_scores[L] = raw_scores[L] - (1.0 - w_prior) * 10.0
        else:
            final_scores[L] = -1e9  # Exclude masked layers completely

    best_layer = max(final_scores, key=lambda L: final_scores[L])
    return best_layer, raw_scores, final_scores

def generate_with_steered_layer(model, tokenizer, prompt, w_dev, alpha, layer, max_new_tokens=150, norm_mode="raw_norm"):
    device = _infer_main_device(model)
    inputs = format_and_tokenize(tokenizer, prompt, device)
    stack, _, _ = get_layer_stack(model)

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not torch.isfinite(hs).all() or hs.size(1) != 1: return out
        hs_f32 = hs.to(torch.float32)
        if norm_mode == "relative":
            h_norm = torch.norm(hs_f32, p=2, dim=-1, keepdim=True)
            steered = hs_f32 + alpha * w_dev.view(1, 1, -1) * h_norm
        else:
            steered = hs_f32 + alpha * w_dev.view(1, 1, -1)
        return (steered.to(hs.dtype), *out[1:]) if isinstance(out, tuple) else steered.to(hs.dtype)

    handle = stack[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
    finally:
        handle.remove()

    prompt_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True), outputs[0]

@torch.no_grad()
def calc_ppl(model, ids):
    out = model(ids.unsqueeze(0), labels=ids.clone().unsqueeze(0))
    return torch.exp(out.loss).item()

def main():
    # Login node execution guard to prevent server overload
    import socket
    import sys
    hostname = socket.gethostname()
    if "hakusan" in hostname:
        print(f"\n[ERROR] This heavy DLS execution script cannot be run directly on the login node '{hostname}'.")
        print("Please submit this script as a SLURM job using sbatch to run it on a compute node.")
        sys.exit(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", required=True)
    ap.add_argument("--vector_bank",  required=True)
    ap.add_argument("--prompts",      required=True)
    ap.add_argument("--input_dir",    default="exp_steering_layer_analysis/results", help="単層スイープ結果ディレクトリ")
    ap.add_argument("--out_dir",      default="exp_steering_dyn_layer_proj_prior/results")
    ap.add_argument("--axis",         type=str, default="extraversion")
    ap.add_argument("--alpha",        type=float, required=True)
    ap.add_argument("--direction",    type=str, choices=["high", "low"], default="high")
    ap.add_argument("--norm_mode",    type=str, choices=["none", "midpoint", "raw_norm", "relative"], default="raw_norm",
                    help="Scaling mode for steering vectors. raw_norm scales by the original difference vector's norm.")
    ap.add_argument("--score_mode",   type=str, choices=["cosine", "rank", "logit_diff", "proj_rank", "proj_cosine", "local_proj_rank"], default="cosine", help="layer selection score mode")
    ap.add_argument("--mask_bank",    default="", help="Path to probe masks bank (.npz)")
    ap.add_argument("--num_prompts",  type=int, default=10, help="Number of prompts to evaluate")
    ap.add_argument("--seed",         type=int, default=42, help="Random seed for generation reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    direction_mult = 1.0 if args.direction == "high" else -1.0
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_mode == "cosine":
        method_name = "cos_only"
    elif args.score_mode == "logit_diff":
        method_name = "logit_diff"
    elif args.score_mode == "proj_rank":
        method_name = "proj_rank_only"
    elif args.score_mode == "proj_cosine":
        method_name = "proj_cos_only"
    elif args.score_mode == "local_proj_rank":
        method_name = "local_proj_rank_only"
    else: # rank
        method_name = "rank_only"
        
    if args.mask_bank:
        method_name = "masked_" + method_name

    out_file = out_dir / f"{method_name}_Val{args.alpha}.jsonl"
    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # Load layer priors
    print("Bypassing layer priors (enforcing 4-29 candidate layer hard mask)...")
    layer_priors = {L: 1.0 if 4 <= L <= 29 else 0.0 for L in LAYERS}

    # Load vectors
    v_data = np.load(args.vector_bank)
    layer_w = {}
    layer_midpoint = {}
    for L in LAYERS:
        w_key = f"{L}|{args.axis}|w"
        raw_norm_key = f"{L}|{args.axis}|raw_norm"
        mp_key = f"{L}|{args.axis}|midpoint"
        if w_key in v_data:
            w_vec = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult
            
            # Scale using original raw norm of difference vector (raw_norm)
            if args.norm_mode in ["midpoint", "raw_norm"]:
                if raw_norm_key in v_data:
                    r_norm = float(v_data[raw_norm_key][0])
                    w_norm = torch.norm(w_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * r_norm
                elif mp_key in v_data:
                    # Fallback to midpoint norm if raw_norm is not present in older vector banks
                    m_vec = torch.tensor(v_data[mp_key], dtype=torch.float32)
                    w_norm = torch.norm(w_vec).item()
                    m_norm = torch.norm(m_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * m_norm
            elif args.norm_mode == "relative":
                w_norm = torch.norm(w_vec).item()
                w_vec = w_vec / (w_norm + 1e-10)
            layer_w[L] = w_vec
        if mp_key in v_data:
            layer_midpoint[L] = torch.tensor(v_data[mp_key], dtype=torch.float32)

    if not layer_w:
        return print("[ERROR] No layer vectors found.")

    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"): continue
            if line.endswith(","): line = line[:-1]
            try: item = json.loads(line)
            except: item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
    prompts = prompts[:args.num_prompts]

    print(f"=== Proj & Prior DLS Execution ===")
    print(f"  Axis  : {args.axis}")
    print(f"  Alpha : {args.alpha}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    layer_w_dev = {L: w.to(device) for L, w in layer_w.items()}
    layer_midpoint_dev = {L: m.to(device) for L, m in layer_midpoint.items()}

    # Load probe masks if they exist
    probe_masks = None
    if args.mask_bank:
        mask_bank_path = Path(args.mask_bank)
        if mask_bank_path.exists():
            try:
                m_data = np.load(mask_bank_path)
                probe_masks = {}
                for L in LAYERS:
                    mask_key = f"{L}|{args.axis}|mask"
                    if mask_key in m_data:
                        raw_mask = m_data[mask_key]
                        if raw_mask.dtype == bool:
                            probe_masks[L] = torch.tensor(raw_mask, dtype=torch.bool).to(device)
                        else:
                            probe_masks[L] = torch.tensor(raw_mask, dtype=torch.float32).to(device)
                print(f"Loaded {len(probe_masks)} probe masks from {args.mask_bank}.")
            except Exception as e:
                print(f"Warning: failed to load probe masks: {e}")
                probe_masks = None
        else:
            print(f"Warning: probe mask bank not found at {args.mask_bank}.")

    results = []

    # Load calibration distributions for rank/zscore modes
    h_pos_dict = None
    sims_ref_dict = None
    if args.score_mode in ["rank", "proj_rank", "local_proj_rank"]:
        h_pos_dict = {"H_pos": {}, "h_pos": {}}
        for L in LAYERS:
            H_pos_key = f"{L}|{args.axis}|H_pos_1000"
            h_pos_key = f"{L}|{args.axis}|h_pos_1000"
            if H_pos_key not in v_data:
                # Fallback to older 30 sample arrays if 1000 samples are not generated
                H_pos_key = f"{L}|{args.axis}|H_pos_30"
                h_pos_key = f"{L}|{args.axis}|h_pos_30"
                
            if H_pos_key in v_data:
                h_pos_dict["H_pos"][L] = v_data[H_pos_key].astype(np.float32)
                h_pos_dict["h_pos"][L] = v_data[h_pos_key].astype(np.float32)

    # Generate baseline once for prompts
    baselines = []
    print("Generating baseline texts...")
    for idx, (orig_idx, p_text) in enumerate(prompts):
        inputs = format_and_tokenize(tokenizer, p_text, device)
        with torch.no_grad():
            base_outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
            )
        prompt_len = inputs.input_ids.shape[1]
        base_text = tokenizer.decode(base_outputs[0][prompt_len:], skip_special_tokens=True)
        base_ppl = calc_ppl(model, base_outputs[0])
        baselines.append((base_text, base_ppl))

    for idx, (orig_idx, p_text) in enumerate(tqdm(prompts)):
        inputs = format_and_tokenize(tokenizer, p_text, device)
        base_text, base_ppl = baselines[idx]

        # Layer Selection
        best_layer, raw_scores, final_scores = select_layer_proj_prior(
            model, inputs.input_ids, layer_w_dev, args.direction, layer_priors, args.score_mode,
            h_pos_dict=h_pos_dict, sims_ref_dict=sims_ref_dict, alpha=args.alpha,
            layer_midpoint_dev=layer_midpoint_dev, norm_mode=args.norm_mode,
            probe_masks=probe_masks
        )

        # Generate
        dyn_text, dyn_ids = generate_with_steered_layer(
            model, tokenizer, p_text, layer_w_dev[best_layer], args.alpha, best_layer,
            norm_mode=args.norm_mode
        )
        dyn_ppl = calc_ppl(model, dyn_ids)

        results.append({
            "idx": idx, "orig_idx": orig_idx, "prompt": p_text,
            "base_text": base_text, "base_ppl": base_ppl,
            "dyn_text": dyn_text, "dyn_ppl": dyn_ppl,
            "dyn_layer": best_layer,
            "raw_scores":   {str(L): float(v) for L, v in raw_scores.items()},
            "final_scores": {str(L): float(v) for L, v in final_scores.items()},
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
