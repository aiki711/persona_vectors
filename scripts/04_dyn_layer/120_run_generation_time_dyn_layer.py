#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/120_run_generation_time_dyn_layer.py
#
# Stateful token-by-token dynamic layer steering during auto-regressive generation.
# Calculates scores (Cosine, Rank, Proj) on GPU at every token step.
# Optimized to run sweeps internally to avoid redundant model loading.
# Casts variables to float32 during calculations to avoid BFloat16/Float mismatch.
#

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import time
import socket
import sys
from pathlib import Path
from tqdm import tqdm

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

LAYERS = list(range(32))
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

class DynamicSteeringController:
    def __init__(self, layer_w, layer_midpoint, score_mode, layer_priors, h_pos_dict, probe_masks, alpha, norm_mode, update_interval=1):
        self.layer_w = layer_w # dict L -> GPU tensor
        self.layer_midpoint = layer_midpoint # dict L -> GPU tensor
        self.score_mode = score_mode
        self.layer_priors = layer_priors # dict L -> float weight
        self.h_pos_dict = h_pos_dict # dict L -> GPU tensor [1000, dim]
        self.probe_masks = probe_masks # dict L -> GPU bool tensor
        self.alpha = alpha
        self.norm_mode = norm_mode
        self.update_interval = update_interval
        
        # State
        self.active_steering_layer = min(layer_w.keys())
        self.recorded_states = {} # L -> GPU tensor [dim]
        self.step_counter = 0
        self.layer_history = []
        
        self.candidate_layers = [L for L, w_prior in layer_priors.items() if w_prior > 1e-5]
        if self.candidate_layers:
            self.active_steering_layer = min(self.candidate_layers)

    def register_hooks(self, model):
        stack, _, _ = get_layer_stack(model)
        self.handles = []
        for L in self.layer_w.keys():
            def make_hook(L_idx):
                def hook(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    
                    if hs.size(1) == 1:
                        # Record hidden state of the current token
                        self.recorded_states[L_idx] = hs[0, 0, :].detach().clone()
                        
                        # Apply steering if active
                        if L_idx == self.active_steering_layer:
                            hs_f32 = hs.to(torch.float32)
                            w_dev = self.layer_w[L_idx]
                            if self.norm_mode == "relative":
                                h_norm = torch.norm(hs_f32, p=2, dim=-1, keepdim=True)
                                steered = hs_f32 + self.alpha * w_dev.view(1, 1, -1) * h_norm
                            else:
                                steered = hs_f32 + self.alpha * w_dev.view(1, 1, -1)
                            
                            return (steered.to(hs.dtype), *out[1:]) if isinstance(out, tuple) else steered.to(hs.dtype)
                    else:
                        # Prompt processing step
                        self.recorded_states[L_idx] = hs[0, -1, :].detach().clone()
                    
                    # Trigger active layer update after the max candidate layer has completed its forward pass
                    if L_idx == max(self.candidate_layers):
                        self.update_layer()
                        
                    return out
                return hook
            self.handles.append(stack[L].register_forward_hook(make_hook(L)))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def update_layer(self):
        if len(self.recorded_states) < len(self.candidate_layers):
            return
            
        self.step_counter += 1
        if self.step_counter % self.update_interval != 0:
            self.layer_history.append(self.active_steering_layer)
            return

        raw_scores = {}
        for L in self.candidate_layers:
            # Cast everything to float32 to avoid dtype mismatches (BFloat16 vs Float)
            h = self.recorded_states[L].to(torch.float32)
            w_dev = self.layer_w[L].to(torch.float32)
            m = self.layer_midpoint.get(L, None)
            if m is not None:
                m = m.to(torch.float32)
            mask = self.probe_masks.get(L, None) if self.probe_masks else None

            if self.score_mode == "local_proj_rank":
                if self.h_pos_dict and L in self.h_pos_dict and m is not None:
                    H_pos = self.h_pos_dict[L].to(torch.float32) # [1000, dim]
                    if mask is not None:
                        H_pos = H_pos[:, mask]
                        m_val = m[mask]
                        h_val = h[mask]
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
                        h_m = h * mask
                        w_m = w_dev * mask
                    else:
                        h_m = h
                        w_m = w_dev
                    h_unit = h_m / (torch.norm(h_m) + 1e-10)
                    w_unit = w_m / (torch.norm(w_m) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
            elif self.score_mode == "proj_rank":
                if self.h_pos_dict and L in self.h_pos_dict and m is not None:
                    H_pos = self.h_pos_dict[L].to(torch.float32) # [1000, dim]
                    if mask is not None:
                        H_pos = H_pos[:, mask]
                        m_val = m[mask]
                        h_val = h[mask]
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
                        h_m = h * mask
                        w_m = w_dev * mask
                    else:
                        h_m = h
                        w_m = w_dev
                    h_unit = h_m / (torch.norm(h_m) + 1e-10)
                    w_unit = w_m / (torch.norm(w_m) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
                    
            elif self.score_mode == "proj_cosine":
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
                        h_m = h * mask
                        w_m = w_dev * mask
                    else:
                        h_m = h
                        w_m = w_dev
                    h_unit = h_m / (torch.norm(h_m) + 1e-10)
                    w_unit = w_m / (torch.norm(w_m) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
                    
            elif self.score_mode == "rank":
                if self.h_pos_dict and L in self.h_pos_dict and m is not None:
                    H_pos = self.h_pos_dict[L].to(torch.float32) # [1000, dim]
                    c = H_pos.mean(dim=0) # [dim]
                    if mask is not None:
                        H_pos = H_pos[:, mask]
                        c = c[mask]
                        h_val = h[mask]
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
                        h_m = h * mask
                        w_m = w_dev * mask
                    else:
                        h_m = h
                        w_m = w_dev
                    h_unit = h_m / (torch.norm(h_m) + 1e-10)
                    w_unit = w_m / (torch.norm(w_m) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()
                    
            else: # cosine
                if m is not None:
                    if mask is not None:
                        h_m = h * mask
                        m_m = m * mask
                    else:
                        h_m = h
                        m_m = m
                    h_unit = h_m / (torch.norm(h_m) + 1e-10)
                    m_unit = m_m / (torch.norm(m_m) + 1e-10)
                    score = torch.dot(h_unit, m_unit).item()
                else:
                    if mask is not None:
                        h_m = h * mask
                        w_m = w_dev * mask
                    else:
                        h_m = h
                        w_m = w_dev
                    h_unit = h_m / (torch.norm(h_m) + 1e-10)
                    w_unit = w_m / (torch.norm(w_m) + 1e-10)
                    score = torch.dot(h_unit, w_unit).item()

            raw_scores[L] = score

        final_scores = {}
        for L in self.candidate_layers:
            w_prior = self.layer_priors.get(L, 0.0)
            final_scores[L] = raw_scores[L] - (1.0 - w_prior) * 10.0

        self.active_steering_layer = max(final_scores, key=lambda L: final_scores[L])
        self.layer_history.append(self.active_steering_layer)

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)

@torch.no_grad()
def calc_ppl(model, ids):
    out = model(ids.unsqueeze(0), labels=ids.clone().unsqueeze(0))
    return torch.exp(out.loss).item()

def main():
    # Login node guard
    hostname = socket.gethostname()
    if "hakusan" in hostname:
        print(f"\n[ERROR] This GPU steering script cannot be run directly on the login node '{hostname}'.")
        print("Please submit this script as a SLURM job to execute on a GPU compute node.")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", required=True)
    ap.add_argument("--vector_bank",  required=True)
    ap.add_argument("--prompts",      required=True)
    ap.add_argument("--out_dir",      default="exp_steering_dyn_gen_time_raw/results")
    ap.add_argument("--axis",         type=str, default="extraversion")
    ap.add_argument("--alpha",        type=float, default=1.0)
    ap.add_argument("--direction",    type=str, choices=["high", "low"], default="high")
    ap.add_argument("--norm_mode",    type=str, choices=["none", "midpoint", "raw_norm", "relative"], default="raw_norm")
    ap.add_argument("--score_mode",   type=str, choices=["cosine", "rank", "proj_rank", "proj_cosine", "local_proj_rank"], default="cosine")
    ap.add_argument("--mask_bank",    default="")
    ap.add_argument("--num_prompts",  type=int, default=10)
    ap.add_argument("--update_interval", type=int, default=1)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--sweep",        action="store_true", help="Run full sweep of 8 methods and 14 alphas internally")
    ap.add_argument("--alphas",       type=str, default=None, help="Comma-separated list of alphas to run (overrides VALS)")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    direction_mult = 1.0 if args.direction == "high" else -1.0
    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model and Tokenizer (ONCE)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    # Load prompts
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

    # 2. Generate Baseline Texts (ONCE)
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

    # Candidates hard mask
    layer_priors = {L: 1.0 if 4 <= L <= 29 else 0.0 for L in LAYERS}

    # Load vectors npz
    v_data = np.load(args.vector_bank)

    # Define tasks to run
    if args.sweep:
        run_vals = VALS
        if args.alphas:
            run_vals = [float(v) for v in args.alphas.split(",")]
        tasks = []
        for val in run_vals:
            # Unmasked methods
            tasks.append(("cos_only", "cosine", "", val))
            tasks.append(("rank_only", "rank", "", val))
            tasks.append(("proj_cos_only", "proj_cosine", "", val))
            tasks.append(("proj_rank_only", "proj_rank", "", val))
            # PDF masked methods
            if args.mask_bank:
                tasks.append(("masked_cos_only", "cosine", args.mask_bank, val))
                tasks.append(("masked_rank_only", "rank", args.mask_bank, val))
                tasks.append(("masked_proj_cos_only", "proj_cosine", args.mask_bank, val))
                tasks.append(("masked_proj_rank_only", "proj_rank", args.mask_bank, val))
            # New Local Proj-Rank
            tasks.append(("local_proj_rank_only", "local_proj_rank", "", val))
            if args.mask_bank:
                tasks.append(("masked_local_proj_rank_only", "local_proj_rank", args.mask_bank, val))
    else:
        method_name = args.score_mode + "_only" if "cosine" not in args.score_mode else args.score_mode
        if method_name == "cosine":
            method_name = "cos_only"
        elif method_name == "rank":
            method_name = "rank_only"
        if args.mask_bank:
            method_name = "masked_" + method_name
        tasks = [(method_name, args.score_mode, args.mask_bank, args.alpha)]

    print(f"Total steering configurations to evaluate: {len(tasks)}")

    # Pre-load all steering vectors and midpoints for current axis onto GPU to avoid loading per-task
    layer_w_dev_all = {}
    layer_midpoint_dev_all = {}
    for L in LAYERS:
        w_key = f"{L}|{args.axis}|w"
        raw_norm_key = f"{L}|{args.axis}|raw_norm"
        mp_key = f"{L}|{args.axis}|midpoint"
        if w_key in v_data:
            w_vec = torch.tensor(v_data[w_key], dtype=torch.float32, device=device) * direction_mult
            if args.norm_mode in ["midpoint", "raw_norm"]:
                if raw_norm_key in v_data:
                    r_norm = float(v_data[raw_norm_key][0])
                    w_norm = torch.norm(w_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * r_norm
                elif mp_key in v_data:
                    m_vec = torch.tensor(v_data[mp_key], dtype=torch.float32, device=device)
                    w_norm = torch.norm(w_vec).item()
                    m_norm = torch.norm(m_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * m_norm
            elif args.norm_mode == "relative":
                w_norm = torch.norm(w_vec).item()
                w_vec = w_vec / (w_norm + 1e-10)
            layer_w_dev_all[L] = w_vec
        if mp_key in v_data:
            layer_midpoint_dev_all[L] = torch.tensor(v_data[mp_key], dtype=torch.float32, device=device)

    # Pre-load masks
    probe_masks_all = None
    if args.mask_bank:
        mask_bank_path = Path(args.mask_bank)
        if mask_bank_path.exists():
            m_data = np.load(mask_bank_path)
            probe_masks_all = {}
            for L in LAYERS:
                mask_key = f"{L}|{args.axis}|mask"
                if mask_key in m_data:
                    probe_masks_all[L] = torch.tensor(m_data[mask_key], dtype=torch.bool, device=device)

    # Pre-load calibration matrices
    h_pos_dict_all = {}
    for L in LAYERS:
        H_pos_key = f"{L}|{args.axis}|H_pos_1000"
        if H_pos_key not in v_data:
            H_pos_key = f"{L}|{args.axis}|H_pos_30"
        if H_pos_key in v_data:
            h_pos_dict_all[L] = torch.tensor(v_data[H_pos_key], dtype=torch.float32, device=device)

    # Run tasks
    for m_name, s_mode, m_bank, alpha in tasks:
        out_file = out_dir / f"{m_name}_Val{alpha}.jsonl"
        if out_file.exists():
            print(f"[SKIP] Already exists: {out_file}")
            continue

        print(f"\n--- Running: {m_name} (alpha={alpha}) ---")

        # Setup controller components
        probe_masks = probe_masks_all if m_bank else None
        h_pos_dict = h_pos_dict_all if s_mode in ["rank", "proj_rank", "local_proj_rank"] else None

        results = []
        for idx, (orig_idx, p_text) in enumerate(prompts):
            inputs = format_and_tokenize(tokenizer, p_text, device)
            base_text, base_ppl = baselines[idx]

            controller = DynamicSteeringController(
                layer_w_dev_all, layer_midpoint_dev_all, s_mode, layer_priors,
                h_pos_dict, probe_masks, alpha, args.norm_mode,
                update_interval=args.update_interval
            )
            controller.register_hooks(model)

            t_start = time.time()
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=150, do_sample=True,
                        temperature=0.7, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
                    )
            finally:
                controller.remove_hooks()
                
            t_generation = time.time() - t_start
            prompt_len = inputs.input_ids.shape[1]
            dyn_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            dyn_ppl = calc_ppl(model, outputs[0])
            num_generated_tokens = outputs[0].shape[0] - prompt_len
            tokens_per_second = num_generated_tokens / (t_generation + 1e-10)

            results.append({
                "idx": idx, "orig_idx": orig_idx, "prompt": p_text,
                "base_text": base_text, "base_ppl": base_ppl,
                "dyn_text": dyn_text, "dyn_ppl": dyn_ppl,
                "tokens_per_sec": tokens_per_second,
                "layer_history": controller.layer_history,
            })

        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved: {out_file}")

    print(f"\nAll tasks for {args.axis} finished successfully.")

if __name__ == "__main__":
    main()
