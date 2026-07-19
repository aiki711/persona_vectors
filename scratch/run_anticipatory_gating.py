#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_anticipatory_gating.py
#
# Anticipatory Dynamic Gating (先読み的予測制御) using Double Forward Pass.
# Computes next token's surprisal (IC) on unsteered pass, determines alpha, and executes steered pass.
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
import copy
from pathlib import Path

# Import utility functions from persona_vectors.live_axes
from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

LAYERS = list(range(32))

def copy_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    try:
        if isinstance(past_key_values, tuple):
            return tuple(
                tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in layer)
                for layer in past_key_values
            )
        else:
            return copy.deepcopy(past_key_values)
    except Exception as e:
        print(f"Warning: Failed to clone past_key_values standard tuple cache: {e}. Using copy.deepcopy.")
        return copy.deepcopy(past_key_values)

class AnticipatoryIntensitySteeringController:
    def __init__(self, layer_w, layer_midpoint, score_mode, layer_priors, h_pos_dict, probe_masks, alpha_max, norm_mode, update_interval=1, static_layer=False):
        self.layer_w = layer_w # dict L -> GPU tensor
        self.layer_midpoint = layer_midpoint # dict L -> GPU tensor
        self.score_mode = score_mode
        self.layer_priors = layer_priors # dict L -> float weight
        self.h_pos_dict = h_pos_dict # dict L -> GPU tensor
        self.probe_masks = probe_masks # dict L -> GPU tensor (soft mask)
        self.alpha_max = alpha_max
        self.norm_mode = norm_mode
        self.update_interval = update_interval
        self.static_layer = static_layer
        
        # State
        self.alpha = alpha_max # dynamic alpha
        self.is_preview = False # If True, skip modifying hidden states and recording states
        self.active_steering_layer = min(layer_w.keys())
        self.recorded_states = {} # L -> GPU tensor
        self.step_counter = 0
        self.layer_history = []
        self.alpha_history = []
        self.surprisal_history = []
        
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
                    steered_out = out
                    
                    if self.is_preview:
                        # 1. Preview Pass (no steering, skip recording and state updates)
                        return steered_out
                    
                    if hs.size(1) == 1:
                        # Record hidden state of the current token in the real pass
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
                            
                            steered_hs = steered.to(hs.dtype)
                            steered_out = (steered_hs, *out[1:]) if isinstance(out, tuple) else steered_hs
                    else:
                        # Prompt processing step (multiple tokens)
                        self.recorded_states[L_idx] = hs[0, -1, :].detach().clone()
                    
                    # Trigger active layer update after the max candidate layer has completed its forward pass
                    if L_idx == max(self.candidate_layers):
                        self.update_layer()
                        
                    return steered_out
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
        if self.static_layer and self.step_counter > 1:
            self.layer_history.append(self.active_steering_layer)
            return
            
        if not self.static_layer and self.step_counter % self.update_interval != 0:
            self.layer_history.append(self.active_steering_layer)
            return

        raw_scores = {}
        for L in self.candidate_layers:
            h = self.recorded_states[L].to(torch.float32)
            w_dev = self.layer_w[L].to(torch.float32)
            m = self.layer_midpoint.get(L, None)
            if m is not None:
                m = m.to(torch.float32)
            mask = self.probe_masks.get(L, None) if self.probe_masks else None

            if self.score_mode == "proj_rank":
                if self.h_pos_dict and L in self.h_pos_dict and m is not None:
                    H_pos = self.h_pos_dict[L].to(torch.float32)
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
                        
                    d_pos = H_pos - m_val.unsqueeze(0)
                    w_avg = d_pos.mean(dim=0)
                    d_h = h_val - m_val
                    
                    d_pos_norm = d_pos / (torch.norm(d_pos, p=2, dim=-1, keepdim=True) + 1e-10)
                    w_avg_norm = w_avg / (torch.norm(w_avg, p=2, dim=-1) + 1e-10)
                    d_h_norm = d_h / (torch.norm(d_h, p=2, dim=-1) + 1e-10)
                    
                    S_i = torch.matmul(d_pos_norm, d_h_norm)
                    S_center = torch.dot(w_avg_norm, d_h_norm).item()
                    
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
    hostname = socket.gethostname()
    if "hakusan" in hostname:
        print(f"\n[ERROR] This GPU steering script cannot be run directly on the login node '{hostname}'.")
        print("Please submit this script as a SLURM job to execute on a GPU compute node.")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", required=True)
    ap.add_argument("--vector_bank",  required=True)
    ap.add_argument("--prompts",      required=True)
    ap.add_argument("--out_dir",      default="exp_token_intensity/exp_anticipatory_gating")
    ap.add_argument("--axis",         type=str, default="extraversion")
    ap.add_argument("--alpha_max",    type=float, default=5.0)
    ap.add_argument("--direction",    type=str, choices=["high", "low"], default="high")
    ap.add_argument("--norm_mode",    type=str, choices=["none", "midpoint", "raw_norm", "relative"], default="raw_norm")
    ap.add_argument("--score_mode",   type=str, default="proj_rank")
    ap.add_argument("--mask_bank",    default="")
    ap.add_argument("--num_prompts",  type=int, default=10)
    
    # Gating parameters
    ap.add_argument("--theta_lo",     type=float, default=3.0)
    ap.add_argument("--theta_hi",     type=float, default=7.0)
    ap.add_argument("--k_lo",         type=float, default=2.0)
    ap.add_argument("--k_hi",         type=float, default=2.0)
    ap.add_argument("--gating_mode",  type=str, choices=["plateau"], default="plateau")
    ap.add_argument("--update_interval", type=int, default=1)
    ap.add_argument("--static_layer", action="store_true")
    ap.add_argument("--resample", action="store_true", default=True, help="Re-sample token from steered distribution")
    ap.add_argument("--no_resample", action="store_false", dest="resample", help="Disable re-sampling (decide token on base distribution)")
    ap.add_argument("--seed",         type=int, default=42)
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

    # 1. Load Model and Tokenizer
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

    # Pre-generate baseline texts (once)
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

    # Steering layer candidate range (4 to 29)
    layer_priors = {L: 1.0 if 4 <= L <= 29 else 0.0 for L in LAYERS}

    # Load vector bank npz
    v_data = np.load(args.vector_bank)

    # Pre-load all steering vectors and midpoints onto GPU
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

    # Pre-load soft masks
    probe_masks_all = None
    if args.mask_bank:
        mask_bank_path = Path(args.mask_bank)
        if mask_bank_path.exists():
            m_data = np.load(mask_bank_path)
            probe_masks_all = {}
            for L in LAYERS:
                mask_key = f"{L}|{args.axis}|mask"
                if mask_key in m_data:
                    raw_mask = m_data[mask_key]
                    if raw_mask.dtype == bool:
                        probe_masks_all[L] = torch.tensor(raw_mask, dtype=torch.bool, device=device)
                    else:
                        probe_masks_all[L] = torch.tensor(raw_mask, dtype=torch.float32, device=device)

    # Pre-load calibration reference sets
    h_pos_dict_all = {}
    for L in LAYERS:
        H_pos_key = f"{L}|{args.axis}|H_pos_1000"
        if H_pos_key not in v_data:
            H_pos_key = f"{L}|{args.axis}|H_pos_30"
        if H_pos_key in v_data:
            h_pos_dict_all[L] = torch.tensor(v_data[H_pos_key], dtype=torch.float32, device=device)

    out_prefix = args.score_mode
    if args.mask_bank:
        out_prefix = "masked_" + out_prefix
        
    suffix = "_anticipatory_resampled" if args.resample else "_anticipatory"
    out_file = out_dir / f"{out_prefix}_theta_{args.theta_lo}_{args.theta_hi}_k_{args.k_lo}_{args.k_hi}{suffix}_Val{args.alpha_max}.jsonl"
    print(f"\nTarget Output: {out_file}")

    if out_file.exists():
        print(f"[SKIP] Already exists: {out_file}")
        return

    # Running generation
    results = []
    for idx, (orig_idx, p_text) in enumerate(prompts):
        inputs = format_and_tokenize(tokenizer, p_text, device)
        base_text, base_ppl = baselines[idx]

        controller = AnticipatoryIntensitySteeringController(
            layer_w_dev_all, layer_midpoint_dev_all, args.score_mode, layer_priors,
            h_pos_dict_all, probe_masks_all, args.alpha_max, args.norm_mode,
            update_interval=args.update_interval, static_layer=args.static_layer
        )
        controller.register_hooks(model)

        t_start = time.time()
        
        past_key_values = None
        input_ids = inputs.input_ids.clone()
        generated_tokens = []
        
        # Initial alpha for prompt processing
        controller.alpha = args.alpha_max
        controller.is_preview = False
        
        try:
            with torch.no_grad():
                for step in range(150):
                    # --- 1. Preview Pass (unsteered) to evaluate next token probability ---
                    past_key_values_base = copy_past_key_values(past_key_values)
                    controller.is_preview = True
                    controller.alpha = 0.0
                    
                    if past_key_values_base is None:
                        outputs_base = model(input_ids, use_cache=True)
                    else:
                        outputs_base = model(input_ids[:, -1:], past_key_values=past_key_values_base, use_cache=True)
                    
                    next_token_logits_base = outputs_base.logits[:, -1, :].clone()
                    
                    # Apply repetition penalty (1.1) to unsteered logits
                    for tok in input_ids[0]:
                        logit_val = next_token_logits_base[0, tok].item()
                        if logit_val < 0:
                            next_token_logits_base[0, tok] = logit_val * 1.1
                        else:
                            next_token_logits_base[0, tok] = logit_val / 1.1
                            
                    # Softmax with temperature 0.7
                    probs_base = F.softmax(next_token_logits_base / 0.7, dim=-1)
                    
                    # Sample temporary token to calculate surprisal
                    temp_token = torch.multinomial(probs_base, num_samples=1)
                    temp_token_id = temp_token.item()
                    
                    # Calculate surprisal in bits based on temporary token
                    token_prob_base = probs_base[0, temp_token_id].item()
                    ic = -np.log2(token_prob_base + 1e-10)
                    
                    # Calculate anticipatory gating factor (Plateau)
                    if ic < args.theta_lo:
                        gating_factor = 2.0 / (1.0 + np.exp(-args.k_lo * (ic - args.theta_lo)))
                    elif ic <= args.theta_hi:
                        gating_factor = 1.0
                    else:
                        gating_factor = 2.0 / (1.0 + np.exp(args.k_hi * (ic - args.theta_hi)))
                    
                    # --- 2. Real Pass (steered) with target alpha ---
                    controller.alpha = args.alpha_max * gating_factor
                    controller.is_preview = False
                    
                    if past_key_values is None:
                        outputs_real = model(input_ids, use_cache=True)
                    else:
                        outputs_real = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                        
                    past_key_values = outputs_real.past_key_values
                    next_token_logits_real = outputs_real.logits[:, -1, :].clone()
                    
                    # Apply repetition penalty to real logits
                    for tok in input_ids[0]:
                        logit_val = next_token_logits_real[0, tok].item()
                        if logit_val < 0:
                            next_token_logits_real[0, tok] = logit_val * 1.1
                        else:
                            next_token_logits_real[0, tok] = logit_val / 1.1
                            
                    # Softmax with temperature 0.7
                    probs_real = F.softmax(next_token_logits_real / 0.7, dim=-1)
                    
                    # Decide token based on resampling flag
                    if args.resample:
                        next_token = torch.multinomial(probs_real, num_samples=1)
                        token_id = next_token.item()
                    else:
                        next_token = temp_token
                        token_id = temp_token_id
                    
                    if token_id == tokenizer.eos_token_id:
                        break
                        
                    generated_tokens.append(token_id)
                    
                    controller.surprisal_history.append(ic)
                    controller.alpha_history.append(controller.alpha)
                    
                    # Concat actual token to sequence
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
        finally:
            controller.remove_hooks()
            
        t_generation = time.time() - t_start
        dyn_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        dyn_ppl = calc_ppl(model, input_ids[0])
        num_generated_tokens = len(generated_tokens)
        tokens_per_second = num_generated_tokens / (t_generation + 1e-10)

        results.append({
            "idx": idx, "orig_idx": orig_idx, "prompt": p_text,
            "base_text": base_text, "base_ppl": base_ppl,
            "dyn_text": dyn_text, "dyn_ppl": dyn_ppl,
            "tokens_per_sec": tokens_per_second,
            "layer_history": controller.layer_history,
            "alpha_history": controller.alpha_history,
            "surprisal_history": controller.surprisal_history,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {out_file}")
    print(f"All tasks for {args.axis} finished successfully.")

if __name__ == "__main__":
    main()
