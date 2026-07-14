#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/02_token_intensity/run_sensitivity_analysis.py
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
import pandas as pd
from pathlib import Path

# Import from persona_vectors.live_axes
from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

LAYERS = list(range(32))

class DynamicIntensitySteeringController:
    def __init__(self, layer_w, layer_midpoint, score_mode, layer_priors, h_pos_dict, probe_masks, alpha_max, norm_mode, static_layer=True):
        self.layer_w = layer_w # dict L -> GPU tensor
        self.layer_midpoint = layer_midpoint # dict L -> GPU tensor
        self.score_mode = score_mode
        self.layer_priors = layer_priors # dict L -> float weight
        self.h_pos_dict = h_pos_dict # dict L -> GPU tensor
        self.probe_masks = probe_masks # dict L -> GPU tensor (soft mask)
        self.alpha_max = alpha_max
        self.norm_mode = norm_mode
        self.static_layer = static_layer
        
        # State
        self.alpha = alpha_max # dynamic alpha, mutable during step
        self.active_steering_layer = min(layer_w.keys())
        self.recorded_states = {} # L -> GPU tensor
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
                    steered_out = out
                    
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
                    
                    h_proj = torch.dot(d_h, w_avg) / (torch.norm(w_avg) + 1e-10)
                    pos_projs = torch.matmul(d_pos, w_avg) / (torch.norm(w_avg) + 1e-10)
                    
                    rank = (pos_projs < h_proj).sum().item()
                    score = rank / len(pos_projs)
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

def clone_past_key_values(pkv):
    if pkv is None:
        return None
    if isinstance(pkv, tuple):
        return tuple(tuple(t.clone() for t in layer) for layer in pkv)
    try:
        return copy.deepcopy(pkv)
    except:
        return pkv

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)

def main():
    hostname = socket.gethostname()
    if "hakusan" in hostname:
        print(f"\n[ERROR] Sensitivity analysis cannot be run directly on the login node '{hostname}'.")
        print("Please submit this script as a SLURM job to execute on a GPU compute node.")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", required=True)
    ap.add_argument("--vector_bank",  required=True)
    ap.add_argument("--prompts",      required=True)
    ap.add_argument("--out_dir",      default="exp_token_intensity/exp_sensitivity_analysis/results")
    ap.add_argument("--alpha_max",    type=float, default=5.0)
    ap.add_argument("--score_mode",   type=str, default="proj_rank")
    ap.add_argument("--mask_bank",    default="vectors/soft_probe_masks.npz")
    ap.add_argument("--num_prompts",  type=int, default=10)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
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

    # Candidates & priors
    layer_priors = {L: 1.0 if 4 <= L <= 29 else 0.0 for L in LAYERS}
    v_data = np.load(args.vector_bank)

    # Precompute masks
    probe_masks_all = None
    if args.mask_bank:
        m_data = np.load(args.mask_bank)

    # Load 5 traits
    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

    all_records = []

    # Iterate over all traits to collect data
    for trait in traits:
        print(f"\n==========================================")
        print(f"Processing trait: {trait}")
        print(f"==========================================")
        
        # Load trait vectors
        layer_w_dev_all = {}
        layer_midpoint_dev_all = {}
        for L in LAYERS:
            w_key = f"{L}|{trait}|w"
            raw_norm_key = f"{L}|{trait}|raw_norm"
            mp_key = f"{L}|{trait}|midpoint"
            if w_key in v_data:
                w_vec = torch.tensor(v_data[w_key], dtype=torch.float32, device=device)
                if raw_norm_key in v_data:
                    r_norm = float(v_data[raw_norm_key][0])
                    w_norm = torch.norm(w_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * r_norm
                layer_w_dev_all[L] = w_vec
            if mp_key in v_data:
                layer_midpoint_dev_all[L] = torch.tensor(v_data[mp_key], dtype=torch.float32, device=device)

        # Load final layer steering vector (layer 31) for alignment gain projection
        w_final_key = f"31|{trait}|w"
        w_final = torch.tensor(v_data[w_final_key], dtype=torch.float32, device=device)
        w_final_unit = w_final / (torch.norm(w_final) + 1e-10)

        # Load trait soft masks
        probe_masks_all = {}
        if args.mask_bank:
            for L in LAYERS:
                mask_key = f"{L}|{trait}|mask"
                if mask_key in m_data:
                    raw_mask = m_data[mask_key]
                    if raw_mask.dtype == bool:
                        probe_masks_all[L] = torch.tensor(raw_mask, dtype=torch.bool, device=device)
                    else:
                        probe_masks_all[L] = torch.tensor(raw_mask, dtype=torch.float32, device=device)

        # Load trait pos dictionary for proj_rank
        h_pos_dict_all = {}
        for L in LAYERS:
            H_pos_key = f"{L}|{trait}|H_pos_1000"
            if H_pos_key not in v_data:
                H_pos_key = f"{L}|{trait}|H_pos_30"
            if H_pos_key in v_data:
                h_pos_dict_all[L] = torch.tensor(v_data[H_pos_key], dtype=torch.float32, device=device)

        trait_prompts = prompts[:args.num_prompts]

        for p_idx, (orig_idx, p_text) in enumerate(trait_prompts):
            print(f"Prompt {p_idx+1}/{len(trait_prompts)}...")
            inputs = format_and_tokenize(tokenizer, p_text, device)
            
            # Setup controller (static_layer = True for static steering context)
            controller = DynamicIntensitySteeringController(
                layer_w_dev_all, layer_midpoint_dev_all, args.score_mode, layer_priors,
                h_pos_dict_all, probe_masks_all, args.alpha_max, norm_mode="raw_norm",
                static_layer=True
            )
            controller.register_hooks(model)
            
            past_key_values = None
            input_ids = inputs.input_ids.clone()
            
            # Start with prompt processing
            controller.alpha = args.alpha_max
            try:
                with torch.no_grad():
                    outputs = model(input_ids, use_cache=True)
                    past_key_values = outputs.past_key_values
                    
                    # Target layer chosen at start
                    L_star = controller.active_steering_layer
                    w_L = layer_w_dev_all[L_star]
                    w_unit = w_L / (torch.norm(w_L) + 1e-10)
                    
                    # Generate 50 tokens for analysis
                    for step in range(50):
                        # 1. Unsteered forward pass
                        controller.alpha = 0.0
                        unsteered_kv = clone_past_key_values(past_key_values)
                        unsteered_out = model(input_ids[:, -1:], past_key_values=unsteered_kv, use_cache=True, output_hidden_states=True)
                        unsteered_logits = unsteered_out.logits[:, -1, :].clone()
                        
                        # 2. Steered forward pass
                        controller.alpha = args.alpha_max
                        steered_out = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                        steered_logits = steered_out.logits[:, -1, :].clone()
                        
                        # Update main cache with steered outputs
                        past_key_values = steered_out.past_key_values
                        
                        # Sample next token from steered distribution
                        probs_steered = F.softmax(steered_logits / 0.7, dim=-1)
                        next_token = torch.multinomial(probs_steered, num_samples=1)
                        token_id = next_token.item()
                        
                        if token_id == tokenizer.eos_token_id:
                            break
                            
                        # Append next token
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        
                        # Calculate Metrics
                        probs_unsteered = F.softmax(unsteered_logits / 0.7, dim=-1)
                        
                        # 1. Information Content (IC) under unsteered
                        token_prob = probs_unsteered[0, token_id].item()
                        ic = -np.log2(token_prob + 1e-10)
                        
                        # 2. KL Divergence (Distortion)
                        # D_KL(P_unsteered || P_steered)
                        kl = torch.sum(probs_unsteered * (torch.log(probs_unsteered + 1e-10) - torch.log(probs_steered + 1e-10))).item()
                        
                        # 3. Alignment Gain (Projection of downstream change in final hidden state onto final steering direction)
                        h_unsteered_final = unsteered_out.hidden_states[-1][0, 0, :]
                        h_steered_final = steered_out.hidden_states[-1][0, 0, :]
                        delta_h_final = h_steered_final - h_unsteered_final
                        align_gain = torch.dot(delta_h_final.to(torch.float32), w_final_unit.to(torch.float32)).item()
                        
                        token_str = tokenizer.decode([token_id])
                        
                        all_records.append({
                            "trait": trait,
                            "prompt_idx": p_idx,
                            "step": step,
                            "token": token_str,
                            "ic": ic,
                            "kl": kl,
                            "align_gain": align_gain,
                            "layer": L_star
                        })
            except Exception as e:
                print(f"[Error in Generation]: {e}")
            finally:
                controller.remove_hooks()

    # Save to CSV
    df_results = pd.DataFrame(all_records)
    out_csv = out_dir / "token_sensitivity_records.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved all token sensitivity records ({len(all_records)} tokens) to: {out_csv}")

if __name__ == "__main__":
    main()
