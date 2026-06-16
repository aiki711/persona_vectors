#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/diagnose_dls_alignment.py
#
# Diagnoses the alignment (correlation) between representation-space steering scores 
# (cosine similarity / projection-cosine / rank / proj-rank / masked equivalents) 
# and downstream logit influence (logit-diff).
# Runs on a few prompts to prevent excessive compute overhead.
#

import argparse
import json
import torch
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

LAYERS = list(range(32))

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       "-c", default="config/mistral_7b.yaml")
    ap.add_argument("--vector_bank",  default="vectors/mean_diff_vectors.npz")
    ap.add_argument("--mask_bank",    default="vectors/probe_masks.npz")
    ap.add_argument("--prompts",      default="inputs/eval_prompts_10.jsonl")
    ap.add_argument("--axis",         type=str, default="extraversion")
    ap.add_argument("--alpha",        type=float, default=5.0)
    ap.add_argument("--num_prompts",  type=int, default=5)
    args = ap.parse_args()

    # Login node execution guard to comply with rule
    import socket
    import sys
    hostname = socket.gethostname()
    if "hakusan" in hostname:
        print(f"\n[ERROR] This script cannot be run directly on the login node '{hostname}'.")
        print("Please submit this script as a SLURM job using sbatch.")
        sys.exit(1)

    print(f"=== Running DLS Alignment Diagnosis ({args.axis}, alpha={args.alpha}) ===")
    
    # Load vectors
    v_data = np.load(args.vector_bank)
    layer_w = {}
    layer_midpoint = {}
    h_pos_dict = {}
    for L in LAYERS:
        w_key = f"{L}|{args.axis}|w"
        raw_norm_key = f"{L}|{args.axis}|raw_norm"
        mp_key = f"{L}|{args.axis}|midpoint"
        if w_key in v_data:
            w_vec = torch.tensor(v_data[w_key], dtype=torch.float32)
            # Apply raw_norm scaling
            if raw_norm_key in v_data:
                r_norm = float(v_data[raw_norm_key][0])
                w_norm = torch.norm(w_vec).item()
                w_vec = (w_vec / (w_norm + 1e-10)) * r_norm
            layer_w[L] = w_vec
        if mp_key in v_data:
            layer_midpoint[L] = torch.tensor(v_data[mp_key], dtype=torch.float32)
            
        # Load calibration representations for rank calculations
        H_pos_key = f"{L}|{args.axis}|H_pos_1000"
        if H_pos_key not in v_data:
            H_pos_key = f"{L}|{args.axis}|H_pos_30"
        if H_pos_key in v_data:
            h_pos_dict[L] = v_data[H_pos_key].astype(np.float32)

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
                prompts.append(item["input"])
            elif isinstance(item, str):
                prompts.append(item)
    prompts = prompts[:args.num_prompts]

    # Load model
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    layer_w_dev = {L: w.to(device) for L, w in layer_w.items()}
    layer_midpoint_dev = {L: m.to(device) for L, m in layer_midpoint.items()}
    stack, _, _ = get_layer_stack(model)

    # Load probe masks if they exist
    mask_bank_path = Path(args.mask_bank)
    probe_masks = {}
    if mask_bank_path.exists():
        try:
            m_data = np.load(mask_bank_path)
            for L in LAYERS:
                mask_key = f"{L}|{args.axis}|mask"
                if mask_key in m_data:
                    # convert to torch boolean tensor
                    probe_masks[L] = torch.tensor(m_data[mask_key], dtype=torch.bool).to(device)
            print(f"Loaded {len(probe_masks)} probe masks from {args.mask_bank}.")
        except Exception as e:
            print(f"Warning: failed to load probe masks: {e}")
    else:
        print(f"Warning: probe mask bank not found at {args.mask_bank}. Masked metrics will fallback to standard metrics.")

    records = []

    for p_idx, prompt in enumerate(prompts):
        inputs = format_and_tokenize(tokenizer, prompt, device)
        input_ids = inputs.input_ids

        # 1. Base logit
        with torch.no_grad():
            out = model(input_ids)
        base_logits = out.logits[0, -1, :].float()

        # 2. Extract base hidden states
        saved_h = {}
        handles = []
        for L in layer_w_dev.keys():
            def get_hook(L_idx):
                def hook(mod, inp, out_val):
                    hs = out_val[0] if isinstance(out_val, tuple) else out_val
                    saved_h[L_idx] = hs[0, -1, :].detach().float()
                return hook
            handles.append(stack[L].register_forward_hook(get_hook(L)))

        try:
            with torch.no_grad():
                _ = model(input_ids)
        finally:
            for h in handles:
                h.remove()

        # 3. Compute logit-diff for each layer individually
        logit_diff_scores = {}
        for L, w_dev in layer_w_dev.items():
            def steer_hook(mod, inp, out_val):
                hs = out_val[0] if isinstance(out_val, tuple) else out_val
                if not torch.isfinite(hs).all(): return out_val
                hs_f32 = hs.to(torch.float32)
                steered = hs_f32 + args.alpha * w_dev.view(1, 1, -1)
                return (steered.to(hs.dtype), *out_val[1:]) if isinstance(out_val, tuple) else steered.to(hs.dtype)

            handle = stack[L].register_forward_hook(steer_hook)
            try:
                with torch.no_grad():
                    out_steered = model(input_ids)
                steered_logits = out_steered.logits[0, -1, :].float()
                # Measure norm of difference in output distribution
                logit_diff_scores[L] = (steered_logits - base_logits).norm().item()
            finally:
                handle.remove()

        # 4. Compute representation-space scores
        cos_scores = {}
        proj_cos_scores = {}
        masked_cos_scores = {}
        masked_proj_cos_scores = {}
        rank_scores = {}
        proj_rank_scores = {}

        for L, w_dev in layer_w_dev.items():
            h = saved_h[L]
            m = layer_midpoint_dev.get(L, None)

            # Cosine similarity
            h_unit = h / (torch.norm(h) + 1e-10)
            w_unit = w_dev / (torch.norm(w_dev) + 1e-10)
            cos_scores[L] = torch.dot(h_unit, w_unit).item()

            # Projection Cosine similarity (relative to midpoint)
            if m is not None:
                h_dev = h - m
                h_dev_unit = h_dev / (torch.norm(h_dev) + 1e-10)
                proj_cos_scores[L] = torch.dot(h_dev_unit, w_unit).item()
            else:
                proj_cos_scores[L] = cos_scores[L]
                
            # Masked Cosine & Masked Projection Cosine
            mask = probe_masks.get(L, None)
            if mask is not None:
                h_masked = h * mask
                w_dev_masked = w_dev * mask
                
                h_masked_unit = h_masked / (torch.norm(h_masked) + 1e-10)
                w_dev_masked_unit = w_dev_masked / (torch.norm(w_dev_masked) + 1e-10)
                
                masked_cos_scores[L] = torch.dot(h_masked_unit, w_dev_masked_unit).item()
                
                if m is not None:
                    h_dev = h - m
                    h_dev_masked = h_dev * mask
                    h_dev_masked_unit = h_dev_masked / (torch.norm(h_dev_masked) + 1e-10)
                    masked_proj_cos_scores[L] = torch.dot(h_dev_masked_unit, w_dev_masked_unit).item()
                else:
                    masked_proj_cos_scores[L] = masked_cos_scores[L]
            else:
                masked_cos_scores[L] = cos_scores[L]
                masked_proj_cos_scores[L] = proj_cos_scores[L]
                
            # Rank score
            if L in h_pos_dict and m is not None:
                H_pos = h_pos_dict[L]
                h_np = h.cpu().numpy()
                h_unit_np = h_np / (np.linalg.norm(h_np) + 1e-10)
                
                H_pos_norm = H_pos / (np.linalg.norm(H_pos, axis=1, keepdims=True) + 1e-10)
                sims_ref = np.dot(H_pos_norm, h_unit_np.T)
                
                m_np = m.cpu().numpy()
                m_norm = m_np / (np.linalg.norm(m_np) + 1e-10)
                sim_mid = np.dot(m_norm, h_unit_np.T).item()
                
                sims_total = np.concatenate([sims_ref, [sim_mid]])
                ranking = np.argsort(sims_total)
                rank_idx = np.where(ranking == len(sims_ref))[0][0]
                rank_scores[L] = rank_idx / float(len(sims_ref))
            else:
                rank_scores[L] = cos_scores[L]

            # Proj-Rank score
            if L in h_pos_dict and m is not None:
                H_pos = h_pos_dict[L]
                m_np = m.cpu().numpy()
                w_np = w_dev.cpu().numpy()
                h_np = h.cpu().numpy()
                w_unit_np = w_np / (np.linalg.norm(w_np) + 1e-10)
                
                p_pos = np.dot(H_pos - m_np, w_unit_np)
                p_h = np.dot(h_np - m_np, w_unit_np).item()
                
                p_total = np.concatenate([p_pos, [p_h]])
                ranking = np.argsort(p_total)
                rank_idx = np.where(ranking == len(p_pos))[0][0]
                percentile = rank_idx / float(len(p_pos))
                proj_rank_scores[L] = 1.0 - percentile
            else:
                proj_rank_scores[L] = cos_scores[L]

        # Calculate correlations for this prompt
        layers = sorted(layer_w_dev.keys())
        ld_vals = [logit_diff_scores[L] for L in layers]
        cos_vals = [cos_scores[L] for L in layers]
        p_cos_vals = [proj_cos_scores[L] for L in layers]
        m_cos_vals = [masked_cos_scores[L] for L in layers]
        m_p_cos_vals = [masked_proj_cos_scores[L] for L in layers]
        rank_vals = [rank_scores[L] for L in layers]
        p_rank_vals = [proj_rank_scores[L] for L in layers]

        p_r_cos, _ = pearsonr(cos_vals, ld_vals)
        s_r_cos, _ = spearmanr(cos_vals, ld_vals)
        
        p_r_pcos, _ = pearsonr(p_cos_vals, ld_vals)
        s_r_pcos, _ = spearmanr(p_cos_vals, ld_vals)
        
        p_r_mcos, _ = pearsonr(m_cos_vals, ld_vals)
        s_r_mcos, _ = spearmanr(m_cos_vals, ld_vals)
        
        p_r_mpcos, _ = pearsonr(m_p_cos_vals, ld_vals)
        s_r_mpcos, _ = spearmanr(m_p_cos_vals, ld_vals)
        
        p_r_rank, _ = pearsonr(rank_vals, ld_vals)
        s_r_rank, _ = spearmanr(rank_vals, ld_vals)
        
        p_r_prank, _ = pearsonr(p_rank_vals, ld_vals)
        s_r_prank, _ = spearmanr(p_rank_vals, ld_vals)

        print(f"Prompt {p_idx+1}:")
        print(f"  Cos similarity  vs Logit-Diff: Pearson={p_r_cos:.4f}, Spearman={s_r_cos:.4f}")
        print(f"  Proj-Cos        vs Logit-Diff: Pearson={p_r_pcos:.4f}, Spearman={s_r_pcos:.4f}")
        print(f"  Mask-Cos        vs Logit-Diff: Pearson={p_r_mcos:.4f}, Spearman={s_r_mcos:.4f}")
        print(f"  Mask-Proj-Cos   vs Logit-Diff: Pearson={p_r_mpcos:.4f}, Spearman={s_r_mpcos:.4f}")
        print(f"  Rank            vs Logit-Diff: Pearson={p_r_rank:.4f}, Spearman={s_r_rank:.4f}")
        print(f"  Proj-Rank       vs Logit-Diff: Pearson={p_r_prank:.4f}, Spearman={s_r_prank:.4f}")

        for L in layers:
            records.append({
                "prompt_idx": p_idx,
                "layer": L,
                "logit_diff": logit_diff_scores[L],
                "cos": cos_scores[L],
                "proj_cos": proj_cos_scores[L],
                "mask_cos": masked_cos_scores[L],
                "mask_proj_cos": masked_proj_cos_scores[L],
                "rank": rank_scores[L],
                "proj_rank": proj_rank_scores[L]
            })

    # Overall Summary
    df = pd.DataFrame(records)
    print("\n=== Diagnosis Summary (Averaged across prompts) ===")
    
    # Compute mean metrics per layer
    mean_df = df.groupby("layer")[["logit_diff", "cos", "proj_cos", "mask_cos", "mask_proj_cos", "rank", "proj_rank"]].mean().reset_index()
    
    overall_p_cos, _ = pearsonr(mean_df["cos"], mean_df["logit_diff"])
    overall_s_cos, _ = spearmanr(mean_df["cos"], mean_df["logit_diff"])
    
    overall_p_pcos, _ = pearsonr(mean_df["proj_cos"], mean_df["logit_diff"])
    overall_s_pcos, _ = spearmanr(mean_df["proj_cos"], mean_df["logit_diff"])

    overall_p_mcos, _ = pearsonr(mean_df["mask_cos"], mean_df["logit_diff"])
    overall_s_mcos, _ = spearmanr(mean_df["mask_cos"], mean_df["logit_diff"])

    overall_p_mpcos, _ = pearsonr(mean_df["mask_proj_cos"], mean_df["logit_diff"])
    overall_s_mpcos, _ = spearmanr(mean_df["mask_proj_cos"], mean_df["logit_diff"])

    overall_p_rank, _ = pearsonr(mean_df["rank"], mean_df["logit_diff"])
    overall_s_rank, _ = spearmanr(mean_df["rank"], mean_df["logit_diff"])

    overall_p_prank, _ = pearsonr(mean_df["proj_rank"], mean_df["logit_diff"])
    overall_s_prank, _ = spearmanr(mean_df["proj_rank"], mean_df["logit_diff"])

    print(f"Overall Layer-wise Correlation:")
    print(f"  Cos-similarity vs Logit-Diff:")
    print(f"    Pearson r  = {overall_p_cos:.4f}")
    print(f"    Spearman r = {overall_s_cos:.4f}")
    print(f"  Proj-Cos-similarity vs Logit-Diff:")
    print(f"    Pearson r  = {overall_p_pcos:.4f}")
    print(f"    Spearman r = {overall_s_pcos:.4f}")
    print(f"  Mask-Cos-similarity vs Logit-Diff:")
    print(f"    Pearson r  = {overall_p_mcos:.4f}")
    print(f"    Spearman r = {overall_s_mcos:.4f}")
    print(f"  Mask-Proj-Cos-similarity vs Logit-Diff:")
    print(f"    Pearson r  = {overall_p_mpcos:.4f}")
    print(f"    Spearman r = {overall_s_mpcos:.4f}")
    print(f"  Rank vs Logit-Diff:")
    print(f"    Pearson r  = {overall_p_rank:.4f}")
    print(f"    Spearman r = {overall_s_rank:.4f}")
    print(f"  Proj-Rank vs Logit-Diff:")
    print(f"    Pearson r  = {overall_p_prank:.4f}")
    print(f"    Spearman r = {overall_s_prank:.4f}")

    # Save details to CSV
    out_path = Path("scratch/dls_diagnosis_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed diagnostic records to: {out_path}")

if __name__ == "__main__":
    main()
