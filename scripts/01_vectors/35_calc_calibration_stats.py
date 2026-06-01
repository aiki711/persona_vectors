#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 35_calc_calibration_stats.py
#
# Pre-computes cosine similarity statistics (mean, std, sorted list)
# on an independent calibration set from the training data.
# This prevents data leakage from the evaluation/test set.
#

import json
import torch
import numpy as np
import yaml
from pathlib import Path
from datasets import load_dataset
from persona_vectors.live_axes import load_model_and_tokenizer, _infer_main_device, get_layer_stack

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
N_POS_SAMPLES = 30
N_CALIB_SAMPLES = 100

def extract_positive_texts(ds, axis, limit=30):
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
    return texts

def extract_calibration_prompts(ds, limit=100):
    prompts = []
    seen = set()
    # Extract unique prompts from the dataset
    for ex in ds:
        ti = (ex.get("train_input") or "").strip()
        if ti and ti not in seen:
            seen.add(ti)
            prompts.append(ti)
            if len(prompts) >= limit:
                break
    return prompts

def main():
    config_path = Path("config/mistral_7b.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_name = cfg.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3")
    print(f"Loading model: {model_name}...")
    model, tok = load_model_and_tokenizer(model_name, quant="4bit")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    layers_stack, N_layers, _ = get_layer_stack(model)
    layer_indices = list(range(N_layers))
    device = _infer_main_device(model)
    model.eval()
    
    print("Loading Big5Chat dataset...")
    ds_all = load_dataset("wenkai-li/big5_chat")
    split_name = next(iter(ds_all.keys()))
    ds = ds_all[split_name]
    
    # 1. Extract 100 independent calibration prompts (user inputs)
    calib_prompts = extract_calibration_prompts(ds, limit=N_CALIB_SAMPLES)
    print(f"Extracted {len(calib_prompts)} independent calibration prompts.")
    
    # Pre-compute hidden states for calibration prompts (reusable across traits)
    print("Extracting hidden states for calibration prompts...")
    h_calib_all = {L: [] for L in layer_indices}
    for p_text in calib_prompts:
        formatted = tok.apply_chat_template([{"role": "user", "content": p_text}], add_generation_prompt=True, tokenize=True)
        input_ids = torch.tensor([formatted]).to(device)
        saved_h = {}
        handles = []
        def get_hook(L):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                saved_h[L] = hs[0, -1, :].detach().cpu().float().numpy()
            return hook
        for L in layer_indices:
            handles.append(layers_stack[L].register_forward_hook(get_hook(L)))
        try:
            with torch.no_grad():
                _ = model(input_ids)
        finally:
            for h in handles: h.remove()
        for L in layer_indices:
            h_calib_all[L].append(saved_h[L])

    # Convert calibration hidden states to stacked numpy arrays
    H_calib_dict = {}
    for L in layer_indices:
        H_calib = np.stack(h_calib_all[L], axis=0) # [N_CALIB_SAMPLES, n_dims]
        H_calib_dict[L] = H_calib / (np.linalg.norm(H_calib, axis=1, keepdims=True) + 1e-10)
        
    vectors_dir = Path("vectors")
    vectors_dir.mkdir(exist_ok=True)
    
    # 2. Process each trait to compute positive mean and compile stats
    for trait in TRAITS:
        print(f"\n=== Processing Trait: {trait} ===")
        pos_texts = extract_positive_texts(ds, trait, limit=N_POS_SAMPLES)
        print(f"Extracted {len(pos_texts)} positive texts.")
        
        h_pos_all = {L: [] for L in layer_indices}
        
        @torch.no_grad()
        def get_hidden_states(texts):
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

        batch_size = 5
        for i in range(0, len(pos_texts), batch_size):
            batch = pos_texts[i:i+batch_size]
            out_hs = get_hidden_states(batch)
            for L in layer_indices:
                h_pos_all[L].append(out_hs[L].cpu().numpy())
                
        h_pos_dict = {}
        for L in layer_indices:
            H_pos = np.concatenate(h_pos_all[L], axis=0)
            h_pos_dict[L] = np.mean(H_pos, axis=0, keepdims=True) # [1, n_dims]
            
        stats_dict = {}
        for L in layer_indices:
            h_pos_norm = h_pos_dict[L] / (np.linalg.norm(h_pos_dict[L]) + 1e-10) # [1, n_dims]
            H_calib_norm = H_calib_dict[L] # [N_CALIB_SAMPLES, n_dims]
            
            sims = np.dot(H_calib_norm, h_pos_norm.T).squeeze() # [N_CALIB_SAMPLES]
            
            mean_val = float(np.mean(sims))
            std_val = float(np.std(sims))
            sorted_sims = sorted(sims.tolist())
            
            # Use string layer indices for json compatibility
            stats_dict[str(L)] = {
                "mean": mean_val,
                "std": std_val,
                "sorted_similarities": sorted_sims,
                "h_pos": h_pos_dict[L].tolist()[0] # Save h_pos itself for online projection inference
            }
            
        out_file = vectors_dir / f"calibration_stats_{trait}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, ensure_ascii=False, indent=2)
        print(f"Saved stats to {out_file}.")

if __name__ == "__main__":
    main()
