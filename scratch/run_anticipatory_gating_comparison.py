#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_anticipatory_gating_comparison.py
# Comparison between 1-Token Delayed Gating vs. Anticipatory (Re-sampling Double Pass) Gating
# using raw_norm vector scaling and proper generation parameters.
#

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import sys
import copy
from pathlib import Path
import subprocess

WORKSPACE = Path("/home/s2550009/persona_vectors")
sys.path.insert(0, str(WORKSPACE))

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
)

OUT_DIR = WORKSPACE / "exp_token_intensity/exp_resampling_vs_delayed"
TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = list(range(32))

# Target Model Configurations
CONFIGS = {
    "Peak_Score": {"theta_lo": 1.2, "k_lo": 1.5, "theta_hi": 6.0, "k_hi": 1.0},
    "Best_PPL":   {"theta_lo": 1.2, "k_lo": 1.5, "theta_hi": 7.0, "k_hi": 1.0}
}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_alpha_plateau(h, alpha_max, theta_lo, k_lo, theta_hi, k_hi):
    f_lo = sigmoid(k_lo * (h - theta_lo))
    f_hi = sigmoid(-k_hi * (h - theta_hi))
    return float(alpha_max * f_lo * f_hi)

def generate_text_anticipatory(
    model, tokenizer, prompt_text, layer_w_dev, probe_masks,
    theta_lo, k_lo, theta_hi, k_hi, alpha_max=5.0, mode="anticipatory", max_new_tokens=100
):
    """
    mode: "anticipatory" (Re-sampling double pass using current step unsteered H_t)
          "delayed" (1-token delay using previous step H_{t-1})
    """
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    curr_ids = inputs["input_ids"]

    generated_tokens = []
    log_probs = []

    # State for delayed mode
    prev_h = 0.0
    layers_stack, N_layers, _ = get_layer_stack(model)

    for step in range(max_new_tokens):
        # 1. Unsteered Pass to compute next-token entropy H_t
        with torch.no_grad():
            outputs_unsteered = model(curr_ids)
            logits_unsteered = outputs_unsteered.logits[:, -1, :]
            probs_unsteered = F.softmax(logits_unsteered, dim=-1)
            log_probs_unsteered = F.log_softmax(logits_unsteered, dim=-1)
            # Compute predictive entropy H_t in nats
            h_t = float(-(probs_unsteered * log_probs_unsteered).sum(dim=-1).item())

        # Determine alpha for current step
        if mode == "anticipatory":
            # Re-sampling / Double pass: Use current step's H_t
            curr_alpha = compute_alpha_plateau(h_t, alpha_max, theta_lo, k_lo, theta_hi, k_hi)
        else:
            # Delayed mode: Use previous step's H_{t-1}
            curr_alpha = compute_alpha_plateau(prev_h, alpha_max, theta_lo, k_lo, theta_hi, k_hi)
            prev_h = h_t

        # 2. Steered Pass with computed alpha
        hooks = []
        try:
            for l_idx in LAYERS:
                if 4 <= l_idx <= 29 and l_idx < N_layers and l_idx in layer_w_dev:
                    layer_obj = layers_stack[l_idx]
                    steer_vec = curr_alpha * layer_w_dev[l_idx]
                    if probe_masks and l_idx in probe_masks:
                        steer_vec = steer_vec * probe_masks[l_idx]

                    def make_hook(sv):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                h_state = output[0]
                                h_state[:, -1, :] += sv
                                return (h_state,) + output[1:]
                            else:
                                output[:, -1, :] += sv
                                return output
                        return hook_fn

                    h_handle = layer_obj.register_forward_hook(make_hook(steer_vec))
                    hooks.append(h_handle)

            with torch.no_grad():
                outputs_steered = model(curr_ids)
                next_token_logits = outputs_steered.logits[:, -1, :].clone()
                
                # Apply repetition penalty
                for tok in curr_ids[0]:
                    l_val = next_token_logits[0, tok].item()
                    if l_val < 0:
                        next_token_logits[0, tok] = l_val * 1.1
                    else:
                        next_token_logits[0, tok] = l_val / 1.1

                probs_steered = F.softmax(next_token_logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs_steered, num_samples=1)
                
                token_log_prob = F.log_softmax(next_token_logits, dim=-1)[0, next_token.item()].item()
                log_probs.append(token_log_prob)

        finally:
            for h in hooks:
                h.remove()

        curr_ids = torch.cat([curr_ids, next_token], dim=-1)
        generated_tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    mean_ppl = float(np.exp(-np.mean(log_probs))) if log_probs else 999.0

    return gen_text, mean_ppl

def run_experiment_for_config(cfg_name, cfg_params, model, tokenizer, v_data, m_data, prompts):
    print(f"\n=======================================================")
    print(f"Running Anticipatory vs Delayed Comparison: [{cfg_name}]")
    print(f"Params: {cfg_params}")
    print(f"=======================================================")

    device = _infer_main_device(model)

    for trait in TRAITS:
        # Load vectors for this trait with raw_norm scaling
        layer_w_dev = {}
        for L in LAYERS:
            w_key = f"{L}|{trait}|w"
            raw_norm_key = f"{L}|{trait}|raw_norm"
            if w_key in v_data:
                w_vec = torch.tensor(v_data[w_key], dtype=torch.float32, device=device)
                if raw_norm_key in v_data:
                    r_norm = float(v_data[raw_norm_key][0])
                    w_norm = torch.norm(w_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * r_norm
                layer_w_dev[L] = w_vec

        # Load masks for this trait
        probe_masks = {}
        if m_data is not None:
            for L in LAYERS:
                m_key = f"{L}|{trait}|mask"
                if m_key in m_data:
                    probe_masks[L] = torch.tensor(m_data[m_key], dtype=torch.float32, device=device)

        for mode in ["anticipatory", "delayed"]:
            trait_out_dir = OUT_DIR / cfg_name / mode / trait
            trait_out_dir.mkdir(parents=True, exist_ok=True)

            out_jsonl = trait_out_dir / f"masked_proj_rank_theta_{cfg_params['theta_lo']:.1f}_{cfg_params['theta_hi']:.1f}_k_{cfg_params['k_lo']:.1f}_{cfg_params['k_hi']:.1f}_entropy_plateau_Val5.0.jsonl"
            
            # Force overwrite to fix previous broken outputs
            print(f"Generating for {cfg_name} | Mode: {mode} | Trait: {trait}...")
            results = []
            for item in prompts:
                prompt_text = item["prompt"]
                prompt_id = item.get("id", len(results))

                gen_text, ppl = generate_text_anticipatory(
                    model, tokenizer, prompt_text, layer_w_dev, probe_masks,
                    theta_lo=cfg_params["theta_lo"],
                    k_lo=cfg_params["k_lo"],
                    theta_hi=cfg_params["theta_hi"],
                    k_hi=cfg_params["k_hi"],
                    alpha_max=5.0,
                    mode=mode,
                    max_new_tokens=100
                )

                results.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "dyn_text": gen_text,
                    "generated_text": gen_text,
                    "dyn_ppl": ppl,
                    "trait": trait,
                    "mode": mode,
                    "config": cfg_name
                })

            with open(out_jsonl, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved: {out_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Anticipatory vs Delayed Gating Comparison")
    parser.add_argument("--config", default="configs/mistral_7b.yaml")
    parser.add_argument("--vector_bank", default="vectors/mean_diff_vectors.npz")
    parser.add_argument("--mask_bank", default="vectors/soft_probe_masks.npz")
    parser.add_argument("--prompts", default="inputs/eval_prompts_10.jsonl")
    args = parser.parse_args()

    print("Loading Mistral-7B-v0.3 Model and Tokenizer...")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3")
    quant = cfg.get("quant", "auto")
    model, tokenizer = load_model_and_tokenizer(model_name, quant=quant)

    print("Loading vector bank and mask bank...")
    v_data = np.load(args.vector_bank)
    m_data = np.load(args.mask_bank) if Path(args.mask_bank).exists() else None

    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"): continue
            if line.endswith(","): line = line[:-1]
            try: item = json.loads(line)
            except: item = line.strip('"')
            if isinstance(item, dict):
                p_text = item.get("input") or item.get("prompt")
                p_id = item.get("orig_idx") or item.get("id") or len(prompts)
                prompts.append({"id": p_id, "prompt": p_text})
            elif isinstance(item, str):
                prompts.append({"id": len(prompts), "prompt": item})

    print(f"Loaded {len(prompts)} evaluation prompts.")

    for cfg_name, cfg_params in CONFIGS.items():
        run_experiment_for_config(cfg_name, cfg_params, model, tokenizer, v_data, m_data, prompts)

    print("\n-------------------------------------------------------")
    print("Anticipatory vs Delayed Experiment Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
