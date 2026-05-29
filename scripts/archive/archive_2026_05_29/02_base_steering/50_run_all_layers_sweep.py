#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 50_run_all_layers_sweep.py
#
# Loops over all 32 layers and 14 alphas to run missing single-layer sweeps.
# Optimizes model loading by loading Mistral-7B once for generation,
# releasing memory, and then loading Llama-3-8B once for evaluation.
#

import argparse
import json
import torch
import numpy as np
import yaml
import re
import pandas as pd
import gc
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt,
    _resolve_hf_token
)

# Constants
LAYERS = list(range(32))
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

# Steerer Context Manager
class ConstantSteerer:
    def __init__(self, model, layer, w, alpha):
        self.model = model
        self.layer = layer
        self.w = w
        self.alpha = alpha
        self.handle = None

    def __enter__(self):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all(): return out
            if hs.size(1) != 1: return out # Skip prefill

            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            device = hs.device
            w_dev = self.w.to(device)

            add = w_dev.view(1, 1, -1)
            steered = hs_f32 + self.alpha * add

            if not torch.isfinite(steered).all(): return out
            steered = steered.to(orig_dtype)
            return (steered, *out[1:]) if isinstance(out, tuple) else steered

        stack, _, _ = get_layer_stack(self.model)
        self.handle = stack[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle:
            self.handle.remove()
            self.handle = None

def generate_text(model, tokenizer, prompt, max_new_tokens=150):
    device = _infer_main_device(model)
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = outputs[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), outputs[0]

@torch.no_grad()
def calc_sequence_ppl(model, input_ids):
    labels = input_ids.clone()
    outputs = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
    return torch.exp(outputs.loss).item()

# Judge LLM
TRAIT_DEFINITIONS = {
    "extraversion": "Extraversion reflects an individual's sociability, assertiveness, and enthusiasm. High scorers are outgoing and energetic; low scorers are solitary and reserved.",
    "neuroticism": "Neuroticism reflects emotional instability and tendency to experience negative emotions. High scorers are anxious and moody; low scorers are calm and confident.",
    "agreeableness": "Agreeableness reflects an individual's tendency to be compassionate and cooperative. High scorers are trusting and helpful; low scorers are competitive and critical.",
    "conscientiousness": "Conscientiousness reflects an individual's level of self-discipline and organization. High scorers are efficient and organized; low scorers are extravagant and careless.",
    "openness": "Openness reflects an individual's intellectual curiosity and creative imagination. High scorers are inventive and curious; low scorers are consistent and cautious."
}

def get_judge_score(model, tokenizer, text, trait, device):
    trait_lower = trait.lower()
    definition = TRAIT_DEFINITIONS.get(trait_lower, "No definition available.")
    system_msg = f"""You are an expert psychologist evaluating personality traits from text.

Trait: **{trait.capitalize()}**
Definition: {definition}

Evaluate the level of {trait.capitalize()} expressed in the text.
First, analyze the text for behavioral evidence and emotional tone.
Second, determine if the text is coherent enough to evaluate.
Finally, provide a score.

Format:
Analysis: <Briefly explain the evidence or if the text is broken/repetitive>
Coherence: <Yes/No>
Score: <0-5> (0: Too broken/repetitive to evaluate, 1-5: Trait level)"""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Text: \"{text}\""}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    score_match = re.search(r'[Ss]core:\s*([0-5])', generated_text)
    score = int(score_match.group(1)) if score_match else 3
    return score, generated_text

def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
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
    return prompts[:10]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--axis", type=str, default="extraversion")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high")
    ap.add_argument("--judge_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()

    direction_mult = 1.0 if args.direction == "high" else -1.0
    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model_name")

    prompts = load_prompts(args.prompts)

    # 1. GENERATION PHASE
    missing_gens = []
    for L in LAYERS:
        for val in VALS:
            out_file = out_dir / f"layer_{L}_Val{val}.jsonl"
            if not out_file.exists():
                missing_gens.append((L, val))

    if missing_gens:
        print(f"=== Single-Layer Sweep Generation ({args.axis}) ===")
        print(f"Found {len(missing_gens)} missing generations. Loading model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
        model.eval()

        v_data = np.load(args.vector_bank)

        # Generate baseline once for all 10 prompts
        print("Generating baseline texts...")
        baselines = []
        for idx, (orig_idx, p_text) in enumerate(prompts):
            txt_b, ids_b = generate_text(model, tokenizer, p_text)
            ppl_b = calc_sequence_ppl(model, ids_b)
            baselines.append((txt_b, ppl_b))

        for L, val in tqdm(missing_gens, desc="Generating"):
            w_key = f"{L}|{args.axis}|w"
            if w_key not in v_data: continue
            w = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult

            results = []
            for idx, (orig_idx, p_text) in enumerate(prompts):
                res = {"idx": idx, "orig_idx": orig_idx, "prompt": p_text}
                
                # Baseline (cached)
                txt_b, ppl_b = baselines[idx]
                res["base_text"] = txt_b
                res["base_ppl"] = ppl_b
                
                # Constant steering
                with ConstantSteerer(model, L, w, val):
                    txt_c, ids_c = generate_text(model, tokenizer, p_text)
                    res["const_text"] = txt_c
                    res["const_ppl"] = calc_sequence_ppl(model, ids_c)
                    
                # Dummy copy to ensure compatibility with old CSV schema
                res["adapt_text"] = res["const_text"]
                res["adapt_ppl"] = res["const_ppl"]
                
                results.append(res)
                
            out_file = out_dir / f"layer_{L}_Val{val}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 2. EVALUATION PHASE
    missing_evals = []
    for L in LAYERS:
        for val in VALS:
            csv_file = out_dir / f"scores_layer_{L}_Val{val}.csv"
            if not csv_file.exists():
                jsonl_file = out_dir / f"layer_{L}_Val{val}.jsonl"
                if jsonl_file.exists():
                    missing_evals.append((L, val, jsonl_file, csv_file))

    if missing_evals:
        print(f"\n=== Single-Layer Sweep Evaluation ({args.axis}) ===")
        print(f"Found {len(missing_evals)} missing evaluations. Loading judge: {args.judge_model}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = _resolve_hf_token()
        tokenizer = AutoTokenizer.from_pretrained(args.judge_model, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model = AutoModelForCausalLM.from_pretrained(
            args.judge_model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
        model.eval()

        # Cache baseline evaluations by prompt index
        base_eval_cache = {}

        for L, val, jsonl_file, csv_file in tqdm(missing_evals, desc="Evaluating"):
            data = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
                    
            results = []
            for row in data:
                idx = row['idx']
                if idx not in base_eval_cache:
                    b_score, b_reason = get_judge_score(model, tokenizer, row['base_text'], args.axis, device)
                    base_eval_cache[idx] = (b_score, b_reason)
                else:
                    b_score, b_reason = base_eval_cache[idx]

                c_score, c_reason = get_judge_score(model, tokenizer, row['const_text'], args.axis, device)
                
                row['base_score'] = b_score
                row['const_score'] = c_score
                row['adapt_score'] = c_score
                row['base_reason'] = b_reason.replace('\n', ' ')
                row['const_reason'] = c_reason.replace('\n', ' ')
                row['adapt_reason'] = c_reason.replace('\n', ' ')
                results.append(row)
                
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)

    print("\nDONE: All layers and values evaluated.")

if __name__ == "__main__":
    main()
