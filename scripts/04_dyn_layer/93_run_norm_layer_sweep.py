#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 93_run_norm_layer_sweep.py
#
# Single-layer steering sweep with steering vector normalized to each layer's
# raw difference vector norm (||w_steered|| = ||w_raw||).
# This matches the norm_mode="raw_norm" used in 82_run_dyn_layer_proj_prior.py,
# enabling a fair comparison between fixed single-layer and dynamic layer methods.
#
# Output: exp_steering_layer_norm/results/{axis}/layer_{L}_Val{alpha}.jsonl
#         exp_steering_layer_norm/results/{axis}/scores_layer_{L}_Val{alpha}.csv
#

import argparse
import json
import re
import torch
import numpy as np
import yaml
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
    _resolve_hf_token,
)

LAYERS = list(range(32))
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]


class ConstantSteerer:
    """Applies constant alpha * w to the output of a specific layer (decode step only)."""

    def __init__(self, model, layer, w, alpha):
        self.model = model
        self.layer = layer
        self.w = w
        self.alpha = alpha
        self.handle = None

    def __enter__(self):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(hs).all():
                return out
            if hs.size(1) != 1:
                return out  # Skip prefill
            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            w_dev = self.w.to(hs.device)
            steered = hs_f32 + self.alpha * w_dev.view(1, 1, -1)
            if not torch.isfinite(steered).all():
                return out
            return (steered.to(orig_dtype), *out[1:]) if isinstance(out, tuple) else steered.to(orig_dtype)

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


TRAIT_DEFINITIONS = {
    "extraversion": "Extraversion reflects an individual's sociability, assertiveness, and enthusiasm. High scorers are outgoing and energetic; low scorers are solitary and reserved.",
    "neuroticism": "Neuroticism reflects emotional instability and tendency to experience negative emotions. High scorers are anxious and moody; low scorers are calm and confident.",
    "agreeableness": "Agreeableness reflects an individual's tendency to be compassionate and cooperative. High scorers are trusting and helpful; low scorers are competitive and critical.",
    "conscientiousness": "Conscientiousness reflects an individual's level of self-discipline and organization. High scorers are efficient and organized; low scorers are extravagant and careless.",
    "openness": "Openness reflects an individual's intellectual curiosity and creative imagination. High scorers are inventive and curious; low scorers are consistent and cautious.",
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
        {"role": "user", "content": f'Text: "{text}"'},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    score_match = re.search(r"[Ss]core:\s*([0-5])", generated_text)
    score = int(score_match.group(1)) if score_match else 3
    return score, generated_text


def load_prompts(path, num_prompts=10):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                item = json.loads(line)
            except Exception:
                item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                prompts.append((item.get("orig_idx", ""), item["input"]))
            elif isinstance(item, str):
                prompts.append(("", item))
    return prompts[:num_prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--vector_bank", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--axis", type=str, default="extraversion")
    ap.add_argument("--direction", type=str, choices=["high", "low"], default="high")
    ap.add_argument("--judge_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--judge_quant", type=str, choices=["auto", "8bit", "4bit", "none"], default="none")
    ap.add_argument("--num_prompts", type=int, default=10, help="Number of prompts to evaluate")
    args = ap.parse_args()

    direction_mult = 1.0 if args.direction == "high" else -1.0
    out_dir = Path(args.out_dir) / args.axis
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model_name")

    prompts = load_prompts(args.prompts, num_prompts=args.num_prompts)

    # Load vector bank (contains 'w', 'raw_norm', and 'midpoint' keys per layer)
    v_data = np.load(args.vector_bank)

    # Build norm-scaled steering vectors: scale to raw difference vector norm (raw_norm), fallback to midpoint
    layer_w_normed = {}
    for L in LAYERS:
        w_key = f"{L}|{args.axis}|w"
        raw_norm_key = f"{L}|{args.axis}|raw_norm"
        mp_key = f"{L}|{args.axis}|midpoint"
        if w_key not in v_data:
            continue
        w_vec = torch.tensor(v_data[w_key], dtype=torch.float32) * direction_mult
        w_norm = torch.norm(w_vec).item()
        
        if raw_norm_key in v_data:
            r_norm = float(v_data[raw_norm_key][0])
            w_normed = (w_vec / (w_norm + 1e-10)) * r_norm
        elif mp_key in v_data:
            m_vec = torch.tensor(v_data[mp_key], dtype=torch.float32)
            m_norm = torch.norm(m_vec).item()
            w_normed = (w_vec / (w_norm + 1e-10)) * m_norm
        else:
            # Fallback: keep raw w if no norm source available
            w_normed = w_vec
        layer_w_normed[L] = w_normed

    # --- GENERATION PHASE ---
    missing_gens = []
    for L in LAYERS:
        for val in VALS:
            out_file = out_dir / f"layer_{L}_Val{val}.jsonl"
            if not out_file.exists():
                missing_gens.append((L, val))

    if missing_gens:
        print(f"=== Norm-Scaled Single-Layer Sweep Generation ({args.axis}) ===")
        print(f"Found {len(missing_gens)} missing generations. Loading model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, quant=cfg.get("quant", "auto"))
        model.eval()

        # Print norm comparison for verification
        print("Steering vector norm (scaled to raw norm / midpoint) per layer:")
        for L in sorted(layer_w_normed.keys()):
            raw_w_norm = np.linalg.norm(v_data[f"{L}|{args.axis}|w"])
            scaled_norm = torch.norm(layer_w_normed[L]).item()
            raw_norm_key = f"{L}|{args.axis}|raw_norm"
            mp_key = f"{L}|{args.axis}|midpoint"
            ref_norm = float(v_data[raw_norm_key][0]) if raw_norm_key in v_data else (np.linalg.norm(v_data[mp_key]) if mp_key in v_data else float("nan"))
            print(f"  Layer {L:2d}: raw_w_norm={raw_w_norm:.3f}, ref_norm={ref_norm:.3f}, scaled_w_norm={scaled_norm:.3f}")

        # Generate baseline once
        print("Generating baseline texts...")
        baselines = []
        for idx, (orig_idx, p_text) in enumerate(prompts):
            txt_b, ids_b = generate_text(model, tokenizer, p_text)
            ppl_b = calc_sequence_ppl(model, ids_b)
            baselines.append((txt_b, ppl_b))

        for L, val in tqdm(missing_gens, desc="Generating"):
            if L not in layer_w_normed:
                continue
            w = layer_w_normed[L]

            results = []
            for idx, (orig_idx, p_text) in enumerate(prompts):
                res = {"idx": idx, "orig_idx": orig_idx, "prompt": p_text}

                txt_b, ppl_b = baselines[idx]
                res["base_text"] = txt_b
                res["base_ppl"] = ppl_b

                with ConstantSteerer(model, L, w, val):
                    txt_c, ids_c = generate_text(model, tokenizer, p_text)
                    res["const_text"] = txt_c
                    res["const_ppl"] = calc_sequence_ppl(model, ids_c)

                res["adapt_text"] = res["const_text"]
                res["adapt_ppl"] = res["const_ppl"]

                results.append(res)

            out_file = out_dir / f"layer_{L}_Val{val}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # --- EVALUATION PHASE ---
    missing_evals = []
    for L in LAYERS:
        for val in VALS:
            csv_file = out_dir / f"scores_layer_{L}_Val{val}.csv"
            if not csv_file.exists():
                jsonl_file = out_dir / f"layer_{L}_Val{val}.jsonl"
                if jsonl_file.exists():
                    missing_evals.append((L, val, jsonl_file, csv_file))

    if missing_evals:
        print(f"\n=== Norm-Scaled Single-Layer Sweep Evaluation ({args.axis}) ===")
        print(f"Found {len(missing_evals)} missing evaluations. Loading judge: {args.judge_model}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quant_val = None if args.judge_quant == "none" else args.judge_quant
        model, tokenizer = load_model_and_tokenizer(args.judge_model, quant=quant_val)
        model.eval()

        base_eval_cache = {}

        for L, val, jsonl_file, csv_file in tqdm(missing_evals, desc="Evaluating"):
            data = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))

            results = []
            for row in data:
                idx = row["idx"]
                if idx not in base_eval_cache:
                    b_score, b_reason = get_judge_score(model, tokenizer, row["base_text"], args.axis, device)
                    base_eval_cache[idx] = (b_score, b_reason)
                else:
                    b_score, b_reason = base_eval_cache[idx]

                c_score, c_reason = get_judge_score(model, tokenizer, row["const_text"], args.axis, device)

                row["base_score"] = b_score
                row["const_score"] = c_score
                row["adapt_score"] = c_score
                row["base_reason"] = b_reason.replace("\n", " ")
                row["const_reason"] = c_reason.replace("\n", " ")
                row["adapt_reason"] = c_reason.replace("\n", " ")
                results.append(row)

            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)
            print(f"  Saved: {csv_file}")

    print("\nDONE: All layers and values evaluated (norm-scaled steering).")


if __name__ == "__main__":
    main()
