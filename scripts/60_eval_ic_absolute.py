#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 60_eval_ic_absolute.py
#
# Absolute score evaluation for IC-Adaptive generated text using Llama-3.
# Input:  ic_adapt_layer{L}_Tau{tau}_S1.5.jsonl
#         Fields: base_text (from baseline), ic_adapt_text, ic_adapt_ppl
# Output: scores_ic_adapt_layer{L}_Tau{tau}.csv
#         Fields: base_score, base_ppl, ic_score, ic_ppl

import argparse
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from persona_vectors.live_axes import _resolve_hf_token

TRAIT_DEFINITIONS = {
    "extraversion": "Extraversion reflects an individual's sociability, assertiveness, and enthusiasm. High scorers are outgoing and energetic; low scorers are solitary and reserved.",
    "neuroticism": "Neuroticism reflects emotional instability and tendency to experience negative emotions. High scorers are anxious and moody; low scorers are calm and confident.",
    "agreeableness": "Agreeableness reflects an individual's tendency to be compassionate and cooperative. High scorers are trusting and helpful; low scorers are competitive and critical.",
    "conscientiousness": "Conscientiousness reflects an individual's level of self-discipline and organization. High scorers are efficient and organized; low scorers are extravagant and careless.",
    "openness": "Openness reflects an individual's intellectual curiosity and creative imagination. High scorers are inventive and curious; low scorers are consistent and cautious.",
}


def get_score(model, tokenizer, text, trait, device):
    definition = TRAIT_DEFINITIONS.get(trait.lower(), "")
    system_msg = (
        f"You are an expert psychologist evaluating personality traits from text.\n\n"
        f"Trait: **{trait.capitalize()}**\n"
        f"Definition: {definition}\n\n"
        f"Evaluate the level of {trait.capitalize()} expressed in the text.\n"
        f"First, analyze the text for behavioral evidence and emotional tone.\n"
        f"Second, determine if the text is coherent enough to evaluate.\n"
        f"Finally, provide a score.\n\n"
        f"Format:\nAnalysis: <brief explanation>\nCoherence: <Yes/No>\nScore: <0-5>"
    )
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
    gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"[Ss]core:\s*([0-5])", gen)
    score = int(m.group(1)) if m else 3
    return score, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Path to ic_adapt_layer*_Tau*.jsonl")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument("--axis",   required=True, help="Personality axis")
    ap.add_argument("--model",  default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = _resolve_hf_token()

    print(f"Loading judge: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", token=token
    )
    model.eval()

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # Determine which field holds the IC-Adaptive text
    ic_text_key = "ic_adapt_text"
    ic_ppl_key  = "ic_adapt_ppl"

    print(f"Evaluating {len(data)} items for [{args.axis}]...")
    results = []
    for row in tqdm(data):
        # Baseline score (base_text may not be present in older JSONL; handle gracefully)
        base_text = row.get("base_text", "")
        if base_text:
            b_score, b_reason = get_score(model, tokenizer, base_text, args.axis, device)
        else:
            b_score, b_reason = 3, "N/A"

        ic_text = row.get(ic_text_key, "")
        ic_score, ic_reason = get_score(model, tokenizer, ic_text, args.axis, device)

        results.append({
            "idx":       row.get("idx", 0),
            "base_score": b_score,
            "base_ppl":   row.get("base_ppl", float("nan")),
            "ic_score":   ic_score,
            "ic_ppl":     row.get(ic_ppl_key, float("nan")),
            "base_reason": b_reason.replace("\n", " "),
            "ic_reason":   ic_reason.replace("\n", " "),
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n--- Averages ---")
    print(f"  Base : PPL={df['base_ppl'].mean():.2f}, Score={df['base_score'].mean():.2f}")
    print(f"  IC   : PPL={df['ic_ppl'].mean():.2f},  Score={df['ic_score'].mean():.2f}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
