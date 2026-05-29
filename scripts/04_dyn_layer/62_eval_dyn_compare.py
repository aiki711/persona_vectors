#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 62_eval_dyn_compare.py

import argparse
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from persona_vectors.live_axes import _resolve_hf_token, load_model_and_tokenizer

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
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": f'Text: "{text}"'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"[Ss]core:\s*([0-5])", gen)
    return int(m.group(1)) if m else 3, gen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--quant", default="auto", choices=["auto", "8bit", "4bit", "none"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_val = None if args.quant == "none" else args.quant
    model, tokenizer = load_model_and_tokenizer(args.model, quant=quant_val)
    model.eval()

    data = [json.loads(line) for line in open(args.input, "r", encoding="utf-8") if line.strip()]

    results = []
    for row in tqdm(data):
        text_key = "fusion_text" if "fusion_text" in row else "dyn_text"
        ppl_key = "fusion_ppl" if "fusion_ppl" in row else "dyn_ppl"
        
        b_score, b_reason = get_score(model, tokenizer, row.get("base_text", ""), args.axis, device) if row.get("base_text") else (3, "N/A")
        dyn_score, dyn_reason = get_score(model, tokenizer, row.get(text_key, ""), args.axis, device)
        results.append({
            "idx": row.get("idx", 0),
            "base_score": b_score, "base_ppl": row.get("base_ppl", float("nan")),
            "dyn_score": dyn_score, "dyn_ppl": row.get(ppl_key, float("nan")),
            "base_reason": b_reason.replace("\n", " "), "dyn_reason": dyn_reason.replace("\n", " "),
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n--- Averages ---")
    print(f"  Base : PPL={df['base_ppl'].mean():.2f}, Score={df['base_score'].mean():.2f}")
    print(f"  DLS  : PPL={df['dyn_ppl'].mean():.2f},  Score={df['dyn_score'].mean():.2f}")

if __name__ == "__main__":
    main()
