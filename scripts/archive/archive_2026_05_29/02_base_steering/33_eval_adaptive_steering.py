#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 33_eval_adaptive_steering.py
#
# Evaluates the personality scores of base, constant, and adaptive text outputs
# using an LLM-as-a-judge (Meta-Llama-3-8B-Instruct).

import argparse
import json
import pandas as pd
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

from persona_vectors.live_axes import _resolve_hf_token

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
            max_new_tokens=256, # Increase to allow reasoning
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Parse Score (searching for "Score: X")
    score_match = re.search(r'[Ss]core:\s*([0-5])', generated_text)
    score = int(score_match.group(1)) if score_match else 3
    
    return score, generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to adaptive JSONL file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--axis", required=True, help="The personality axis (e.g., extraversion)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading Judge Model: {args.model}...")
    token = _resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16,
        device_map="auto",
        token=token
    )
    model.eval()

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Evaluating {len(data)} items for trait {args.axis}...")
    
    results = []
    for row in tqdm(data):
        b_score, b_reason = get_judge_score(model, tokenizer, row['base_text'], args.axis, device)
        c_score, c_reason = get_judge_score(model, tokenizer, row['const_text'], args.axis, device)
        a_score, a_reason = get_judge_score(model, tokenizer, row['adapt_text'], args.axis, device)
        
        row['base_score'] = b_score
        row['const_score'] = c_score
        row['adapt_score'] = a_score
        # 理由も保存する（カンマなどが CSV を壊さないように改行などを置換）
        row['base_reason'] = b_reason.replace('\n', ' ')
        row['const_reason'] = c_reason.replace('\n', ' ')
        row['adapt_reason'] = a_reason.replace('\n', ' ')
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    print("\n--- Averages ---")
    print(f"Base    - PPL: {df['base_ppl'].mean():.2f}, Score: {df['base_score'].mean():.2f}")
    print(f"Const   - PPL: {df['const_ppl'].mean():.2f}, Score: {df['const_score'].mean():.2f}")
    print(f"Adapt   - PPL: {df['adapt_ppl'].mean():.2f}, Score: {df['adapt_score'].mean():.2f}")
    
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
