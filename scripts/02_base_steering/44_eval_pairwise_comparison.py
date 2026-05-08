#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 44_eval_pairwise_comparison.py
#
# Relative evaluation: Compare "Base Text" vs "Steered Text".
# Uses Llama-3-8B-Instruct to determine which one better reflects the trait.
#

import argparse
import json
import pandas as pd
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from persona_vectors.live_axes import _resolve_hf_token

TRAIT_DEFINITIONS = {
    "extraversion": "Extraversion reflects personality traits such as being outgoing, talkative, high on energy, and assertiveness.",
    "neuroticism": "Neuroticism reflects emotional instability and tendency to experience negative emotions. High scorers are anxious and moody; low scorers are calm and confident.",
    "openness": "Openness to experience involves a willingness to try new things and a fascination for new ideas. High scorers are creative and imaginative.",
    "agreeableness": "Agreeableness reflects an individual's tendency to be compassionate and cooperative. High scorers are trusting and helpful; low scorers are competitive and critical.",
    "conscientiousness": "Conscientiousness reflects an individual's level of self-discipline and organization. High scorers are efficient and organized; low scorers are extravagant and careless."
}

def get_pairwise_score(model, tokenizer, base_text, steered_text, trait, device):
    definition = TRAIT_DEFINITIONS.get(trait.lower(), "No definition available.")
    
    system_msg = f"""You are an expert psychologist specialized in comparative personality analysis.
Your task is to compare two text snippets (A and B) and determine which one expresses a higher level of the personality trait: **{trait.capitalize()}**.

Definition of {trait.capitalize()}: {definition}

Instructions:
1. Analyze both texts for subtle cues, emotional tone, and behavioral markers.
2. Determine if Text B shows an INCREASE, DECREASE, or NO CHANGE in {trait.capitalize()} compared to Text A.
3. Provide a Relative Score on a 7-point scale:
   +3: Text B is significantly higher in {trait.capitalize()} than Text A.
   +2: Text B is higher in {trait.capitalize()} than Text A.
   +1: Text B is slightly higher in {trait.capitalize()} than Text A.
    0: No detectable difference between Text A and Text B.
   -1: Text B is slightly lower in {trait.capitalize()} than Text A (shifted to the opposite trait).
   -2: Text B is lower in {trait.capitalize()} than Text A.
   -3: Text B is significantly lower in {trait.capitalize()} than Text A.

Format:
Reasoning: <A few sentences comparing the two texts>
Relative Score: <An integer from -3 to +3>"""

    user_content = f"Text A (Baseline): \"{base_text}\"\n\nText B (Steered): \"{steered_text}\"\n\nComparison:"
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=300, 
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Parse Score (searching for "Relative Score: X")
    score_match = re.search(r'[Rr]elative\s*[Ss]core:\s*([\-\+]?[0-3])', generated_text)
    score = int(score_match.group(1)) if score_match else 0
    
    return score, generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL result file")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--axis", required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    token = _resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", token=token)
    model.eval()

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    results = []
    for row in tqdm(data, desc=f"Pairwise [{args.axis}]"):
        # Evaluate Constant
        score_c, reason_c = get_pairwise_score(model, tokenizer, row['base_text'], row['const_text'], args.axis, device)
        # Evaluate Adaptive
        score_a, reason_a = get_pairwise_score(model, tokenizer, row['base_text'], row['adapt_text'], args.axis, device)
        
        results.append({
            "idx": row["idx"],
            "orig_idx": row.get("orig_idx", ""),
            "const_shift": score_c,
            "adapt_shift": score_a,
            "const_reason": reason_c.replace('\n', ' '),
            "adapt_reason": reason_a.replace('\n', ' '),
            "const_ppl": row["const_ppl"],
            "adapt_ppl": row["adapt_ppl"]
        })

    df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved pairwise results to {args.output}")

if __name__ == "__main__":
    main()
