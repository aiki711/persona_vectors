#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    
    score_match = re.search(r'[Rr]elative\s*[Ss]core:\s*([\-\+]?[0-3])', generated_text)
    score = int(score_match.group(1)) if score_match else 0
    
    return score, generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    # Base texts
    base_file = Path(f"exp_steering_layer_analysis/results/{args.trait}/layer_0_Val5.jsonl")
    base_texts = {}
    if base_file.exists():
        with open(base_file, "r") as f:
            for line in f:
                item = json.loads(line)
                base_texts[item["orig_idx"]] = item["base_text"]
    else:
        print(f"Error: Base file not found for {args.trait} at {base_file}")
        return

    # In/Out dirs
    in_dir = Path(f"exp_steering_ic_adaptive/results/{args.trait}")
    out_dir = Path(f"exp_steering_ic_adaptive/pairwise_results/{args.trait}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(list(in_dir.glob("*.jsonl")))
    if not jsonl_files:
        print(f"No jsonl files found in {in_dir}")
        return

    # Load Model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Llama-3 for {args.trait} on {device}...")
    token = _resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", token=token)
    model.eval()

    print(f"Evaluating {len(jsonl_files)} files for {args.trait}...")
    for j_path in tqdm(jsonl_files, desc=args.trait):
        csv_name = j_path.name.replace(".jsonl", "_pairwise.csv")
        csv_path = out_dir / csv_name
        
        if csv_path.exists() and csv_path.stat().st_size > 0:
            continue

        results = []
        with open(j_path, "r") as f:
            for line in f:
                item = json.loads(line)
                b_text = base_texts.get(item["orig_idx"], "")
                if not b_text: continue
                
                s_text = item["ic_adapt_text"]
                ppl = item.get("ic_adapt_ppl", None)
                
                score, reasoning = get_pairwise_score(model, tokenizer, b_text, s_text, args.trait, device)
                
                results.append({
                    "orig_idx": item["orig_idx"],
                    "prompt": item["prompt"],
                    "base_text": b_text,
                    "steered_text": s_text,
                    "base_ppl": item.get("base_ppl", None), # May not exist in IC file, but can be loaded if needed. Actually we didn't save base_ppl in IC files.
                    "ic_adapt_ppl": ppl,
                    "pairwise_score": score,
                    "reasoning": reasoning
                })
                
        # Load base_ppl from base_file just to be sure
        base_ppls = {}
        with open(base_file, "r") as f:
             for line in f:
                 b_item = json.loads(line)
                 base_ppls[b_item["orig_idx"]] = b_item["base_ppl"]
        
        for r in results:
             r["base_ppl"] = base_ppls.get(r["orig_idx"], None)

        if results:
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()
