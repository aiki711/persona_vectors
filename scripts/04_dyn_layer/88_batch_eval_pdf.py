#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/88_batch_eval_pdf.py
#
# Batch evaluates all PDF dynamic steering results for a specific trait axis.
# Loads the 70B judge model once and loops over all alphas and methods.
#

import argparse
import json
import re
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from persona_vectors.live_axes import load_model_and_tokenizer

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
    
    score_val = 3
    idx = gen.lower().find("score:")
    if idx != -1:
        m = re.search(r"([0-5])", gen[idx + 6:])
        if m:
            score_val = int(m.group(1))
    return score_val, gen

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, help="Personality trait axis to evaluate")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-70B-Instruct")
    ap.add_argument("--results_dir", default="exp_steering_dyn_layer_pdf/results")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    
    trait = args.axis.lower()
    if trait not in TRAITS:
        print(f"Error: unknown axis '{trait}'")
        return

    results_dir = Path(args.results_dir)
    if args.methods:
        methods_to_eval = args.methods
    else:
        methods_to_eval = ["masked_cos_only", "masked_rank_only", "masked_proj_cos_only", "masked_proj_rank_only", "masked_proj_cos_prior", "masked_proj_rank_prior"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading judge model {args.model} for axis {trait}...")
    model, tokenizer = load_model_and_tokenizer(args.model, quant="4bit")
    model.eval()
    
    for val in VALS:
        for method in methods_to_eval:
            jsonl_in = results_dir / trait / f"{method}_Val{val}.jsonl"
            if not jsonl_in.exists():
                jsonl_in = results_dir / trait / f"{method}_Val{float(val)}.jsonl"
            
            if not jsonl_in.exists():
                print(f"Warning: skipped missing input file {jsonl_in}")
                continue
                
            csv_out = results_dir / trait / f"scores_{method}_Val{val}.csv"
            if csv_out.exists():
                print(f"Already evaluated: {csv_out}")
                continue
                
            print(f"\n--- Evaluating {trait} - {method} - Val {val} ---")
            
            # Load JSONL
            data = [json.loads(line) for line in open(jsonl_in, "r", encoding="utf-8") if line.strip()]
            results = []
            for row in tqdm(data):
                text_key = "fusion_text" if "fusion_text" in row else "dyn_text"
                ppl_key = "fusion_ppl" if "fusion_ppl" in row else "dyn_ppl"
                
                b_score, b_reason = get_score(model, tokenizer, row.get("base_text", ""), trait, device) if row.get("base_text") else (3, "N/A")
                dyn_score, dyn_reason = get_score(model, tokenizer, row.get(text_key, ""), trait, device)
                results.append({
                    "idx": row.get("idx", 0),
                    "base_score": b_score, "base_ppl": row.get("base_ppl", float("nan")),
                    "dyn_score": dyn_score, "dyn_ppl": row.get(ppl_key, float("nan")),
                    "base_reason": b_reason.replace("\n", " "), "dyn_reason": dyn_reason.replace("\n", " "),
                })
            df = pd.DataFrame(results)
            df.to_csv(csv_out, index=False)
            print(f"Saved to {csv_out}")

if __name__ == "__main__":
    main()
