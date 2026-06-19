#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 115_batch_eval.py
#
# Caches evaluations and runs the Llama-3-70B judge model on all generated jsonl files
# under a directory under a single model load.
#

import argparse
import json
import re
import torch
import pandas as pd
from pathlib import Path
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
    if not text or not text.strip():
        return 3, "Empty text"
        
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Path to the results directory (e.g. exp_steering_dyn_layer_raw/results/extraversion)")
    ap.add_argument("--axis", required=True, help="Personality trait (e.g. extraversion)")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-70B-Instruct")
    ap.add_argument("--quant", default="4bit", choices=["auto", "8bit", "4bit", "none"])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Directory not found: {results_dir}")
        return

    # Find all jsonl files that do not have completed csv files
    jsonl_files = sorted(list(results_dir.glob("*.jsonl")))
    pending_files = []
    for f in jsonl_files:
        csv_file = f.with_name(f"scores_{f.stem}.csv")
        if not csv_file.exists():
            pending_files.append(f)

    if not pending_files:
        print(f"All {len(jsonl_files)} files in {results_dir} already have completed evaluations.")
        return

    print(f"Found {len(pending_files)} pending evaluations out of {len(jsonl_files)} files in {results_dir}.")

    # Collect all unique texts that need evaluation to prevent duplicate calls
    unique_texts = set()
    file_data_map = {}

    for f in pending_files:
        try:
            data = [json.loads(line) for line in open(f, "r", encoding="utf-8") if line.strip()]
            file_data_map[f] = data
            for row in data:
                text_key = "fusion_text" if "fusion_text" in row else "dyn_text"
                if row.get("base_text"):
                    unique_texts.add(row["base_text"])
                if row.get(text_key):
                    unique_texts.add(row[text_key])
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    unique_list = sorted(list(unique_texts))
    print(f"Total unique texts to evaluate: {len(unique_list)}")

    if not unique_list:
        print("No texts found to evaluate.")
        return

    # Load judge model
    print(f"Loading judge model: {args.model} ({args.quant})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_val = None if args.quant == "none" else args.quant
    model, tokenizer = load_model_and_tokenizer(args.model, quant=quant_val)
    model.eval()

    # Evaluate all unique texts
    cache = {}
    print("Evaluating unique texts...")
    for text in tqdm(unique_list):
        score, reason = get_score(model, tokenizer, text, args.axis, device)
        cache[text] = (score, reason)

    # Save results to corresponding csv files
    print("Writing CSV files...")
    for f, data in file_data_map.items():
        results = []
        for row in data:
            text_key = "fusion_text" if "fusion_text" in row else "dyn_text"
            ppl_key = "fusion_ppl" if "fusion_ppl" in row else "dyn_ppl"
            
            b_text = row.get("base_text", "")
            dyn_text = row.get(text_key, "")
            
            b_score, b_reason = cache.get(b_text, (3, "N/A")) if b_text else (3, "N/A")
            dyn_score, dyn_reason = cache.get(dyn_text, (3, "N/A")) if dyn_text else (3, "N/A")
            
            results.append({
                "idx": row.get("idx", 0),
                "base_score": b_score, "base_ppl": row.get("base_ppl", float("nan")),
                "dyn_score": dyn_score, "dyn_ppl": row.get(ppl_key, float("nan")),
                "base_reason": b_reason.replace("\n", " "), "dyn_reason": dyn_reason.replace("\n", " "),
            })
            
        df = pd.DataFrame(results)
        csv_file = f.with_name(f"scores_{f.stem}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")

    print("\nBatch evaluation completed successfully.")

if __name__ == "__main__":
    main()
