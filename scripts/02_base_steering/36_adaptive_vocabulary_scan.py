#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 36_adaptive_vocabulary_scan.py
#
# Project the learned SVM boundary normal vectors (w) to the vocabulary space
# using the LM head of the un-embedding matrix. This reveals what tokens the
# SVM considers the most indicative of each personality trait (High vs Low).

import torch
import numpy as np
import argparse
import os
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

def scan_model(model_name, boundary_bank, out_csv, top_k=20):
    print(f"=== Scanning Boundary Vectors: {boundary_bank} ===")
    print(f"Loading vectors from {boundary_bank}...")
    try:
        data = np.load(boundary_bank)
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract the un-embedding language modeling head
    lm_head = model.get_output_embeddings()
    TRAITS = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
    results = []

    print("\n[Projecting Vectors to Vocabulary Space]")
    for trait in TRAITS:
        for layer in tqdm(range(32), desc=f"Scanning {trait}"):
            w_key = f"{layer}|{trait}|w"
            if w_key not in data:
                continue
            
            # (H,) dimension vector
            vec = data[w_key]
            # Convert to appropriate tensor matching the LM head dtype
            vec_tensor = torch.tensor(vec, dtype=lm_head.weight.dtype, device=lm_head.weight.device)
            
            with torch.no_grad():
                # logits = W_{vocab, H} @ vec_{H}
                logits = torch.matmul(lm_head.weight, vec_tensor)
                
                # Top K (Positive extreme -> strongly drives the trait "High")
                top_v, top_i = torch.topk(logits, top_k)
                tokens_top = [repr(tokenizer.decode([idx.item()]).strip()) for idx in top_i]
                top_str = ", ".join(tokens_top)

                # Bottom K (Negative extreme -> strongly drives the trait "Low")
                bot_v, bot_i = torch.topk(logits, top_k, largest=False)
                tokens_bot = [repr(tokenizer.decode([idx.item()]).strip()) for idx in bot_i]
                bot_str = ", ".join(tokens_bot)
                
                results.append({
                    "Trait": trait,
                    "Layer": layer,
                    "Top_Tokens (High)": top_str,
                    "Bottom_Tokens (Low)": bot_str
                })

    print(f"\nSaving results to {out_csv}...")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Trait", "Layer", "Top_Tokens (High)", "Bottom_Tokens (Low)"])
        writer.writeheader()
        writer.writerows(results)
    
    # Cleanup memory
    del model
    del tokenizer
    del lm_head
    torch.cuda.empty_cache()
    gc.collect()
    print("[Done] Cleaned up memory.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model for LM head")
    parser.add_argument("--boundary_bank", type=str, default="exp_adaptive_steering/vectors/boundary_vectors.npz", help="Path to boundary vectors")
    parser.add_argument("--out_csv", type=str, default="exp_adaptive_steering/results/adaptive_vocab_scan.csv", help="Output CSV path")
    parser.add_argument("--top_k", type=int, default=20, help="Number of tokens to extract per pole")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    scan_model(args.model, args.boundary_bank, args.out_csv, top_k=args.top_k)

if __name__ == "__main__":
    main()
