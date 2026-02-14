import torch
import numpy as np
import argparse
import os
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

def scan_model(model_id, axes_path, out_csv, top_k=10):
    print(f"=== Scanning Model: {model_id} ===")
    print(f"Loading vectors from {axes_path}...")
    try:
        data = np.load(axes_path)
        # Check keys to infer keys format if needed, but assuming layer|trait
        print(f"Keys in NPZ: {list(data.keys())[:5]} ...")
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    lm_head = model.get_output_embeddings()
    # Traits to scan (match keys in npz)
    TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    results = []

    print("Starting scan...")
    # Iterate through all keys in data to handle any layer range
    # Or iterate standard range
    
    # Let's iterate traits and check available layers
    for trait in TRAITS:
        print(f"Scanning trait: {trait}...")
        found_layers = 0
        for layer in range(100): # Scan up to 100 layers
            key = f"{layer}|{trait}"
            if key not in data:
                continue
            
            found_layers += 1
            vec = data[key]
            vec_tensor = torch.tensor(vec, dtype=model.dtype, device=device)
            
            # Normalize vector for consistent cosine sim analysis? 
            # Usually vocabulary projection is just dot product with embedding matrix.
            # Logits = Valid token probabilities direction.
            
            with torch.no_grad():
                # (Vocab, Dim) x (Dim) -> (Vocab)
                logits = torch.matmul(lm_head.weight, vec_tensor)
                
                # Top K (Positive Direction)
                top_v, top_i = torch.topk(logits, top_k)
                tokens_top = [repr(tokenizer.decode([idx.item()]).strip()) for idx in top_i]
                top_str = ", ".join(tokens_top)

                # Bottom K (Negative Direction)
                bot_v, bot_i = torch.topk(logits, top_k, largest=False)
                tokens_bot = [repr(tokenizer.decode([idx.item()]).strip()) for idx in bot_i]
                bot_str = ", ".join(tokens_bot)
                
                results.append({
                    "Trait": trait,
                    "Layer": layer,
                    "Top_Tokens": top_str,
                    "Bottom_Tokens": bot_str
                })
        print(f"  Found {found_layers} vectors for {trait}")

    print(f"Saving results to {out_csv}...")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Trait", "Layer", "Top_Tokens", "Bottom_Tokens"])
        writer.writeheader()
        writer.writerows(results)
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--axes_path", type=str, required=True, help="Path to .npz vector file")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()

    scan_model(args.model_id, args.axes_path, args.out_csv)

if __name__ == "__main__":
    main()
