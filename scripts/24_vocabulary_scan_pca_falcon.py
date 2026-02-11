
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
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")
        
        # Load model in 8bit or similar if possible to save mem? 
        # Original script loaded float16. Falcon 7B fits in 64GB mem easily.
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
    TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    results = []

    for trait in TRAITS:
        print(f"Scanning trait: {trait}...")
        # Check available layers in data
        keys = [k for k in data.files if k.endswith(f"|{trait}")]
        layers = sorted([int(k.split('|')[0]) for k in keys])
        
        for layer in tqdm(layers, desc=f"Layers ({trait})"):
            key = f"{layer}|{trait}"
            vec = data[key]
            vec_tensor = torch.tensor(vec, dtype=model.dtype, device=device)
            
            with torch.no_grad():
                logits = torch.matmul(lm_head.weight, vec_tensor)
                
                # Top K
                top_v, top_i = torch.topk(logits, top_k)
                tokens_top = [repr(tokenizer.decode([idx.item()]).strip()) for idx in top_i]
                top_str = ", ".join(tokens_top)

                # Bottom K
                bot_v, bot_i = torch.topk(logits, top_k, largest=False)
                tokens_bot = [repr(tokenizer.decode([idx.item()]).strip()) for idx in bot_i]
                bot_str = ", ".join(tokens_bot)
                
                results.append({
                    "Trait": trait,
                    "Layer": layer,
                    "Top_Tokens": top_str,
                    "Bottom_Tokens": bot_str
                })

    print(f"Saving results to {out_csv}...")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Trait", "Layer", "Top_Tokens", "Bottom_Tokens"])
        writer.writeheader()
        writer.writerows(results)
    
    print("Cleaned up memory.")

def main():
    os.makedirs("analysis_results/vocabulary_scans", exist_ok=True)
    
    scan_model(
        model_id="tiiuae/Falcon3-7B-Instruct",
        axes_path="exp_pca/falcon3_7b/axes_base_pca_asst.npz",
        out_csv="analysis_results/vocabulary_scans/vocab_scan_falcon3_7b_pca.csv"
    )

if __name__ == "__main__":
    main()
