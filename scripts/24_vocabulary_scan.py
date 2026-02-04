import torch
import numpy as np
import argparse
import os
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

def scan_model(model_id, axes_path, out_csv, top_k=5):
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
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    lm_head = model.get_output_embeddings()
    TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    results = []

    for trait in TRAITS:
        print(f"Scanning trait: {trait}...")
        for layer in tqdm(range(100), desc=f"Layers ({trait})"):
            key = f"{layer}|{trait}"
            if key not in data:
                if layer > 0: 
                    break
                else: 
                    continue
            
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
    
    # Cleanup to save VRAM for next model
    del model
    del tokenizer
    del lm_head
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleaned up memory.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="analysis_results/vocabulary_scans", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Defined in run_layer_sweep.sh
    # "Tag | HF_ID" (We only need the instruct ID mostly, or base? Using Instruct usually for better vocab mapping?? 
    # Actually, the vectors are extracted from BASE usually in this codebase (axes_base_asst...).
    # Let's check: the vectors used in axes_base_asst_pairwise.npz were extracted from BASE model.
    # So we should probably use the BASE model to interpret them to be perfectly consistent.
    # BUT, we often steer the INSTRUCT model. 
    # Usually base and instruct share vocabulary. Let's use Instruct model for tokenization/embedding 
    # as it's the target of steering.
    
    MODELS = [
        ("mistral_7b", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct"),
        ("olmo3_7b", "allenai/Olmo-3-7B-Instruct"),
        ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct"),
        ("gemma2_9b", "google/gemma-2-9b-it"),
        ("falcon3_7b", "tiiuae/Falcon3-7B-Instruct"),
    ]

    for tag, hf_id in MODELS:
        axes_file = f"exp/{tag}/axes_base_asst_pairwise.npz"
        out_file = os.path.join(args.out_dir, f"vocab_scan_{tag}.csv")
        
        if not os.path.exists(axes_file):
            print(f"Skipping {tag}: Axes file {axes_file} not found.")
            continue
            
        scan_model(hf_id, axes_file, out_file)

if __name__ == "__main__":
    main()
