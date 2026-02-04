import torch
import numpy as np
import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Project steering vectors to vocabulary to interpret their meaning.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID (e.g. mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--axes_path", type=str, required=True, help="Path to .npz file containing steering vectors")
    parser.add_argument("--layer", type=int, default=15, help="Layer index to inspect (default: 15)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top tokens to show")
    parser.add_argument("--trait", type=str, default="openness", help="Trait to inspect")
    
    args = parser.parse_args()

    print(f"Loading vectors from {args.axes_path}...")
    try:
        data = np.load(args.axes_path)
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return

    # Construct key: "layer|trait"
    key = f"{args.layer}|{args.trait}"
    if key not in data:
        print(f"Key '{key}' not found in {args.axes_path}")
        print(f"Available keys (sample): {list(data.keys())[:5]} ...")
        return

    vec = data[key]
    print(f"Vector shape: {vec.shape}")
    
    # Convert to torch tensor
    # Ensure float32 for model compatibility initially
    vec_tensor = torch.tensor(vec, dtype=torch.float32)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Load model explicitly on GPU if available, else CPU. 
        # avoiding 'auto' to prevent meta device issues for this simple projection script
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Projecting vector to vocabulary...")
    
    # The "meaning" of a direction in the residual stream (at later layers) 
    # is often interpreted by projecting it onto the output vocabulary.
    # Logits = x @ W_U.T
    # Here x is our steering vector.
    
    lm_head = model.get_output_embeddings()
    # W_U shape: [vocab_size, hidden_size]
    # vec shape: [hidden_size]
    
    # Move vec to same device/dtype as weight
    vec_tensor = vec_tensor.to(lm_head.weight.device).to(lm_head.weight.dtype)
    
    # Project
    # logits = vec @ W_U.T
    logits = torch.matmul(lm_head.weight, vec_tensor)
    
    # Get Top-K
    top_values, top_indices = torch.topk(logits, args.top_k)
    
    # Get Bottom-K (what does it suppress?)
    bot_values, bot_indices = torch.topk(logits, args.top_k, largest=False)
    
    print(f"\n=== Top {args.top_k} Tokens for '{args.trait}' at Layer {args.layer} ===")
    print(f"{'Token':<20} | {'Logit':<10}")
    print("-" * 35)
    for idx, val in zip(top_indices, top_values):
        token_str = tokenizer.decode([idx.item()]).strip()
        # Handle newlines/special chars for display
        token_str = repr(token_str)
        print(f"{token_str:<20} | {val.item():.4f}")

    print(f"\n=== Bottom {args.top_k} Tokens (Suppressed) ===")
    print(f"{'Token':<20} | {'Logit':<10}")
    print("-" * 35)
    for idx, val in zip(bot_indices, bot_values):
        token_str = tokenizer.decode([idx.item()]).strip()
        token_str = repr(token_str)
        print(f"{token_str:<20} | {val.item():.4f}")

    print("\nanalysis complete.")

if __name__ == "__main__":
    main()
