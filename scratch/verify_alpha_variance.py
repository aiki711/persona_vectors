import torch
import numpy as np
import yaml
from pathlib import Path
from persona_vectors.live_axes import load_model_and_tokenizer, get_layer_stack, _format_prompt

def verify_alpha_variance():
    # Setup
    config_path = "config/mistral_7b.yaml"
    vector_bank = "exp_steering_layer_sweep_1-40/vectors/mean_diff_vectors.npz"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], quant="4bit")
    v_data = np.load(vector_bank)
    
    layer = 24
    axis = "extraversion"
    w = torch.tensor(v_data[f"{layer}|{axis}|w"], dtype=torch.float32).to("cuda")
    b = float(v_data[f"{layer}|{axis}|b"][0])
    tau = 25.0
    
    prompt = "I'm interested in"
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    
    alphas_recorded = []

    def adaptive_hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        # We only care about generation or the last token
        hs_f32 = hs.to(torch.float32)
        dot_product = (hs_f32 * w).sum(dim=-1)
        dist = dot_product + b
        alpha = torch.clamp((tau - dist), min=0.0)
        
        # Log mean alpha for this step
        alphas_recorded.append(alpha.mean().item())
        
        # Apply steering
        steered = hs_f32 + alpha.unsqueeze(-1) * w
        return (steered.to(hs.dtype), *out[1:]) if isinstance(out, tuple) else steered.to(hs.dtype)

    # Register hook
    stack, _, _ = get_layer_stack(model)
    handle = stack[layer].register_forward_hook(adaptive_hook)
    
    print(f"Generating with Adaptive Steering (Tau={tau})...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False, # Use greedy to be deterministic
        )
    
    handle.remove()
    
    print("\nCalculated Alphas (Token-by-Token):")
    for i, a in enumerate(alphas_recorded):
        print(f"  Token {i}: Alpha = {a:.4f}")
    
    if len(set([round(a, 4) for a in alphas_recorded])) > 1:
        print("\n[VERIFIED] Alpha values are VARYING. This is true Adaptive Steering.")
    else:
        print("\n[WARNING] Alpha values are CONSTANT. Something might be wrong.")

if __name__ == "__main__":
    verify_alpha_variance()
