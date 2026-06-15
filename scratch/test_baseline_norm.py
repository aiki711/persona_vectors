import json
import torch
import numpy as np
import yaml
from pathlib import Path
from persona_vectors.live_axes import load_model_and_tokenizer, _infer_main_device, get_layer_stack, _format_prompt

# Hardcoded prompt 0 from inputs/eval_prompts_10.jsonl
prompt = "Aislin, from the moment I saw you, I knew you were special. Your kindness and beauty shine bright. I sense you'd be a loyal and protective mate. Would you like to get to know me better? I'd love to take you on a hunt and show you the best of our lands."

def format_and_tokenize(tokenizer, prompt, device):
    formatted = _format_prompt(tokenizer, prompt)
    return tokenizer(formatted, return_tensors="pt").to(device)

def get_base_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
    return out.logits[0, -1, :].float()

def test_norm_mode(model, tokenizer, device, vector_bank_path, norm_mode):
    print(f"\n--- Testing Vector Bank: {vector_bank_path} | Norm Mode: {norm_mode} ---")
    
    # Load vectors
    v_data = np.load(vector_bank_path)
    layer_w = {}
    for L in range(5): # Check first 5 layers
        w_key = f"{L}|extraversion|w"
        raw_norm_key = f"{L}|extraversion|raw_norm"
        mp_key = f"{L}|extraversion|midpoint"
        if w_key in v_data:
            w_vec = torch.tensor(v_data[w_key], dtype=torch.float32)
            if norm_mode in ["midpoint", "raw_norm"]:
                if raw_norm_key in v_data:
                    r_norm = float(v_data[raw_norm_key][0])
                    w_norm = torch.norm(w_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * r_norm
                    print(f"  Layer {L}: Scaled by raw_norm = {r_norm:.4f}")
                elif mp_key in v_data:
                    m_vec = torch.tensor(v_data[mp_key], dtype=torch.float32)
                    w_norm = torch.norm(w_vec).item()
                    m_norm = torch.norm(m_vec).item()
                    w_vec = (w_vec / (w_norm + 1e-10)) * m_norm
                    print(f"  Layer {L}: Scaled by midpoint norm = {m_norm:.4f}")
                else:
                    print(f"  Layer {L}: No raw_norm or midpoint key! Norm = {torch.norm(w_vec).item():.4f}")
            else:
                print(f"  Layer {L}: Norm Mode 'none'. Norm = {torch.norm(w_vec).item():.4f}")
            layer_w[L] = w_vec.to(device)

    inputs = format_and_tokenize(tokenizer, prompt, device)
    base_logits = get_base_logits(model, inputs.input_ids)
    
    stack, _, _ = get_layer_stack(model)
    alpha = 1.0
    
    raw_scores = {}
    for L, w_dev in layer_w.items():
        def hook(mod, inp, out_val):
            hs = out_val[0] if isinstance(out_val, tuple) else out_val
            if not torch.isfinite(hs).all(): return out_val
            hs_f32 = hs.to(torch.float32)
            steered = hs_f32 + alpha * w_dev.view(1, 1, -1)
            return (steered.to(hs.dtype), *out_val[1:]) if isinstance(out_val, tuple) else steered.to(hs.dtype)

        handle = stack[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out_steered = model(inputs.input_ids)
            steered_logits = out_steered.logits[0, -1, :].float()
            raw_scores[L] = (steered_logits - base_logits).norm().item()
        finally:
            handle.remove()
            
    print("Logit-Diff Raw Scores (first 5 layers):", {str(L): v for L, v in raw_scores.items()})

def main():
    with open("config/mistral_7b.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()
    
    # We test combinations to see which one matches the archived:
    # {'0': 7.588772773742676, '1': 7.159839153289795, '2': 9.47221565246582, '3': 11.689706802368164, '4': 16.45771026611328}
    test_norm_mode(model, tokenizer, device, "vectors/mean_diff_vectors.npz", "raw_norm")
    test_norm_mode(model, tokenizer, device, "vectors/mean_diff_vectors.npz", "midpoint")
    test_norm_mode(model, tokenizer, device, "exp_steering_layer_analysis/vectors/mean_diff_vectors.npz", "raw_norm")
    test_norm_mode(model, tokenizer, device, "exp_steering_layer_analysis/vectors/mean_diff_vectors.npz", "midpoint")

if __name__ == "__main__":
    main()
