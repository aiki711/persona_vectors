#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/prepare_vocab_projected_vectors.py
# Extract Vocab-Projected Steering Vectors using SVD on Target Vocab Subspace of Mistral-7B-v0.3
# Output: vectors/vocab_projected_vectors.npz
#

import torch
import numpy as np
import yaml
from pathlib import Path
from persona_vectors.live_axes import load_model_and_tokenizer, _infer_main_device

WORKSPACE = Path("/home/s2550009/persona_vectors")
RAW_VEC_PATH = WORKSPACE / "vectors/mean_diff_vectors.npz"
OUT_VEC_PATH = WORKSPACE / "vectors/vocab_projected_vectors.npz"

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = list(range(32))
K_COMPONENTS = 32  # Top-k singular vectors for Vocab Subspace U_vocab

def main():
    print("=======================================================")
    print("Preparing Vocab-Projected Steering Vectors (Mistral-7B-v0.3)")
    print(f"Input Vector Bank: {RAW_VEC_PATH}")
    print(f"Output Vector Bank: {OUT_VEC_PATH}")
    print(f"Subspace Rank k: {K_COMPONENTS}")
    print("=======================================================")

    with open(WORKSPACE / "configs/mistral_7b.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    # Extract Unembedding Matrix W_unembed [V, d]
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W_unembed = model.lm_head.weight.detach().float().to(device)
    else:
        W_unembed = model.get_output_embeddings().weight.detach().float().to(device)

    print(f"Unembedding Matrix Shape: {W_unembed.shape}")

    # Load Raw Vectors
    raw_data = np.load(RAW_VEC_PATH)
    out_dict = {}

    # Preserve all non-vector metadata keys
    for k in raw_data.files:
        out_dict[k] = raw_data[k]

    for trait in TRAITS:
        print(f"\nProcessing Trait: {trait}...")
        
        # Determine target vocabulary indices for this trait based on mean_diff in logit space
        # Calculate W_unembed @ w_L for sample layers to find high-activation tokens for trait
        sample_w_keys = [f"{L}|{trait}|w" for L in [12, 16, 20] if f"{L}|{trait}|w" in raw_data]
        if sample_w_keys:
            sample_w = torch.tensor(np.mean([raw_data[k] for k in sample_w_keys], axis=0), dtype=torch.float32, device=device)
            logit_proj = torch.matmul(W_unembed, sample_w) # [V]
            top_vocab_indices = torch.topk(logit_proj, k=200).indices # Top 200 trait-aligned tokens
        else:
            top_vocab_indices = torch.arange(200, device=device)

        W_target = W_unembed[top_vocab_indices, :] # [m, d]
        print(f"  Target Vocab Subspace Matrix W_target Shape: {W_target.shape}")

        # Perform SVD: W_target = U * S * V^T
        # V is [d, d] -> Right singular vectors (columns of V)
        # Note: torch.linalg.svd returns V^T (Vh), so V = Vh.T
        _, _, Vh = torch.linalg.svd(W_target, full_matrices=False)
        U_vocab = Vh.T[:, :K_COMPONENTS] # [d, k]
        print(f"  U_vocab Subspace Basis Shape: {U_vocab.shape}")

        for L in LAYERS:
            w_key = f"{L}|{trait}|w"
            if w_key in raw_data:
                w_raw = torch.tensor(raw_data[w_key], dtype=torch.float32, device=device) # [d]
                orig_norm = torch.norm(w_raw, p=2).item()

                if orig_norm > 1e-8:
                    # Step 2 & 3: Orthogonal Projection w_tilde = U_vocab @ (U_vocab^T @ w_raw) in O(dk)
                    coeff = torch.matmul(U_vocab.T, w_raw) # [k]
                    w_proj = torch.matmul(U_vocab, coeff) # [d]
                    
                    proj_norm = torch.norm(w_proj, p=2).item()
                    if proj_norm > 1e-8:
                        # Re-normalization to original L2 norm
                        w_refined = (w_proj / proj_norm) * orig_norm
                    else:
                        w_refined = w_raw
                else:
                    w_refined = w_raw

                out_dict[w_key] = w_refined.cpu().numpy()

    np.savez_compressed(OUT_VEC_PATH, **out_dict)
    print(f"\n=======================================================")
    print(f"Successfully created: {OUT_VEC_PATH}")
    print("=======================================================")

if __name__ == "__main__":
    main()
