#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/verify_vocab_projected_vectors.py
# Verification and Comparative Analysis of Raw vs Vocab-Projected Vectors
#

import numpy as np
import torch
import yaml
from pathlib import Path
from persona_vectors.live_axes import load_model_and_tokenizer, _infer_main_device

WORKSPACE = Path("/home/s2550009/persona_vectors")
RAW_VEC_PATH = WORKSPACE / "vectors/mean_diff_vectors.npz"
PROJ_VEC_PATH = WORKSPACE / "vectors/vocab_projected_vectors.npz"

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
SAMPLE_LAYERS = [8, 12, 16, 20, 24]

def main():
    print("=======================================================")
    print("Verifying Raw vs Vocab-Projected Steering Vectors")
    print("=======================================================")

    raw_data = np.load(RAW_VEC_PATH)
    proj_data = np.load(PROJ_VEC_PATH)

    with open(WORKSPACE / "configs/mistral_7b.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W_unembed = model.lm_head.weight.detach().float().to(device)
    else:
        W_unembed = model.get_output_embeddings().weight.detach().float().to(device)

    print(f"\n{'Trait':18s} | {'Layer':5s} | {'Cosine Sim':10s} | {'Raw Vocab Norm':15s} | {'Proj Vocab Norm':16s} | {'Norm Ratio':10s}")
    print("-" * 85)

    for trait in TRAITS:
        for L in SAMPLE_LAYERS:
            w_key = f"{L}|{trait}|w"
            if w_key in raw_data and w_key in proj_data:
                w_raw = torch.tensor(raw_data[w_key], dtype=torch.float32, device=device)
                w_proj = torch.tensor(proj_data[w_key], dtype=torch.float32, device=device)

                # Cosine similarity between raw and projected vector
                sim = torch.dot(w_raw, w_proj) / (torch.norm(w_raw) * torch.norm(w_proj) + 1e-10)

                # Unembedding projection norm (reach to vocab space)
                raw_reach = torch.norm(torch.matmul(W_unembed, w_raw), p=2).item()
                proj_reach = torch.norm(torch.matmul(W_unembed, w_proj), p=2).item()

                ratio = proj_reach / (raw_reach + 1e-10)

                print(f"{trait.capitalize():18s} | Layer {L:2d} | {sim.item():10.4f} | {raw_reach:15.2f} | {proj_reach:16.2f} | {ratio:10.2f}x")

    print("\n-------------------------------------------------------")
    print("Vector Verification Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
