#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_entropy_and_gating.py
#

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import shutil
import yaml
from persona_vectors.live_axes import load_model_and_tokenizer, _infer_main_device

def main():
    plt.close("all")
    
    # Part 1: Plot the Gating Function curves
    h = np.linspace(0, 8, 400)
    ic = np.linspace(0, 12, 400)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: Entropy Gating curves (theta = 1.0, 1.5, 2.0, k = 8.0)
    axes[0].plot(h, 1.0 / (1.0 + np.exp(-8.0 * (h - 1.0))), color="#e74c3c", linewidth=2.5, linestyle=":", label=r"$\theta_H = 1.0$ (Very Loose)")
    axes[0].plot(h, 1.0 / (1.0 + np.exp(-8.0 * (h - 1.5))), color="#8e44ad", linewidth=3.5, label=r"$\theta_H = 1.5$ (Optimal Syntax Cutoff)")
    axes[0].plot(h, 1.0 / (1.0 + np.exp(-8.0 * (h - 2.0))), color="#3498db", linewidth=2.5, linestyle="--", label=r"$\theta_H = 2.0$ (Conservative)")
    
    axes[0].axvspan(0, 1.5, color="#e74c3c", alpha=0.05)
    axes[0].text(0.75, 0.4, "Syntax Protection\n(Steering OFF)", color="#c0392b", fontsize=10, fontweight="bold", ha="center")
    axes[0].axvspan(1.5, 8, color="#2ecc71", alpha=0.05)
    axes[0].text(4.75, 0.5, "Semantic Decision Points\n(Steering ON at 100%)", color="#27ae60", fontsize=11, fontweight="bold", ha="center")
    
    axes[0].set_xlabel("Predictive Entropy H [bits]", fontsize=12, fontweight="bold", labelpad=8)
    axes[0].set_ylabel("Steering Intensity Multiplier", fontsize=12, fontweight="bold", labelpad=8)
    axes[0].set_title("Predictive Entropy Gate (Zero-Delay)", fontsize=13, fontweight="bold", pad=12)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].set_xlim(0, 8)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="lower right")
    
    # Right subplot: Dual Gating 2D visualization (H on X-axis, IC on Y-axis)
    H_grid, IC_grid = np.meshgrid(np.linspace(0, 8, 200), np.linspace(0, 12, 200))
    # Combined multiplier
    f_syntax = 1.0 / (1.0 + np.exp(-8.0 * (H_grid - 1.5)))
    f_rare = 1.0 / (1.0 + np.exp(8.0 * (IC_grid - 5.5)))
    Z = f_syntax * f_rare
    
    cp = axes[1].contourf(H_grid, IC_grid, Z, levels=50, cmap="Purples")
    cbar = fig.colorbar(cp, ax=axes[1])
    cbar.set_label("Gating Factor Multiplier (Beta)", fontsize=10, fontweight="bold")
    
    # Add boundaries
    axes[1].axvline(x=1.5, color="#e74c3c", linestyle="--", linewidth=2)
    axes[1].axhline(y=5.5, color="#e67e22", linestyle="--", linewidth=2)
    
    axes[1].text(0.75, 6.0, "Blocked\n(Syntax)", color="#c0392b", fontsize=9, fontweight="bold", ha="center")
    axes[1].text(4.0, 9.0, "Blocked\n(Rare/Fact)", color="#d35400", fontsize=9, fontweight="bold", ha="center")
    axes[1].text(4.0, 3.0, "Steering Active\n(100% Strength)", color="#27ae60", fontsize=10, fontweight="bold", ha="center")
    
    axes[1].set_xlabel("Predictive Entropy H [bits]", fontsize=12, fontweight="bold", labelpad=8)
    axes[1].set_ylabel("Token Surprisal (IC) [bits]", fontsize=12, fontweight="bold", labelpad=8)
    axes[1].set_title("Dual Gating Active Steering Region", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlim(0, 8)
    axes[1].set_ylim(0, 12)
    
    plt.suptitle("Predictive Entropy Gate Curve & Dual Gating Active Region", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_dir = Path("exp_token_intensity/exp_entropy_gating")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entropy_gating_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved curves plot to: {out_path}")
    
    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    shutil.copy(out_path, artifact_dir / "entropy_gating_curves.png")
    
    # Part 2: Extract empirical entropy from sample generations
    print("Loading model to extract empirical token entropy...")
    with open("configs/mistral_7b.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    model, tokenizer = load_model_and_tokenizer(cfg.get("model_name"), quant=cfg.get("quant", "auto"))
    device = _infer_main_device(model)
    model.eval()
    
    # Reconstruct entropy for 5 sample texts from our evaluations
    # Let's read from exp_token_intensity/exp_entropy_gating/extraversion/scores_masked_proj_rank_theta_2.0_7.0_k_1.5_2.0_entropy_Val5.0.csv (or similar)
    # Let's search for generated JSONL files in the folder
    jsonl_files = list((out_dir / "extraversion").glob("*.jsonl"))
    if not jsonl_files:
        print("No generated jsonl files found to extract empirical entropy.")
        return
        
    sample_file = jsonl_files[0]
    print(f"Reading sample generation from: {sample_file}")
    
    samples = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= 3:
                    break
                    
    # Recompute step-wise entropy for these sequences
    all_entropy_vals = []
    token_labels = [] # list of (token_str, entropy)
    
    with torch.no_grad():
        for s in samples:
            prompt = s["prompt"]
            dyn_text = s["dyn_text"]
            
            # Format prompt and encode
            full_text = prompt + " " + dyn_text
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = inputs.input_ids[0]
            
            prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
            
            # Forward pass over the whole sequence to get logits
            outputs = model(input_ids.unsqueeze(0), use_cache=False)
            logits = outputs.logits[0] # seq_len, vocab_size
            
            # For each token generated (from prompt_len to end-1)
            for t_idx in range(prompt_len - 1, len(input_ids) - 1):
                t_logits = logits[t_idx]
                probs = F.softmax(t_logits / 0.7, dim=-1)
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1).item()
                
                next_token_id = input_ids[t_idx + 1].item()
                token_str = tokenizer.decode([next_token_id])
                
                all_entropy_vals.append(entropy)
                token_labels.append((token_str, entropy))
                
    # Create the empirical histogram
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_entropy_vals, bins=35, color="#8e44ad", alpha=0.7, edgecolor="#7d3c98", label="Empirical Token Distribution")
    ax.axvline(x=1.5, color="#e74c3c", linestyle="--", linewidth=2.5, label=r"Optimal Boundary ($\theta_H = 1.5$)")
    
    # Label specific tokens to show where they fall
    # Sort token labels to find representative tokens in each zone
    token_labels_sorted = sorted(token_labels, key=lambda x: x[1])
    
    # Annotate low entropy tokens
    low_tokens = [tok for tok, ent in token_labels_sorted if ent < 1.3]
    unique_low = []
    for tok in low_tokens:
        tok_clean = tok.replace("\n", "\\n").replace(" ", " ")
        if tok_clean.strip() == "":
            tok_clean = "space"
        if tok_clean not in unique_low and len(unique_low) < 5:
            unique_low.append(tok_clean)
            
    ax.text(0.75, ax.get_ylim()[1] * 0.8, "Grammar Zone\nPredictable tokens:\n" + "\n".join([f"• '{t}'" for t in unique_low]),
            color="#c0392b", fontsize=9, fontweight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.9, edgecolor='#c0392b'))
            
    # Annotate high entropy tokens
    high_tokens = [tok for tok, ent in token_labels_sorted if ent > 2.5]
    unique_high = []
    for tok in high_tokens:
        tok_clean = tok.replace("\n", "\\n").replace(" ", " ")
        if tok_clean.strip() != "" and tok_clean not in unique_high and len(unique_high) < 5:
            unique_high.append(tok_clean)
            
    ax.text(4.5, ax.get_ylim()[1] * 0.8, "Semantic Choice Zone\nAdjectives / Nouns / Verbs:\n" + "\n".join([f"• '{t}'" for t in unique_high]),
            color="#27ae60", fontsize=9, fontweight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.9, edgecolor='#27ae60'))
            
    ax.set_xlabel("Predictive Entropy H [bits]", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Token Count", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Empirical Predictive Entropy Distribution during Generation (Mistral-7B)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    hist_path = out_dir / "empirical_entropy_distribution.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    print(f"Saved empirical distribution plot to: {hist_path}")
    
    # Copy to artifacts
    shutil.copy(hist_path, artifact_dir / "empirical_entropy_distribution.png")
    print("Copied empirical distribution plot to artifacts.")

if __name__ == "__main__":
    main()
