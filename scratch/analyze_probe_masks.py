#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/analyze_probe_masks.py
#
# Analyzes the dimension overlap of probe-trained personality masks:
# 1. Layer-to-layer overlap (Jaccard similarity) for each trait.
# 2. Trait-to-trait overlap at each layer.
# Saves plots to exp_steering_dyn_layer_pdf/figures/
#

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = list(range(32))

def jaccard_similarity(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / float(union) if union > 0 else 0.0

def main():
    mask_bank_path = Path("vectors/probe_masks.npz")
    if not mask_bank_path.exists():
        print(f"Error: mask bank not found at {mask_bank_path}. Please run train_probe_filters.py first.")
        return

    m_data = np.load(mask_bank_path)
    
    # Create figures directory
    fig_dir = Path("exp_steering_dyn_layer_pdf/figures/analysis")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/eb5ffadd-d5e7-40a3-a0b3-5e88bfefda49/images")
    
    print("=== Analyzing Probe Masks ===")
    
    # 1. Layer-to-Layer Jaccard Similarity Heatmap for each trait
    for trait in TRAITS:
        overlap_matrix = np.zeros((32, 32))
        for l1 in LAYERS:
            mask1 = m_data.get(f"{l1}|{trait}|mask")
            if mask1 is None: continue
            for l2 in LAYERS:
                mask2 = m_data.get(f"{l2}|{trait}|mask")
                if mask2 is None: continue
                overlap_matrix[l1, l2] = jaccard_similarity(mask1, mask2)
                
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, cmap="Blues", vmin=0.0, vmax=0.5, 
                    xticklabels=5, yticklabels=5, cbar_kws={'label': 'Jaccard Similarity'})
        plt.title(f"Layer-to-Layer Mask Dimension Overlap: {trait.capitalize()}", fontsize=12, fontweight="bold")
        plt.xlabel("Layer Index")
        plt.ylabel("Layer Index")
        
        out_path = fig_dir / f"mask_overlap_layers_{trait}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved layer overlap heatmap for {trait} to: {out_path}")
        
        if artifact_dir.exists():
            shutil_dest = artifact_dir / f"mask_overlap_layers_{trait}.png"
            import shutil
            shutil.copy(out_path, shutil_dest)
            
    # 2. Cross-Trait Dimension Overlap at each Layer
    overlap_records = []
    for L in LAYERS:
        row = {"layer": L}
        for i in range(len(TRAITS)):
            t1 = TRAITS[i]
            mask1 = m_data.get(f"{L}|{t1}|mask")
            if mask1 is None: continue
            for j in range(i+1, len(TRAITS)):
                t2 = TRAITS[j]
                mask2 = m_data.get(f"{L}|{t2}|mask")
                if mask2 is None: continue
                row[f"{t1}_vs_{t2}"] = jaccard_similarity(mask1, mask2)
        overlap_records.append(row)
        
    df_overlap = pd.DataFrame(overlap_records)
    
    plt.figure(figsize=(12, 6))
    cols_to_plot = [c for c in df_overlap.columns if c != "layer"]
    for col in cols_to_plot:
        plt.plot(df_overlap["layer"], df_overlap[col], label=col.replace("_vs_", " vs ").title(), marker='o', alpha=0.7)
        
    plt.title("Cross-Trait Mask Dimension Overlap Across Layers", fontsize=12, fontweight="bold")
    plt.xlabel("Layer Index")
    plt.ylabel("Jaccard Similarity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path = fig_dir / "mask_overlap_cross_traits.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved cross-trait overlap plot to: {out_path}")
    
    if artifact_dir.exists():
        shutil_dest = artifact_dir / "mask_overlap_cross_traits.png"
        import shutil
        shutil.copy(out_path, shutil_dest)

    print("\n[DONE] Probe mask analysis completed successfully.")

if __name__ == "__main__":
    main()
