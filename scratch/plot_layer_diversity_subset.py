#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/plot_layer_diversity_subset.py
#

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_LABELS = {
    "extraversion": "Extraversion",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness": "Agreeableness"
}

# Only these two methods
METHODS = [
    "logit_diff",
    "masked_proj_rank_only"
]

METHOD_LABELS = {
    "logit_diff": "Logit Diff",
    "masked_proj_rank_only": "PDF Proj Rank"
}

def load_data(results_dir: Path, alpha=5.0):
    records = []
    for trait in TRAITS:
        trait_dir = results_dir / trait
        if not trait_dir.exists():
            continue
        for method in METHODS:
            jsonl_path = trait_dir / f"{method}_Val{float(alpha)}.jsonl"
            if not jsonl_path.exists():
                jsonl_path = trait_dir / f"{method}_Val{int(alpha)}.jsonl"
                
            if jsonl_path.exists():
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            if "dyn_layer" in data:
                                records.append({
                                    "trait": TRAIT_LABELS[trait],
                                    "method": METHOD_LABELS[method],
                                    "idx": data.get("idx", 0),
                                    "dyn_layer": data["dyn_layer"]
                                })
                except Exception as e:
                    print(f"Warning: Failed to load {jsonl_path}: {e}")
    return pd.DataFrame(records)

def main():
    results_dir = Path("exp_layer_selection/exp_steering_dyn_layer_raw/results")
    df = load_data(results_dir, alpha=5.0)
    
    if df.empty:
        print("Error: No data loaded. Check files.")
        return
        
    print(f"Loaded {len(df)} records for the subset.")

    # Plotting
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    method_order = [METHOD_LABELS[m] for m in METHODS]
    
    # Boxplot in light gray
    sns.boxplot(
        data=df, x="method", y="dyn_layer", order=method_order,
        ax=ax, color="#f8f9fa", width=0.4, fliersize=0, boxprops={"zorder": 1, "edgecolor": "gray"}
    )
    # Strip plot to show individual layers for each trait
    sns.stripplot(
        data=df, x="method", y="dyn_layer", order=method_order, hue="trait",
        ax=ax, size=6, jitter=0.2, palette="tab10", alpha=0.8, zorder=2
    )
    
    ax.set_title("Layer Selection Diversity Comparison (Alpha=5.0)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Steering Method", fontsize=11, labelpad=8)
    ax.set_ylabel("Selected Layer", fontsize=11)
    ax.set_ylim(3.5, 30.5)
    ax.set_yticks(range(4, 31, 2))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Trait", loc="upper right", bbox_to_anchor=(1.25, 1.0))
    
    plt.tight_layout()
    
    # Save to exp_layer_selection
    out_dir = Path("exp_layer_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fixed_layer_diversity_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved subset boxplot to: {out_path}")
    
    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, artifact_dir / "fixed_layer_diversity_comparison.png")
    print("Copied plot to artifacts.")

if __name__ == "__main__":
    main()
