#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_LABELS = {
    "extraversion": "Extraversion",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness": "Agreeableness"
}

METHODS = [
    "logit_diff",
    "cos_only",
    "rank_only",
    "proj_cos_only",
    "proj_rank_only",
    "masked_cos_only",
    "masked_rank_only",
    "masked_proj_cos_only",
    "masked_proj_rank_only"
]

METHOD_LABELS = {
    "logit_diff": "Logit Diff",
    "cos_only": "Cos-Only",
    "rank_only": "Rank-Only",
    "proj_cos_only": "Proj Cos-Only",
    "proj_rank_only": "Proj Rank-Only",
    "masked_cos_only": "PDF Cos-Only",
    "masked_rank_only": "PDF Rank-Only",
    "masked_proj_cos_only": "PDF Proj Cos",
    "masked_proj_rank_only": "PDF Proj Rank"
}

def load_data(results_dir: Path, alpha=4.0):
    records = []
    
    for trait in TRAITS:
        trait_dir = results_dir / trait
        if not trait_dir.exists():
            continue
            
        for method in METHODS:
            # Try both Val4.0 and Val4
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
    results_dir = Path("exp_steering_dyn_layer_raw/results")
    df = load_data(results_dir, alpha=4.0)
    
    if df.empty:
        print("Error: No data loaded. Check if the directory 'exp_steering_dyn_layer_raw/results' exists and contains jsonl files.")
        return
        
    print(f"Loaded {len(df)} records.")
    
    # Calculate statistics per method and trait
    stats = []
    for (trait, method), group in df.groupby(["trait", "method"]):
        layers = group["dyn_layer"].values
        stats.append({
            "Trait": trait,
            "Method": method,
            "Mean Layer": np.mean(layers),
            "Std Dev": np.std(layers),
            "Min Layer": np.min(layers),
            "Max Layer": np.max(layers),
            "Unique Layers": len(np.unique(layers)),
            "Layers Selected": ", ".join(map(str, sorted(list(np.unique(layers)))))
        })
        
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv("scratch/fixed_layer_diversity_stats.csv", index=False)
    print("Saved statistics to scratch/fixed_layer_diversity_stats.csv")
    
    # Also calculate summary across all traits
    summary = []
    for method, group in df.groupby("method"):
        layers = group["dyn_layer"].values
        # For average std, we average the std dev within each trait to avoid mixing trait-specific offsets
        trait_stds = []
        trait_uniques = []
        for trait, t_group in group.groupby("trait"):
            t_layers = t_group["dyn_layer"].values
            trait_stds.append(np.std(t_layers))
            trait_uniques.append(len(np.unique(t_layers)))
            
        summary.append({
            "Method": method,
            "Global Mean": np.mean(layers),
            "Avg Std Dev (Within-Trait)": np.mean(trait_stds),
            "Avg Unique Layers (Per Trait)": np.mean(trait_uniques),
            "Global Unique Layers": len(np.unique(layers)),
            "Global Min": np.min(layers),
            "Global Max": np.max(layers),
        })
    df_summary = pd.DataFrame(summary).sort_values("Avg Std Dev (Within-Trait)", ascending=False)
    df_summary.to_csv("scratch/fixed_layer_diversity_summary.csv", index=False)
    
    print("\n--- Summary Table (Sorted by Avg Std Dev within Trait) ---")
    print(df_summary.to_markdown(index=False))
    
    # Plotting
    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    method_order = [METHOD_LABELS[m] for m in METHODS]
    
    # Subplots for each trait
    for i, trait in enumerate(TRAIT_LABELS.values()):
        ax = axes[i]
        df_trait = df[df["trait"] == trait]
        
        if df_trait.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.set_title(trait)
            continue
            
        sns.boxplot(
            data=df_trait, x="method", y="dyn_layer", order=method_order,
            ax=ax, color="#f8f9fa", width=0.5, fliersize=0, boxprops={"zorder": 1, "edgecolor": "gray"}
        )
        sns.stripplot(
            data=df_trait, x="method", y="dyn_layer", order=method_order,
            ax=ax, size=6, jitter=0.25, palette="Set2", alpha=0.9, zorder=2, hue="method", legend=False
        )
        
        ax.set_title(f"{trait} Layer Selection Diversity (Alpha=4.0)", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Selected Layer", fontsize=10)
        ax.set_ylim(3.5, 30.5)
        ax.set_yticks(range(4, 31, 2))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        
    # The 6th plot is the global combined view (offset-corrected or raw)
    ax_all = axes[5]
    sns.boxplot(
        data=df, x="method", y="dyn_layer", order=method_order,
        ax=ax_all, color="#f8f9fa", width=0.5, fliersize=0, boxprops={"zorder": 1, "edgecolor": "gray"}
    )
    sns.stripplot(
        data=df, x="method", y="dyn_layer", order=method_order, hue="trait",
        ax=ax_all, size=5, jitter=0.25, palette="tab10", alpha=0.7, zorder=2
    )
    ax_all.set_title("Combined Layer Selection Diversity (All Traits)", fontsize=12, fontweight="bold")
    ax_all.set_xlabel("")
    ax_all.set_ylabel("Selected Layer", fontsize=10)
    ax_all.set_ylim(3.5, 30.5)
    ax_all.set_yticks(range(4, 31, 2))
    ax_all.set_xticklabels(ax_all.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax_all.grid(axis="y", linestyle=":", alpha=0.5)
    ax_all.legend(title="Trait", loc="lower right", fontsize=8)
    
    plt.suptitle("Fixed-Layer Steering: Layer Selection Diversity across Prompts (Alpha=4.0)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save figures
    out_dir = Path("exp_steering_dyn_layer_raw/figures/layey_selection_diversity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fixed_layer_diversity.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved diversity plot to: {out_path}")
    
    # Copy to artifact folder for user viewing
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "fixed_layer_diversity.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()
