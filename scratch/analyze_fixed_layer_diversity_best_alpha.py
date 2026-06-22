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

VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def calculate_repetition_rate(text: str, n: int) -> float:
    if not isinstance(text, str):
        return 0.0
    words = [w.strip(".,!?:;()\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    unique_ngrams = set(ngrams)
    return (len(ngrams) - len(unique_ngrams)) / len(ngrams)

def find_best_alpha(results_dir: Path, trait: str, method: str) -> float:
    """Finds the alpha that yields the maximum score under strict safety criteria."""
    best_score = -1.0
    best_alpha = 4.0 # default fallback
    trait_dir = results_dir / trait
    
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        jsonl_path = trait_dir / f"{method}_Val{float(val)}.jsonl"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
        if not jsonl_path.exists():
            jsonl_path = trait_dir / f"{method}_Val{val}.jsonl"
            
        if csv_path.exists() and jsonl_path.exists():
            try:
                # Load CSV
                df = pd.read_csv(csv_path)
                if "dyn_score" in df.columns:
                    df["dyn_score"] = df["dyn_score"].replace(0, np.nan)
                mean_score = df["dyn_score"].mean()
                mean_ppl = df["dyn_ppl"].mean()
                max_ppl = df["dyn_ppl"].max() if "dyn_ppl" in df.columns else np.nan
                
                # Load JSONL to calculate repetition
                dyn_texts = []
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if "dyn_text" in data:
                            dyn_texts.append(data["dyn_text"])
                            
                rep_3gram_list = [calculate_repetition_rate(txt, 3) for txt in dyn_texts]
                rep_4gram_list = [calculate_repetition_rate(txt, 4) for txt in dyn_texts]
                max_3gram = max(rep_3gram_list) if rep_3gram_list else 0.0
                max_4gram = max(rep_4gram_list) if rep_4gram_list else 0.0
                
                if "dyn_reason" in df.columns:
                    coherence_rate = df["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean()
                else:
                    coherence_rate = 1.0
                    
                # Strict Safety criteria
                is_safe = (
                    mean_ppl <= 25.0 and
                    coherence_rate >= 0.8 and
                    max_ppl <= 35.0
                )
                
                if is_safe:
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = val
            except Exception:
                pass
    return best_alpha

def load_data(results_dir: Path):
    records = []
    alpha_mapping = {}
    
    for trait in TRAITS:
        alpha_mapping[trait] = {}
        trait_dir = results_dir / trait
        if not trait_dir.exists():
            continue
            
        for method in METHODS:
            best_alpha = find_best_alpha(results_dir, trait, method)
            alpha_mapping[trait][method] = best_alpha
            
            # Load the jsonl for this best alpha
            jsonl_path = trait_dir / f"{method}_Val{float(best_alpha)}.jsonl"
            if not jsonl_path.exists():
                jsonl_path = trait_dir / f"{method}_Val{int(best_alpha)}.jsonl"
                
            if jsonl_path.exists():
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            if "dyn_layer" in data:
                                records.append({
                                    "trait": TRAIT_LABELS[trait],
                                    "method": METHOD_LABELS[method],
                                    "best_alpha": best_alpha,
                                    "idx": data.get("idx", 0),
                                    "dyn_layer": data["dyn_layer"]
                                })
                except Exception as e:
                    print(f"Warning: Failed to load {jsonl_path}: {e}")
                    
    return pd.DataFrame(records), alpha_mapping

def main():
    results_dir = Path("exp_steering_dyn_layer_raw/results")
    df, alpha_map = load_data(results_dir)
    
    if df.empty:
        print("Error: No data loaded.")
        return
        
    print(f"Loaded {len(df)} records using best alphas.")
    
    # Print the alpha mapping for reference
    print("\n--- Selected Best Alpha per Trait and Method ---")
    alpha_table = []
    for method in METHODS:
        row = {"Method": METHOD_LABELS[method]}
        for trait in TRAITS:
            row[TRAIT_LABELS[trait]] = alpha_map[trait].get(method, np.nan)
        alpha_table.append(row)
    df_alphas = pd.DataFrame(alpha_table)
    print(df_alphas.to_markdown(index=False))
    df_alphas.to_csv("scratch/fixed_layer_diversity_best_alphas.csv", index=False)
    
    # Calculate statistics per method and trait
    stats = []
    for (trait, method), group in df.groupby(["trait", "method"]):
        layers = group["dyn_layer"].values
        best_alpha = group["best_alpha"].iloc[0]
        stats.append({
            "Trait": trait,
            "Method": method,
            "Best Alpha": best_alpha,
            "Mean Layer": np.mean(layers),
            "Std Dev": np.std(layers),
            "Min Layer": np.min(layers),
            "Max Layer": np.max(layers),
            "Unique Layers": len(np.unique(layers)),
            "Layers Selected": ", ".join(map(str, sorted(list(np.unique(layers)))))
        })
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv("scratch/fixed_layer_diversity_best_alpha_stats.csv", index=False)
    
    # Calculate summary across all traits
    summary = []
    for method, group in df.groupby("method"):
        layers = group["dyn_layer"].values
        trait_stds = []
        trait_uniques = []
        for trait, t_group in group.groupby("trait"):
            t_layers = t_group["dyn_layer"].values
            trait_stds.append(np.std(t_layers))
            trait_uniques.append(len(np.unique(t_layers)))
            
        summary.append({
            "Method": method,
            "Avg Std Dev (Within-Trait)": np.mean(trait_stds),
            "Avg Unique Layers (Per Trait)": np.mean(trait_uniques),
            "Global Unique Layers": len(np.unique(layers)),
            "Global Min": np.min(layers),
            "Global Max": np.max(layers),
        })
    df_summary = pd.DataFrame(summary).sort_values("Avg Std Dev (Within-Trait)", ascending=False)
    df_summary.to_csv("scratch/fixed_layer_diversity_best_alpha_summary.csv", index=False)
    
    print("\n--- Summary Table (Best Alpha, Sorted by Avg Std Dev within Trait) ---")
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
        
        # Add labels to the points indicating the best alpha
        # Get x coordinates of the categories
        for j, m_name in enumerate(method_order):
            sub_df = df_trait[df_trait["method"] == m_name]
            if not sub_df.empty:
                alpha_val = sub_df["best_alpha"].iloc[0]
                ax.text(j, 29.0, f"a={alpha_val}", ha="center", va="bottom", fontsize=7.5, color="darkred", fontweight="bold")
        
        ax.set_title(f"{trait} Layer Selection Diversity (Optimal Alpha)", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Selected Layer", fontsize=10)
        ax.set_ylim(3.5, 30.5)
        ax.set_yticks(range(4, 31, 2))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        
    # The 6th plot is the global combined view
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
    
    plt.suptitle("Fixed-Layer Steering: Layer Selection Diversity at Max Safe Score Alpha (Strict Safety)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save figures
    out_dir = Path("exp_steering_dyn_layer_raw/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fixed_layer_diversity_best_alpha.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved diversity plot to: {out_path}")
    
    # Copy to artifact folder for user viewing
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/316d92fc-a09f-45ab-a84d-a1a4060ccdb9/images")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "fixed_layer_diversity_best_alpha.png"
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()
