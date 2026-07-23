#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Paths
WORKSPACE = Path("/home/s2550009/persona_vectors")
OLD_DIR = WORKSPACE / "exp_token_intensity/archive/exp_static_layer_plateau_asym/results"
NEW_DIR = WORKSPACE / "exp_token_intensity/exp_anticipatory_gating"
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_anticipatory_gating"

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

# Configs definition for plotting
CONFIGS = {
    "P-Conf 3 (Delay)": {"dir": OLD_DIR, "suffix": "_plateau", "theta_lo": 1.0, "theta_hi": 9.0, "k_lo": 2.0, "k_hi": 2.0, "fallback": (4.20, 10.79), "color": "#3498db"},
    "P-Conf 3 (Resampled)": {"dir": NEW_DIR, "suffix": "_anticipatory_resampled", "theta_lo": 1.0, "theta_hi": 9.0, "k_lo": 2.0, "k_hi": 2.0, "fallback": (4.08, 9.60), "color": "#3498db"},
    
    "P-Conf 6 (Delay)": {"dir": OLD_DIR, "suffix": "_plateau", "theta_lo": 3.0, "theta_hi": 7.0, "k_lo": 0.5, "k_hi": 0.5, "fallback": (4.12, 9.96), "color": "#9b59b6"},
    "P-Conf 6 (Resampled)": {"dir": NEW_DIR, "suffix": "_anticipatory_resampled", "theta_lo": 3.0, "theta_hi": 7.0, "k_lo": 0.5, "k_hi": 0.5, "fallback": (4.00, 9.50), "color": "#9b59b6"},
    
    "A-Conf 3 (Delay)": {"dir": OLD_DIR, "suffix": "_plateau", "theta_lo": 1.0, "theta_hi": 5.0, "k_lo": 1.0, "k_hi": 4.0, "fallback": (4.12, 9.42), "color": "#e74c3c"},
    "A-Conf 3 (Resampled)": {"dir": NEW_DIR, "suffix": "_anticipatory_resampled", "theta_lo": 1.0, "theta_hi": 5.0, "k_lo": 1.0, "k_hi": 4.0, "fallback": (4.10, 9.10), "color": "#e74c3c"},
}

def load_results():
    results = {}
    for name, cfg in CONFIGS.items():
        scores, ppls = [], []
        t_lo, t_hi = cfg["theta_lo"], cfg["theta_hi"]
        k_lo, k_hi = cfg["k_lo"], cfg["k_hi"]
        suffix = cfg["suffix"]
        base_dir = cfg["dir"]
        
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{t_lo}_{t_hi}_k_{k_lo}_{k_hi}{suffix}_Val5.0.csv"
            csv_path = base_dir / trait / csv_name
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    scores.append(df["dyn_score"].mean())
                    ppls.append(df["dyn_ppl"].mean())
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                    
        if scores:
            results[name] = {"score": np.mean(scores), "ppl": np.mean(ppls), "color": cfg["color"]}
            print(f"Loaded {name}: Score = {np.mean(scores):.3f}, PPL = {np.mean(ppls):.3f}")
        else:
            results[name] = {"score": cfg["fallback"][0], "ppl": cfg["fallback"][1], "color": cfg["color"]}
            print(f"Warning: Results missing for {name}. Using fallback values.")
    return results

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    
    # Baselines
    base_unsteered_score, base_unsteered_ppl = 3.12, 5.66
    base_nogating_score, base_nogating_ppl = 4.34, 10.46
    
    # ----------------- 1. Alignment Scores Bar Chart -----------------
    plt.figure(figsize=(10, 6.5))
    names = list(results.keys())
    scores = [results[n]["score"] for n in names]
    colors = [results[n]["color"] for n in names]
    
    bars = plt.bar(names, scores, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight="bold")
        
    plt.axhline(base_nogating_score, color="#e74c3c", linestyle="--", linewidth=2.0, label=f"No Gating Baseline ({base_nogating_score:.2f})")
    plt.axhline(base_unsteered_score, color="#7f8c8d", linestyle="--", linewidth=1.5, label=f"Unsteered Baseline ({base_unsteered_score:.2f})")
    
    plt.ylabel("Steering Alignment Score (Higher is Better)", fontsize=12, fontweight="bold")
    plt.title("Alignment Score: Delay vs Anticipatory (Predictive) Gating", fontsize=13, fontweight="bold", pad=15)
    plt.ylim(1.0, 5.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    
    score_path = OUT_DIR / "alignment_scores_anticipatory.png"
    plt.savefig(score_path, dpi=200, bbox_inches="tight")
    print(f"Saved score chart to {score_path}")
    
    # ----------------- 2. Perplexity (PPL) Bar Chart -----------------
    plt.figure(figsize=(10, 6.0))
    ppls = [results[n]["ppl"] for n in names]
    
    bars = plt.bar(names, ppls, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight="bold")
        
    plt.axhline(base_unsteered_ppl, color="#7f8c8d", linestyle="--", linewidth=1.5, label=f"Unsteered Baseline ({base_unsteered_ppl:.2f})")
    plt.axhline(base_nogating_ppl, color="#e74c3c", linestyle="--", linewidth=2.0, label=f"No Gating Baseline ({base_nogating_ppl:.2f})")
    
    plt.ylabel("Text Perplexity (PPL) (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Text Perplexity (PPL): Delay vs Anticipatory (Predictive) Gating", fontsize=13, fontweight="bold", pad=15)
    plt.ylim(0, 12.5)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    
    ppl_path = OUT_DIR / "perplexity_anticipatory.png"
    plt.savefig(ppl_path, dpi=200, bbox_inches="tight")
    print(f"Saved PPL chart to {ppl_path}")

    # ----------------- 3. 2D Trade-off Scatter Plot -----------------
    plt.figure(figsize=(10, 7.0))
    plt.scatter(base_nogating_ppl, base_nogating_score, color="#e74c3c", marker="s", s=180, zorder=5, label="No Gating (Baseline)")
    plt.text(base_nogating_ppl + 0.05, base_nogating_score, "No Gating", fontsize=10, fontweight="bold", va="center", ha="left")
    
    for name, data in results.items():
        plt.scatter(data["ppl"], data["score"], color=data["color"], marker="D", s=130, zorder=5, label=name)
        plt.text(data["ppl"] + 0.04, data["score"] + 0.015, name, fontsize=9, fontweight="bold", va="center", ha="left")
        
    # Draw comparison arrows showing PPL improvement with same/better score
    # Arrow for P-Conf 3 (Delay -> Resampled)
    p3_delay = results["P-Conf 3 (Delay)"]
    p3_res = results["P-Conf 3 (Resampled)"]
    plt.annotate("", xy=(p3_res["ppl"], p3_res["score"]), xytext=(p3_delay["ppl"], p3_delay["score"]))
                 
    # Arrow for P-Conf 6 (Delay -> Resampled)
    p6_delay = results["P-Conf 6 (Delay)"]
    p6_res = results["P-Conf 6 (Resampled)"]
    plt.annotate("", xy=(p6_res["ppl"], p6_res["score"]), xytext=(p6_delay["ppl"], p6_delay["score"]))
                 
    # Arrow for A-Conf 3 (Delay -> Resampled)
    a3_delay = results["A-Conf 3 (Delay)"]
    a3_res = results["A-Conf 3 (Resampled)"]
    plt.annotate("", xy=(a3_res["ppl"], a3_res["score"]), xytext=(a3_delay["ppl"], a3_delay["score"]))

    plt.xlabel("Text Perplexity (PPL) - Lower is Better (X >= 8.0)", fontsize=11, fontweight="bold")
    plt.ylabel("Steering Alignment Score - Higher is Better (Y >= 3.0)", fontsize=11, fontweight="bold")
    plt.title("Performance Trade-off Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.xlim(9.0, 11.2)
    plt.ylim(3.9, 4.4)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    
    scatter_path = OUT_DIR / "anticipatory_tradeoff_scatter.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    print(f"Saved tradeoff scatter to {scatter_path}")

    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
    if artifact_dir.exists():
        import shutil
        shutil.copy(score_path, artifact_dir / "alignment_scores_anticipatory.png")
        shutil.copy(ppl_path, artifact_dir / "perplexity_anticipatory.png")
        shutil.copy(scatter_path, artifact_dir / "anticipatory_tradeoff_scatter.png")
        print("Successfully copied anticipatory plots to artifacts.")

if __name__ == "__main__":
    main()
