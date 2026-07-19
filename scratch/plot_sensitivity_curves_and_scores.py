#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Paths
WORKSPACE = Path("/home/s2550009/persona_vectors")
RESULTS_DIR = WORKSPACE / "exp_token_intensity/exp_sensitivity_analysis"
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_sensitivity_analysis"

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

CONFIGS = {
    "Opt-Plat-1": {"theta_lo": 2.0, "theta_hi": 6.0, "k_lo": 1.5, "k_hi": 8.0},
    "Opt-Plat-2": {"theta_lo": 2.0, "theta_hi": 5.5, "k_lo": 1.5, "k_hi": 10.0},
    "Opt-Plat-3": {"theta_lo": 2.0, "theta_hi": 6.5, "k_lo": 1.5, "k_hi": 8.0},
    "Opt-Plat-4 (High-Pass)": {"theta_lo": 2.0, "theta_hi": 15.0, "k_lo": 2.0, "k_hi": 2.0},
    "Optimized Gating": {"theta_lo": 2.0, "theta_hi": 7.0, "k_lo": 1.0, "k_hi": 4.0},
}

def sigmoid(x, k, theta):
    z = -k * (x - theta)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(z))

def gating_function(ic, k_lo, k_hi, theta_lo, theta_hi):
    g = sigmoid(ic, k_lo, theta_lo) * sigmoid(ic, -k_hi, theta_hi)
    # Find theoretical max to normalize
    grid = np.linspace(min(theta_lo, theta_hi) - 2.0, max(theta_lo, theta_hi) + 2.0, 1000)
    g_grid = sigmoid(grid, k_lo, theta_lo) * sigmoid(grid, -k_hi, theta_hi)
    g_max = np.max(g_grid)
    return g / g_max

def main():
    # Load results dynamically
    results_data = {}
    for name, params in CONFIGS.items():
        scores, ppls = [], []
        t_lo, t_hi = params["theta_lo"], params["theta_hi"]
        k_lo, k_hi = params["k_lo"], params["k_hi"]
        
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{t_lo}_{t_hi}_k_{k_lo}_{k_hi}_plateau_Val5.0.csv"
            csv_path = RESULTS_DIR / trait / csv_name
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    scores.append(df["dyn_score"].mean())
                    ppls.append(df["dyn_ppl"].mean())
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                    
        if scores:
            results_data[name] = {"score": np.mean(scores), "ppl": np.mean(ppls)}
            print(f"Loaded {name}: Score = {np.mean(scores):.3f}, PPL = {np.mean(ppls):.3f}")
        else:
            # Fallback values if files are missing
            fallbacks = {
                "Opt-Plat-1": {"score": 3.920, "ppl": 9.949},
                "Opt-Plat-2": {"score": 3.960, "ppl": 9.619},
                "Opt-Plat-3": {"score": 3.960, "ppl": 9.841},
                "Opt-Plat-4 (High-Pass)": {"score": 3.800, "ppl": 9.755},
                "Optimized Gating": {"score": 3.980, "ppl": 9.985},
            }
            results_data[name] = fallbacks[name]
            print(f"Fallback {name}: Score = {fallbacks[name]['score']:.3f}, PPL = {fallbacks[name]['ppl']:.3f}")

    # Add baselines
    baselines = {
        "No Steering": {"score": 3.120, "ppl": 5.660, "color": "#7f8c8d", "marker": "o"},
        "No Gating (PDF Proj Rank)": {"score": 4.340, "ppl": 10.460, "color": "#e74c3c", "marker": "s"},
    }

    # Color palette for gating configurations
    colors = {
        "Opt-Plat-1": "#1abc9c",
        "Opt-Plat-2": "#3498db",
        "Opt-Plat-3": "#9b59b6",
        "Opt-Plat-4 (High-Pass)": "#e67e22",
        "Optimized Gating": "#2ecc71",
    }

    # ----------------- Plotting -----------------
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # 1. Gating Curves
    ic_values = np.linspace(0, 15, 500)
    for name, params in CONFIGS.items():
        gain = gating_function(ic_values, params["k_lo"], params["k_hi"], params["theta_lo"], params["theta_hi"])
        ax1.plot(ic_values, gain * 5.0, label=name, color=colors[name], linewidth=2.5)
        ax1.fill_between(ic_values, 0, gain * 5.0, color=colors[name], alpha=0.05)

    ax1.axhline(5.0, color="#e74c3c", linestyle="--", linewidth=1.8, label="No Gating (α=5.0)")
    ax1.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.2)

    ax1.set_xlabel("Information Content (IC) [bits]", fontsize=12)
    ax1.set_ylabel("Steering Intensity (α)", fontsize=12)
    ax1.set_title("Gating Curves: Steering Strength α vs Token Information Content", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 12)
    ax1.set_ylim(-0.2, 5.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=10)

    # 2. Score vs PPL Trade-off Scatter Plot
    # Plot baselines
    for name, base_data in baselines.items():
        ax2.scatter(base_data["ppl"], base_data["score"], color=base_data["color"], marker=base_data["marker"], s=150, zorder=5, label=name)
        ax2.text(base_data["ppl"] + 0.08, base_data["score"], name, fontsize=10, fontweight="bold", va="center", ha="left")

    # Plot configs
    for name, data in results_data.items():
        ax2.scatter(data["ppl"], data["score"], color=colors[name], marker="D", s=120, zorder=5, label=name)
        
        # Position annotations slightly offset to avoid overlap
        y_offset = 0.0
        if "Opt-Plat-3" in name:
            y_offset = -0.06
        elif "Opt-Plat-2" in name:
            y_offset = 0.06
            
        ax2.text(data["ppl"] + 0.08, data["score"] + y_offset, f"{name}\n(PPL: {data['ppl']:.2f}, Score: {data['score']:.2f})", 
                 fontsize=9, va="center", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[name], alpha=0.85))

    # Draw arrow showing trade-off direction
    ax2.annotate("Trade-off Direction\n(Better PPL, Lower Score)", 
                 xy=(9.7, 3.85), xytext=(10.3, 3.3),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, ha="center")

    ax2.set_xlabel("Text Perplexity (PPL) - Lower is Better", fontsize=12)
    ax2.set_ylabel("Steering Alignment Score - Higher is Better", fontsize=12)
    ax2.set_title("Performance Trade-off: Alignment Score vs Text Perplexity", fontsize=13, fontweight="bold")
    ax2.set_xlim(5.0, 11.5)
    ax2.set_ylim(2.8, 4.6)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=10)

    plt.suptitle("Information-Content Dynamic Steering: Gating Curve & Performance Analysis", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save
    out_path = OUT_DIR / "sensitivity_gating_curves_and_scores.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
    if artifact_dir.exists():
        import shutil
        shutil.copy(out_path, artifact_dir / "sensitivity_gating_curves_and_scores.png")
        print(f"Copied figure to artifacts: {artifact_dir / 'sensitivity_gating_curves_and_scores.png'}")

if __name__ == "__main__":
    main()
