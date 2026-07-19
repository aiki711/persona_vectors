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

# All IC-based dynamic gating configurations tested
ALL_CONFIGS = {
    # Name: (theta_lo, theta_hi, k_lo, k_hi, gating_mode, score, ppl)
    "P-Conf 2 (Base Plat)": (3.0, 7.0, 2.0, 2.0, "plateau", 3.72, 9.35),
    "P-Conf 3 (Wider Plat)": (1.0, 9.0, 2.0, 2.0, "plateau", 4.20, 10.79),
    "P-Conf 4 (Narrow Plat)": (4.0, 6.0, 2.0, 2.0, "plateau", 3.60, 9.56),
    "P-Conf 5 (Sharp Plat)": (3.0, 7.0, 8.0, 8.0, "plateau", 3.58, 9.43),
    "P-Conf 6 (Gentle Plat)": (3.0, 7.0, 0.5, 0.5, "plateau", 4.12, 9.96),
    "A-Conf 1 (Gent/Sharp)": (3.0, 7.0, 0.5, 8.0, "max_normalized", 3.52, 8.99),
    "A-Conf 2 (Sharp/Gent)": (3.0, 7.0, 8.0, 0.5, "max_normalized", 3.52, 9.19),
    "A-Conf 3 (Low IC Focus)": (1.0, 5.0, 1.0, 4.0, "max_normalized", 4.12, 9.42),
    "A-Conf 4 (High IC Focus)": (5.0, 9.0, 4.0, 1.0, "max_normalized", 3.22, 9.19),
    "Opt-Plat-1": (2.0, 6.0, 1.5, 8.0, "plateau", 3.92, 9.95),
    "Opt-Plat-2": (2.0, 5.5, 1.5, 10.0, "plateau", 3.96, 9.62),
    "Opt-Plat-3": (2.0, 6.5, 1.5, 8.0, "plateau", 3.96, 9.84),
    "Opt-Plat-4 (High-Pass)": (2.0, 15.0, 2.0, 2.0, "plateau", 3.80, 9.76),
    "Optimized Gating": (2.0, 7.0, 1.0, 4.0, "plateau", 3.98, 9.99),
}

def sigmoid(x, k, theta):
    z = -k * (x - theta)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(z))

def gating_function(ic, k_lo, k_hi, theta_lo, theta_hi):
    g = sigmoid(ic, k_lo, theta_lo) * sigmoid(ic, -k_hi, theta_hi)
    grid = np.linspace(min(theta_lo, theta_hi) - 2.0, max(theta_lo, theta_hi) + 2.0, 1000)
    g_grid = sigmoid(grid, k_lo, theta_lo) * sigmoid(grid, -k_hi, theta_hi)
    g_max = np.max(g_grid)
    return g / g_max

def main():
    # ----------------- 1. Plot Gating Curves Matrix -----------------
    plt.close("all")
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    
    ic_values = np.linspace(0, 15, 500)
    
    for idx, (name, val) in enumerate(ALL_CONFIGS.items()):
        theta_lo, theta_hi, k_lo, k_hi, mode, score, ppl = val
        ax = axes[idx]
        gain = gating_function(ic_values, k_lo, k_hi, theta_lo, theta_hi)
        
        # Color categorizing
        if "P-Conf" in name:
            color = "#3498db" # Blue
        elif "A-Conf" in name:
            color = "#9b59b6" # Purple
        else:
            color = "#2ecc71" # Green
            
        ax.plot(ic_values, gain * 5.0, color=color, linewidth=2.0)
        ax.fill_between(ic_values, 0, gain * 5.0, color=color, alpha=0.08)
        
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.2, 5.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        # Labeling details inside subplot
        box_text = f"θ: {theta_lo}-{theta_hi}\nk: {k_lo}-{k_hi}\nScore: {score:.2f}\nPPL: {ppl:.2f}"
        ax.text(0.05, 0.95, box_text, transform=ax.transAxes, fontsize=9.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#dddddd", alpha=0.85))
        
        ax.set_title(name, fontsize=11, fontweight="bold")
        
    # Disable unused subplots (we have 14 configs, so 16 - 14 = 2 unused axes)
    for i in range(14, 16):
        fig.delaxes(axes[i])
        
    # Axis labels
    for i in range(4):
        for j in range(4):
            idx = 4 * i + j
            if idx < 14:
                if i == 3 or (i == 2 and j >= 2): # bottom row subplots
                    axes[idx].set_xlabel("IC [bits]", fontsize=10)
                if j == 0:
                    axes[idx].set_ylabel("Steering α", fontsize=10)
                    
    plt.suptitle("Tested Information-Content Gating Function Shapes ( Mistral-7B )", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    matrix_path = OUT_DIR / "gating_curves_matrix.png"
    plt.savefig(matrix_path, dpi=200, bbox_inches="tight")
    print(f"Saved matrix curves to {matrix_path}")

    # ----------------- 2. Plot Alignment Scores Bar Chart -----------------
    # Extract data for bar charts
    names = list(ALL_CONFIGS.keys())
    scores = [val[5] for val in ALL_CONFIGS.values()]
    ppls = [val[6] for val in ALL_CONFIGS.values()]
    
    # Sort by alignment score
    sorted_indices_score = np.argsort(scores)[::-1]
    sorted_names_score = [names[i] for i in sorted_indices_score]
    sorted_scores = [scores[i] for i in sorted_indices_score]
    
    # Baselines
    base_unsteered_score = 3.12
    base_nogating_score = 4.34
    
    plt.figure(figsize=(12, 7))
    bar_colors_score = []
    for name in sorted_names_score:
        if "P-Conf" in name:
            bar_colors_score.append("#3498db")
        elif "A-Conf" in name:
            bar_colors_score.append("#9b59b6")
        else:
            bar_colors_score.append("#2ecc71")
            
    bars = plt.bar(sorted_names_score, sorted_scores, color=bar_colors_score, edgecolor="black", alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=9.5, fontweight="bold")
        
    plt.axhline(base_nogating_score, color="#e74c3c", linestyle="--", linewidth=2.0, label=f"No Gating Baseline ({base_nogating_score:.2f})")
    plt.axhline(base_unsteered_score, color="#7f8c8d", linestyle="--", linewidth=1.5, label=f"Unsteered Baseline ({base_unsteered_score:.2f})")
    
    plt.ylabel("Steering Alignment Score (Higher is Better)", fontsize=12, fontweight="bold")
    plt.title("Steering Alignment Score Comparison (Sorted by Score)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylim(1.0, 5.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    
    score_path = OUT_DIR / "alignment_scores_comparison.png"
    plt.savefig(score_path, dpi=200, bbox_inches="tight")
    print(f"Saved score bar chart to {score_path}")

    # ----------------- 3. Plot Perplexity (PPL) Bar Chart -----------------
    # Sort by perplexity (lower is better, so ascending order)
    sorted_indices_ppl = np.argsort(ppls)
    sorted_names_ppl = [names[i] for i in sorted_indices_ppl]
    sorted_ppls = [ppls[i] for i in sorted_indices_ppl]
    
    base_unsteered_ppl = 5.66
    base_nogating_ppl = 10.46
    
    plt.figure(figsize=(12, 7))
    bar_colors_ppl = []
    for name in sorted_names_ppl:
        if "P-Conf" in name:
            bar_colors_ppl.append("#3498db")
        elif "A-Conf" in name:
            bar_colors_ppl.append("#9b59b6")
        else:
            bar_colors_ppl.append("#2ecc71")
            
    bars = plt.bar(sorted_names_ppl, sorted_ppls, color=bar_colors_ppl, edgecolor="black", alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f}", ha='center', va='bottom', fontsize=9.5, fontweight="bold")
        
    plt.axhline(base_unsteered_ppl, color="#7f8c8d", linestyle="--", linewidth=1.5, label=f"Unsteered Baseline ({base_unsteered_ppl:.2f})")
    plt.axhline(base_nogating_ppl, color="#e74c3c", linestyle="--", linewidth=2.0, label=f"No Gating Baseline ({base_nogating_ppl:.2f})")
    
    plt.ylabel("Text Perplexity (PPL) (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Text Perplexity (PPL) Comparison (Sorted by PPL - Lower is Better)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 12.5)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    
    ppl_path = OUT_DIR / "perplexity_comparison.png"
    plt.savefig(ppl_path, dpi=200, bbox_inches="tight")
    print(f"Saved PPL bar chart to {ppl_path}")

    # ----------------- 4. Plot 2D Performance Trade-off Scatter Plot -----------------
    plt.figure(figsize=(11, 8))
    
    # Baselines
    baselines = {
        "No Steering": {"score": 3.12, "ppl": 5.66, "color": "#7f8c8d", "marker": "o"},
        "No Gating (PDF Proj Rank)": {"score": 4.34, "ppl": 10.46, "color": "#e74c3c", "marker": "s"},
    }
    
    # Plot baselines
    for name, base_data in baselines.items():
        plt.scatter(base_data["ppl"], base_data["score"], color=base_data["color"], marker=base_data["marker"], s=180, zorder=5, label=name)
        # Only annotate if within visible bounds (No Steering will be cut off by xlim>=8)
        if base_data["ppl"] >= 8.0:
            plt.text(base_data["ppl"] + 0.05, base_data["score"], name, fontsize=10, fontweight="bold", va="center", ha="left")

    # Plot gating configs
    for name, val in ALL_CONFIGS.items():
        theta_lo, theta_hi, k_lo, k_hi, mode, score, ppl = val
        if "P-Conf" in name:
            color = "#3498db"
            marker = "^"
        elif "A-Conf" in name:
            color = "#9b59b6"
            marker = "v"
        else:
            color = "#2ecc71"
            marker = "D"
            
        # Draw scatter point
        plt.scatter(ppl, score, color=color, marker=marker, s=120, zorder=5)
        
        # Position labels dynamically to minimize overlap
        y_offset = 0.012
        x_offset = 0.02
        if "P-Conf 3" in name:
            y_offset = 0.02
        elif "A-Conf 3" in name:
            y_offset = -0.025
        elif "Optimized Gating" in name:
            x_offset = -0.04
            y_offset = -0.025
        elif "P-Conf 6" in name:
            y_offset = 0.02
            
        plt.text(ppl + x_offset, score + y_offset, name.split(" (")[0], fontsize=8.5, va="center", ha="left",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.75))

    plt.xlabel("Text Perplexity (PPL) - Lower is Better (X >= 8.0)", fontsize=12, fontweight="bold")
    plt.ylabel("Steering Alignment Score - Higher is Better (Y >= 3.0)", fontsize=12, fontweight="bold")
    plt.title("Performance Trade-off: Alignment Score vs Text Perplexity (IC Gating)", fontsize=14, fontweight="bold", pad=15)
    
    # User constraint: Score >= 3, PPL >= 8
    plt.xlim(8.0, 11.2)
    plt.ylim(3.0, 4.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='No Gating (Baseline)', markerfacecolor='#e74c3c', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='P-Conf (Plateau)', markerfacecolor='#3498db', markersize=10),
        Line2D([0], [0], marker='v', color='w', label='A-Conf (Asymmetric)', markerfacecolor='#9b59b6', markersize=10),
        Line2D([0], [0], marker='D', color='w', label='Opt-Plat / Optimized', markerfacecolor='#2ecc71', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="lower left", fontsize=10)
    plt.tight_layout()
    
    scatter_path = OUT_DIR / "performance_tradeoff_scatter.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    print(f"Saved scatter chart to {scatter_path}")

    # Copy all files to artifacts directory
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
    if artifact_dir.exists():
        import shutil
        shutil.copy(matrix_path, artifact_dir / "gating_curves_matrix.png")
        shutil.copy(score_path, artifact_dir / "alignment_scores_comparison.png")
        shutil.copy(ppl_path, artifact_dir / "perplexity_comparison.png")
        shutil.copy(scatter_path, artifact_dir / "performance_tradeoff_scatter.png")
        print("Successfully copied all figures to artifacts.")

if __name__ == "__main__":
    main()
