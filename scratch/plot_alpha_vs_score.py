#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# scratch/plot_alpha_vs_score.py
#
# Plots Alpha vs Personality Score for logit_diff, cos_only, and rank_only.
# Loads results dynamically from unseen test results (extraversion) to ensure consistency.
# Only plots the safe region (PPL <= 25).
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Data setup
vals = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
results_base_dir = Path("exp_steering_dyn_layer_proj_prior/results_test_unseen")

def load_method_data(method_name):
    y_scores = []
    y_ppls = []
    valid_vals = []
    
    for val in vals:
        val_scores = []
        val_ppls = []
        for trait in traits:
            csv_path = results_base_dir / trait / f"scores_{method_name}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = results_base_dir / trait / f"scores_{method_name}_Val{val}.csv"
                
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "dyn_score" in df.columns:
                        df["dyn_score"] = df["dyn_score"].replace(0, 1)
                    val_scores.append(df["dyn_score"].mean())
                    val_ppls.append(df["dyn_ppl"].mean())
                except Exception:
                    pass
        
        if val_scores:
            y_scores.append(np.mean(val_scores))
            y_ppls.append(np.mean(val_ppls))
            valid_vals.append(val)
            
    return np.array(valid_vals), np.array(y_scores), np.array(y_ppls)

def get_safe_only(x, y_score, y_ppl, threshold=25.0):
    """Returns only the safe (PPL <= threshold) segments."""
    x_safe, y_safe = [], []
    for val, score, ppl in zip(x, y_score, y_ppl):
        if ppl <= threshold:
            x_safe.append(val)
            y_safe.append(score)
    return np.array(x_safe), np.array(y_safe)

# Load data dynamically
x_l, y_l_score, y_l_ppl = load_method_data("logit_diff")
x_c, y_c_score, y_c_ppl = load_method_data("cos_only")
x_r, y_r_score, y_r_ppl = load_method_data("rank_only")

# Filter safe only
x_l_safe, y_l_safe = get_safe_only(x_l, y_l_score, y_l_ppl)
x_c_safe, y_c_safe = get_safe_only(x_c, y_c_score, y_c_ppl)
x_r_safe, y_r_safe = get_safe_only(x_r, y_r_score, y_r_ppl)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. logit_diff (Slate/Gray)
plt.plot(x_l_safe, y_l_safe, color='#475569', label='logit_diff', linewidth=2.5, marker='o', markersize=6)

# 2. cos_only (Amber/Orange)
plt.plot(x_c_safe, y_c_safe, color='#d97706', label='cos_only', linewidth=2.5, marker='o', markersize=6)

# 3. rank_only (Emerald/Green)
plt.plot(x_r_safe, y_r_safe, color='#059669', label='rank_only (Proposed)', linewidth=3.0, marker='o', markersize=8)

# Annotate best safe points based on the actual loaded data
if len(y_r_safe) > 0:
    idx_r_best = np.argmax(y_r_safe)
    best_r_alpha = x_r_safe[idx_r_best]
    best_r_score = y_r_safe[idx_r_best]
    # find corresponding ppl
    best_r_ppl = y_r_ppl[np.where(x_r == best_r_alpha)[0][0]]
    plt.annotate(f"Best Rank-Only\n(Score: {best_r_score:.2f}, PPL: {best_r_ppl:.1f})", 
                 xy=(best_r_alpha, best_r_score), 
                 xytext=(best_r_alpha - 2.5, best_r_score + 0.12),
                 arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight='bold', color='#065f46')

if len(y_c_safe) > 0:
    idx_c_best = np.argmax(y_c_safe)
    best_c_alpha = x_c_safe[idx_c_best]
    best_c_score = y_c_safe[idx_c_best]
    best_c_ppl = y_c_ppl[np.where(x_c == best_c_alpha)[0][0]]
    plt.annotate(f"Best Cos-Only\n(Score: {best_c_score:.2f}, PPL: {best_c_ppl:.1f})", 
                 xy=(best_c_alpha, best_c_score), 
                 xytext=(best_c_alpha + 0.5, best_c_score - 0.15),
                 arrowprops=dict(facecolor='#d97706', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight='bold', color='#92400e')

# Styling to fit the bounds tightly
plt.title("DLS Method Comparison: Steering Strength (Alpha) vs Personality Score (Safe Region: PPL $\leq$ 25)", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Steering Strength (Alpha / Val)", fontsize=10)
plt.ylabel("Avg Personality Score (1.0 - 5.0)", fontsize=10)

# Set x-limits tightly to fit the plotted range
max_safe_alpha = max(x_l_safe.max(), x_c_safe.max(), x_r_safe.max())
plt.xlim(0.3, max_safe_alpha + 0.7)
plt.ylim(2.5, 4.5)  # Adjusted since scores are higher in the actual data (up to 4.2)
plt.xticks([0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0][:len(x_c_safe)])
plt.grid(True, linestyle=':', alpha=0.6)

# Legend placement
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right', fontsize=9.5)

# Ensure tight layout to minimize padding around the edge
plt.tight_layout()

# Save
out_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/967cd169-1aa5-48db-a243-174e45692380/images")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "dls_alpha_vs_score.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Saved figure to: {out_path}")
