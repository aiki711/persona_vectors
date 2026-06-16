#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# scratch/plot_alpha_vs_score_per_trait.py
#
# Plots Alpha vs Personality Score for logit_diff, cos_only, and rank_only,
# both for the average and for each individual personality trait.
# Saves plots to exp_steering_dyn_layer_proj_prior/figures/ and copies them
# to the current active artifact directory.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# Data setup
vals = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
trait_labels = {
    "extraversion": "Extraversion",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness": "Agreeableness"
}

results_base_dir = Path("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen")
new_results_dir = Path("exp_steering_dyn_layer_proj_prior/results")
out_dir = Path("exp_steering_dyn_layer_pdf/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Correct active artifact directory
artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/eb5ffadd-d5e7-40a3-a0b3-5e88bfefda49/images")
artifact_dir.mkdir(parents=True, exist_ok=True)

def load_method_data(method_name, trait_name=None):
    y_scores = []
    y_ppls = []
    valid_vals = []
    
    target_dir = new_results_dir if method_name in ["rank_only", "cos_only"] else results_base_dir
    
    for val in vals:
        val_scores = []
        val_ppls = []
        
        traits_to_load = [trait_name] if trait_name else traits
        
        for trait in traits_to_load:
            csv_path = target_dir / trait / f"scores_{method_name}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = target_dir / trait / f"scores_{method_name}_Val{val}.csv"
                
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

def get_base_score(trait_name=None):
    traits_to_load = [trait_name] if trait_name else traits
    all_base_scores = []
    
    for trait in traits_to_load:
        found = False
        for target_dir in [new_results_dir, results_base_dir]:
            trait_dir = target_dir / trait
            if trait_dir.exists():
                csv_files = list(trait_dir.glob("scores_*.csv"))
                if csv_files:
                    try:
                        df = pd.read_csv(csv_files[0])
                        if "base_score" in df.columns:
                            all_base_scores.append(df["base_score"].mean())
                            found = True
                            break
                    except Exception:
                        pass
        if not found:
            print(f"Warning: Could not find base_score for trait {trait}")
            
    if all_base_scores:
        return np.mean(all_base_scores)
    return None

def get_safe_only(x, y_score, y_ppl, threshold=25.0):
    """Returns only the safe (PPL <= threshold) segments."""
    x_safe, y_safe = [], []
    for val, score, ppl in zip(x, y_score, y_ppl):
        if ppl <= threshold:
            x_safe.append(val)
            y_safe.append(score)
    return np.array(x_safe), np.array(y_safe)

def plot_and_save(trait_name=None):
    # Load data dynamically
    x_l, y_l_score, y_l_ppl = load_method_data("logit_diff", trait_name)
    x_c, y_c_score, y_c_ppl = load_method_data("cos_only", trait_name)
    x_r, y_r_score, y_r_ppl = load_method_data("rank_only", trait_name)
    
    # Filter out alpha >= 20.0 for agreeableness as requested by the user
    if trait_name == "agreeableness":
        mask_l = x_l < 20.0
        x_l, y_l_score, y_l_ppl = x_l[mask_l], y_l_score[mask_l], y_l_ppl[mask_l]
        
        mask_c = x_c < 20.0
        x_c, y_c_score, y_c_ppl = x_c[mask_c], y_c_score[mask_c], y_c_ppl[mask_c]
        
        mask_r = x_r < 20.0
        x_r, y_r_score, y_r_ppl = x_r[mask_r], y_r_score[mask_r], y_r_ppl[mask_r]
        
    # Filter safe only
    x_l_safe, y_l_safe = get_safe_only(x_l, y_l_score, y_l_ppl)
    x_c_safe, y_c_safe = get_safe_only(x_c, y_c_score, y_c_ppl)
    x_r_safe, y_r_safe = get_safe_only(x_r, y_r_score, y_r_ppl)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 0. Unsteered Base Score (Horizontal Dashed Line)
    base_score = get_base_score(trait_name)
    if base_score is not None:
        plt.axhline(y=base_score, color='#64748b', linestyle='--', linewidth=1.5, label=f'Unsteered Base ({base_score:.2f})', zorder=1)
    
    # 1. logit_diff (Slate/Gray)
    if len(x_l_safe) > 0:
        plt.plot(x_l_safe, y_l_safe, color='#475569', label='logit_diff', linewidth=2.5, marker='o', markersize=6)
    
    # 2. cos_only (Amber/Orange)
    if len(x_c_safe) > 0:
        plt.plot(x_c_safe, y_c_safe, color='#d97706', label='cos_only', linewidth=2.5, marker='o', markersize=6)
    
    # 3. rank_only (Proposed) (Emerald/Green)
    if len(x_r_safe) > 0:
        plt.plot(x_r_safe, y_r_safe, color='#059669', label='rank_only (Proposed)', linewidth=3.0, marker='o', markersize=8)
    
    # Title setting
    title_suffix = f" ({trait_labels[trait_name]})" if trait_name else " (Average)"
    plt.title(f"DLS Method Comparison: Steering Strength (Alpha) vs Personality Score{title_suffix}\n(Safe Region: PPL $\\leq$ 25)", 
              fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Steering Strength (Alpha / Val)", fontsize=10)
    plt.ylabel("Avg Personality Score (1.0 - 5.0)", fontsize=10)
    
    # Set x-limits and ticks
    all_safe_alphas = []
    if len(x_l_safe) > 0: all_safe_alphas.append(x_l_safe.max())
    if len(x_c_safe) > 0: all_safe_alphas.append(x_c_safe.max())
    if len(x_r_safe) > 0: all_safe_alphas.append(x_r_safe.max())
    
    if all_safe_alphas:
        max_safe_alpha = max(all_safe_alphas)
        plt.xlim(0.3, max_safe_alpha + 0.7)
    else:
        plt.xlim(0.3, 10.5)
        
    plt.ylim(1.5, 5.1)
    plt.xticks([0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0])
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Legend
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right', fontsize=9.5)
    plt.tight_layout()
    
    # Filenames
    file_suffix = f"_{trait_name}" if trait_name else "_average"
    filename = f"dls_alpha_vs_score{file_suffix}.png"
    
    fig_path = out_dir / filename
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Saved figure to: {fig_path}")
    
    # Copy to artifacts
    dest_path = artifact_dir / filename
    shutil.copy(fig_path, dest_path)
    print(f"Copied figure to artifacts: {dest_path}")

def main():
    # 1. Plot Average
    plot_and_save(None)
    
    # 2. Plot for each individual trait
    for trait in traits:
        plot_and_save(trait)

if __name__ == "__main__":
    main()
