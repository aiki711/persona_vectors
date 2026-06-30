#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/123_plot_gen_time_results.py
#
# Unified plotting script for Generation-time Dynamic Layer Steering (DLS) comparison:
#   1. Individual heatmaps (Score & PPL) for all 5 traits across 8 methods.
#   2. Summary heatmap (All Traits Avg Score & PPL) across 8 methods.
#   3. Grouped bar chart comparing maximum safe steering scores (Strict Safety) of all 8 methods.
#   4. Dynamic layer transition history line plots for sample prompts.
#
# Inputs: exp_steering_dyn_gen_time_raw/results/{trait}/scores_{method}_Val{alpha}.csv
# Outputs: exp_steering_dyn_gen_time_raw/figures/
#

import argparse
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_LABELS = {
    "extraversion":      "Extraversion",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness":     "Agreeableness",
}
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

# Selected 8 methods (excluding logit_diff)
METHODS = [
    ("DLS Cos-Only",          "cos_only",               "#e67e22"),
    ("DLS Rank-Only",         "rank_only",              "#2c3e50"),
    ("DLS Proj Cos-Only",     "proj_cos_only",          "#3498db"),
    ("DLS Proj Rank-Only",    "proj_rank_only",         "#1abc9c"),
    ("PDF Cos-Only",          "masked_cos_only",        "#f1c40f"),
    ("PDF Rank-Only",         "masked_rank_only",       "#8e44ad"),
    ("PDF Proj Cos-Only",     "masked_proj_cos_only",   "#e74c3c"),
    ("PDF Proj Rank-Only",    "masked_proj_rank_only",  "#d35400"),
]

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

def load_summary(results_dir: Path, axis: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = results_dir / axis
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
                df_csv = pd.read_csv(csv_path)
                if "dyn_score" in df_csv.columns:
                    df_csv["dyn_score"] = df_csv["dyn_score"].replace(0, np.nan)
                
                # Load JSONL
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
                
                if "dyn_reason" in df_csv.columns:
                    coherence_rate = df_csv["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean()
                else:
                    coherence_rate = 1.0
                
                max_ppl = df_csv["dyn_ppl"].max() if "dyn_ppl" in df_csv.columns else np.nan
                
                records.append({
                    "val":       val,
                    "dyn_score": df_csv["dyn_score"].mean(),
                    "dyn_ppl":   df_csv["dyn_ppl"].mean(),
                    "dyn_coherence_rate": coherence_rate,
                    "dyn_max_ppl": max_ppl,
                    "dyn_max_3gram_rep": max_3gram,
                    "dyn_max_4gram_rep": max_4gram,
                })
            except Exception:
                pass
    return pd.DataFrame(records)

def load_all_methods(results_dir: Path, axis: str):
    data_dict = {}
    for display_name, loader_key, _ in METHODS:
        data_dict[loader_key] = load_summary(results_dir, axis, loader_key)
    return data_dict

def build_pivot(method_data_dict):
    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}
    coherence_rows = {v: {} for v in VALS}
    max_ppl_rows = {v: {} for v in VALS}
    rep3_rows = {v: {} for v in VALS}
    rep4_rows = {v: {} for v in VALS}

    for display_name, loader_key, _ in METHODS:
        df = method_data_dict.get(loader_key, pd.DataFrame())
        if df.empty:
            continue
        idx = df.set_index("val")
        for val in VALS:
            if val in idx.index:
                score_rows[val][display_name] = idx.loc[val, "dyn_score"]
                ppl_rows[val][display_name]   = idx.loc[val, "dyn_ppl"]
                coherence_rows[val][display_name] = idx.loc[val, "dyn_coherence_rate"]
                max_ppl_rows[val][display_name]   = idx.loc[val, "dyn_max_ppl"]
                rep3_rows[val][display_name]      = idx.loc[val, "dyn_max_3gram_rep"]
                rep4_rows[val][display_name]      = idx.loc[val, "dyn_max_4gram_rep"]

    p_score = pd.DataFrame.from_dict(score_rows, orient="index").reindex(VALS)
    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index").reindex(VALS)
    p_coherence = pd.DataFrame.from_dict(coherence_rows, orient="index").reindex(VALS)
    p_max_ppl = pd.DataFrame.from_dict(max_ppl_rows, orient="index").reindex(VALS)
    p_rep3 = pd.DataFrame.from_dict(rep3_rows, orient="index").reindex(VALS)
    p_rep4 = pd.DataFrame.from_dict(rep4_rows, orient="index").reindex(VALS)

    cols = [m[0] for m in METHODS if m[0] in p_score.columns]
    p_score = p_score[cols]
    p_ppl = p_ppl[cols]
    p_coherence = p_coherence[cols]
    p_max_ppl = p_max_ppl[cols]
    p_rep3 = p_rep3[cols]
    p_rep4 = p_rep4[cols]

    return p_score, p_ppl, p_coherence, p_max_ppl, p_rep3, p_rep4

def highlight_safe_cells(ax, p_ppl, p_coherence, p_max_ppl, p_rep3, p_rep4,
                         ppl_threshold=25.0, coherence_threshold=0.8,
                         max_ppl_threshold=35.0):
    if p_ppl is None or p_ppl.empty:
        return
    for i in range(len(p_ppl.index)):
        for j in range(len(p_ppl.columns)):
            col_name = p_ppl.columns[j]
            val = p_ppl.index[i]
            
            val_ppl = p_ppl.iloc[i, j]
            val_coherence = p_coherence.loc[val, col_name] if col_name in p_coherence.columns else np.nan
            val_max_ppl = p_max_ppl.loc[val, col_name] if col_name in p_max_ppl.columns else np.nan
            
            if (not np.isnan(val_ppl) and not np.isnan(val_coherence) and 
                not np.isnan(val_max_ppl)):
                
                is_safe = (
                    val_ppl <= ppl_threshold and
                    val_coherence >= coherence_threshold and
                    val_max_ppl <= max_ppl_threshold
                )
                if is_safe:
                    rect = Rectangle((j, i), 1, 1, fill=False,
                                      edgecolor="black", lw=2.5, clip_on=False)
                    ax.add_patch(rect)

def plot_trait(axis, method_data_dict, out_dir, artifact_dir, title_prefix):
    print(f"[{axis}] plotting generation-time DLS comparison heatmap...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_score, p_ppl, p_coherence, p_max_ppl, p_rep3, p_rep4 = build_pivot(method_data_dict)
    
    n_methods = len(p_score.columns) if not p_score.empty else 1
    fig_w = max(10, n_methods * 1.5 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 13))

    configs = [
        (axes[0], p_score, f"Score [{axis.capitalize()}]",
         "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   f"PPL   [{axis.capitalize()}]",
         "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, title, cmap, vmin, vmax, fmt in configs:
        if p_data.empty:
            ax_obj.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_obj.transAxes, fontsize=14)
            continue
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.8, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 9})
        highlight_safe_cells(ax_obj, p_ppl, p_coherence, p_max_ppl, None, None)
        ax_obj.set_title(
            f"{title} (Black Border: Strict Safety Criteria)",
            fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Generation-Time DLS Method", fontsize=10)
        ax_obj.set_ylabel("Steering Intensity (Alpha / Val)", fontsize=10)

    plt.suptitle(
        f"{title_prefix} DLS 8-Method Comparison: {axis.capitalize()} (Generation-Time DLS)",
        fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    file_name = f"heatmap_dyn_{axis}.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / f"{title_prefix.lower()}_heatmap_dyn_{axis}.png"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")

def plot_summary(all_method_data, out_dir, artifact_dir, title_prefix):
    print("[Summary] plotting DLS comparison summary heatmap (all traits avg)...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    score_acc     = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    ppl_acc       = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    coherence_acc = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    max_ppl_acc   = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    rep3_acc      = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    rep4_acc      = {k: {v: [] for v in VALS} for _, k, _ in METHODS}

    for method_data_dict in all_method_data:
        for _, loader_key, _ in METHODS:
            df = method_data_dict.get(loader_key, pd.DataFrame())
            if df.empty:
                continue
            idx = df.set_index("val")
            for val in VALS:
                if val in idx.index:
                    score_acc[loader_key][val].append(idx.loc[val, "dyn_score"])
                    ppl_acc[loader_key][val].append(idx.loc[val, "dyn_ppl"])
                    coherence_acc[loader_key][val].append(idx.loc[val, "dyn_coherence_rate"])
                    max_ppl_acc[loader_key][val].append(idx.loc[val, "dyn_max_ppl"])
                    rep3_acc[loader_key][val].append(idx.loc[val, "dyn_max_3gram_rep"])
                    rep4_acc[loader_key][val].append(idx.loc[val, "dyn_max_4gram_rep"])

    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}
    coherence_rows = {v: {} for v in VALS}
    max_ppl_rows   = {v: {} for v in VALS}
    rep3_rows      = {v: {} for v in VALS}
    rep4_rows      = {v: {} for v in VALS}
    
    for display_name, loader_key, _ in METHODS:
        for val in VALS:
            scores = score_acc[loader_key][val]
            ppls   = ppl_acc[loader_key][val]
            coherences = coherence_acc[loader_key][val]
            max_ppls   = max_ppl_acc[loader_key][val]
            rep3s      = rep3_acc[loader_key][val]
            rep4s      = rep4_acc[loader_key][val]
            
            if scores:
                score_rows[val][display_name] = np.mean(scores)
                ppl_rows[val][display_name]   = np.mean(ppls)
                coherence_rows[val][display_name] = np.mean(coherences)
                max_ppl_rows[val][display_name]   = np.mean(max_ppls)
                rep3_rows[val][display_name]      = np.mean(rep3s)
                rep4_rows[val][display_name]      = np.mean(rep4s)

    p_score = pd.DataFrame.from_dict(score_rows, orient="index").reindex(VALS)
    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index").reindex(VALS)
    p_coherence = pd.DataFrame.from_dict(coherence_rows, orient="index").reindex(VALS)
    p_max_ppl = pd.DataFrame.from_dict(max_ppl_rows, orient="index").reindex(VALS)
    p_rep3 = pd.DataFrame.from_dict(rep3_rows, orient="index").reindex(VALS)
    p_rep4 = pd.DataFrame.from_dict(rep4_rows, orient="index").reindex(VALS)

    cols = [m[0] for m in METHODS if m[0] in p_score.columns]
    p_score = p_score[cols]
    p_ppl = p_ppl[cols]
    p_coherence = p_coherence[cols]
    p_max_ppl = p_max_ppl[cols]
    p_rep3 = p_rep3[cols]
    p_rep4 = p_rep4[cols]

    n_methods = len(p_score.columns) if not p_score.empty else 1
    fig_w = max(10, n_methods * 1.5 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 13))

    configs = [
        (axes[0], p_score, "Score (All Traits Avg)",
         "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   "PPL   (All Traits Avg)",
         "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, title, cmap, vmin, vmax, fmt in configs:
        if p_data.empty:
            ax_obj.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_obj.transAxes, fontsize=14)
            continue
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.8, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 9})
        highlight_safe_cells(ax_obj, p_ppl, p_coherence, p_max_ppl, None, None)
        ax_obj.set_title(
            f"{title} (Black Border: Strict Safety Criteria)",
            fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Generation-Time DLS Method", fontsize=10)
        ax_obj.set_ylabel("Steering Intensity (Alpha / Val)", fontsize=10)

    plt.suptitle(
        f"{title_prefix} DLS 8-Method Comparison Summary (All Traits Avg) (Generation-Time DLS)",
        fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    file_name = "summary_dyn_all_traits.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / f"{title_prefix.lower()}_summary_dyn_all_traits.png"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")

def get_max_safe_score(results_dir: Path, trait: str, method: str) -> tuple[float, float, float, float]:
    best_score = 0.0
    best_alpha = np.nan
    best_ppl = np.nan
    best_coherence = np.nan
    
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
                
                # Load JSONL
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
                
                # Strict Safety check
                is_safe = (
                    mean_ppl <= 25.0 and
                    coherence_rate >= 0.8 and
                    max_ppl <= 35.0
                )
                
                if is_safe:
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = val
                        best_ppl = mean_ppl
                        best_coherence = coherence_rate
            except Exception:
                pass
    return best_score, best_alpha, best_ppl, best_coherence

def get_unsteered_baseline_score(results_dir: Path, trait: str) -> float:
    trait_dir = results_dir / trait
    # Look for any CSV file to load base_score
    for display_name, loader_key, _ in METHODS:
        for val in VALS:
            csv_path = trait_dir / f"scores_{loader_key}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_{loader_key}_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "base_score" in df.columns:
                        df["base_score"] = df["base_score"].replace(0, np.nan)
                        val_mean = df["base_score"].mean()
                        if not np.isnan(val_mean):
                            return val_mean
                except Exception:
                    pass
    return 3.0

def plot_max_safe_bar(results_dir: Path, out_dir: Path, artifact_dir: Path, title_prefix: str):
    print("Generating max safe score grouped bar chart...")
    data = []
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(results_dir, trait)
        method_results = {}
        for display_name, loader_key, _ in METHODS:
            score, alpha, ppl, coherence = get_max_safe_score(results_dir, trait, loader_key)
            method_results[display_name] = (score, alpha, ppl, coherence)
            
        data.append({
            "trait": TRAIT_LABELS[trait],
            "Unsteered Baseline": (ub_score, np.nan, np.nan, np.nan),
            **method_results
        })

    # Calculate average
    avg_ub = np.mean([d["Unsteered Baseline"][0] for d in data])
    avg_results = {}
    for display_name, loader_key, _ in METHODS:
        avg_score = np.mean([d[display_name][0] for d in data if d[display_name][0] > 0.0])
        avg_results[display_name] = (avg_score, np.nan, np.nan, np.nan)
        
    data.append({
        "trait": "Average",
        "Unsteered Baseline": (avg_ub, np.nan, np.nan, np.nan),
        **avg_results
    })

    # Plot setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    
    num_bars = 1 + len(METHODS)
    width = 0.08
    
    fig, ax = plt.subplots(figsize=(24, 10))
    
    colors = {
        "Unsteered Baseline": "#7f8c8d"
    }
    for display_name, _, color in METHODS:
        colors[display_name] = color
        
    offset_start = - (num_bars - 1) / 2.0
    
    rects_list = []
    labels_list = []
    
    # Plot Unsteered
    rects_list.append(ax.bar(x + (offset_start * width), [d["Unsteered Baseline"][0] for d in data], width, label="Unsteered Baseline", color=colors["Unsteered Baseline"], zorder=3))
    labels_list.append("Unsteered Baseline")
    
    # Plot 8 methods
    for i, (display_name, _, _) in enumerate(METHODS):
        rects_list.append(ax.bar(x + ((offset_start + 1 + i) * width), [d[display_name][0] for d in data], width, label=display_name, color=colors[display_name], zorder=3))
        labels_list.append(display_name)
        
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    
    title_text = f"{title_prefix} DLS — Maximum Safe Steering Score Comparison (Generation-Time DLS)"
    ax.set_title(title_text, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)

    for r_idx, rects in enumerate(rects_list):
        data_key = labels_list[r_idx]
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height == 0.0 or np.isnan(height):
                continue
            
            label_text = f"{height:.2f}"
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                        color="#333333")
            
            info = data[i][data_key]
            alpha_val = info[1]
            if not np.isnan(alpha_val):
                alpha_text = f"α={alpha_val}"
                ax.annotate(alpha_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -14),
                            textcoords="offset points",
                            ha="center", va="top",
                            fontsize=6.5, color="white", fontweight="semibold", rotation=90)

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9, ncol=2)
    
    file_name = "max_safe_score_compare.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison bar chart to: {out_path}")
    
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / f"{title_prefix.lower()}_max_safe_score_compare.png"
        shutil.copy(out_path, dest_path)
        print(f"Copied comparison bar chart to artifacts: {dest_path}")

def plot_layer_history_samples(results_dir: Path, out_dir: Path, artifact_dir: Path):
    """
    Reads a sample JSONL file and plots the token-by-token selected layer transition
    for the first two prompts to visually show how selection shifts during generation.
    """
    print("Plotting layer selection history samples...")
    sample_trait = "agreeableness"
    sample_file = results_dir / sample_trait / "masked_proj_cos_only_Val4.0.jsonl"
    
    # Fallback to any file if this specific one isn't finished yet
    if not sample_file.exists():
        jsonl_files = list(results_dir.glob("**/*.jsonl"))
        if jsonl_files:
            sample_file = jsonl_files[0]
            print(f"  Fallback sample file: {sample_file}")
        else:
            print("  No JSONL files found yet to plot layer history.")
            return

    try:
        histories = []
        texts = []
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "layer_history" in data and data["layer_history"]:
                    histories.append(data["layer_history"])
                    texts.append(data["dyn_text"])
                if len(histories) >= 2:
                    break

        if not histories:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, hist in enumerate(histories):
            tokens_idx = np.arange(len(hist))
            ax.plot(tokens_idx, hist, marker="o", markersize=3, alpha=0.8, linewidth=1.5, label=f"Prompt {i+1}")

        ax.set_title(f"Generation-Time Layer Selection Transitions ({sample_file.stem})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Generated Token Index", fontsize=11)
        ax.set_ylabel("Steered Layer (Candidate Range: 4 to 29)", fontsize=11)
        ax.set_ylim(3.5, 30.5)
        ax.set_yticks(range(4, 30, 2))
        ax.grid(axis="both", linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white")

        out_path = out_dir / "layer_history_samples.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved layer history sample plot to: {out_path}")

        if artifact_dir:
            dest = artifact_dir / "layer_history_samples.png"
            shutil.copy(out_path, dest)
            print(f"Copied layer history sample plot to artifacts: {dest}")
    except Exception as e:
        print(f"Failed to plot layer history samples: {e}")

def main():
    ap = argparse.ArgumentParser(description="Plot generation-time dynamic steering method comparison.")
    ap.add_argument("--results_dir", required=True, help="Path to results directory")
    ap.add_argument("--out_dir", required=True, help="Output folder for figures")
    ap.add_argument("--artifact_dir", default=None, help="Folder to copy results for conversation viewing")
    ap.add_argument("--title_prefix", default="Raw_GenTime", help="Title prefix")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir      = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    title_prefix = args.title_prefix

    all_method_data = []

    for axis in TRAITS:
        method_data = load_all_methods(results_dir, axis)
        all_method_data.append(method_data)
        plot_trait(axis, method_data, out_dir / axis, artifact_dir, title_prefix)

    plot_summary(all_method_data, out_dir, artifact_dir, title_prefix)
    plot_max_safe_bar(results_dir, out_dir, artifact_dir, title_prefix)
    plot_layer_history_samples(results_dir, out_dir, artifact_dir)
    print(f"\nGeneration-time DLS plotting finished successfully.")

if __name__ == "__main__":
    main()
