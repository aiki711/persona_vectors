#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/126_plot_gen_time_interval_results.py
#
# Unified plotting and analysis script for update_interval sweeps (1, 4, 8)
# and Fixed-Layer DLS comparison under the strict safety filter:
#   - Mean PPL <= 25.0
#   - Coherence Rate >= 80%
#   - Max PPL <= 25.0 (All 10 prompts must have PPL <= 25)
#
# Outputs:
#   - Individual and summary heatmaps for interval 4 & 8.
#   - Max safe score comparison bar charts for interval 4 & 8.
#   - A comparative summary chart and table across Fixed-Layer, Interval 1, 4, and 8.
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

def load_summary(results_dir: Path, trait: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = results_dir / trait
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            
        if csv_path.exists():
            try:
                df_csv = pd.read_csv(csv_path)
                score_col = "dyn_score" if "dyn_score" in df_csv.columns else df_csv.columns[2]
                ppl_col = "dyn_ppl" if "dyn_ppl" in df_csv.columns else "fusion_ppl"
                reason_col = "dyn_reason" if "dyn_reason" in df_csv.columns else "fusion_reason"
                
                df_csv[score_col] = df_csv[score_col].replace(0, np.nan)
                mean_score = df_csv[score_col].mean()
                mean_ppl = df_csv[ppl_col].mean()
                max_ppl = df_csv[ppl_col].max()
                
                if reason_col in df_csv.columns:
                    coherence_rate = df_csv[reason_col].str.contains("Coherence: Yes", case=False, na=False).mean()
                else:
                    coherence_rate = 1.0
                
                records.append({
                    "val":                  val,
                    "dyn_score":            mean_score,
                    "dyn_ppl":              mean_ppl,
                    "dyn_coherence_rate":   coherence_rate,
                    "dyn_max_ppl":          max_ppl,
                })
            except Exception:
                pass
    return pd.DataFrame(records)

def load_all_methods(results_dir: Path, trait: str):
    data_dict = {}
    for display_name, loader_key, _ in METHODS:
        data_dict[loader_key] = load_summary(results_dir, trait, loader_key)
    return data_dict

def build_pivot(method_data_dict):
    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}
    coherence_rows = {v: {} for v in VALS}
    max_ppl_rows = {v: {} for v in VALS}

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

    p_score = pd.DataFrame.from_dict(score_rows, orient="index").reindex(VALS)
    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index").reindex(VALS)
    p_coherence = pd.DataFrame.from_dict(coherence_rows, orient="index").reindex(VALS)
    p_max_ppl = pd.DataFrame.from_dict(max_ppl_rows, orient="index").reindex(VALS)

    cols = [m[0] for m in METHODS if m[0] in p_score.columns]
    p_score = p_score[cols]
    p_ppl = p_ppl[cols]
    p_coherence = p_coherence[cols]
    p_max_ppl = p_max_ppl[cols]

    return p_score, p_ppl, p_coherence, p_max_ppl

def highlight_safe_cells(ax, p_ppl, p_coherence, p_max_ppl,
                         ppl_threshold=25.0, coherence_threshold=0.8,
                         max_ppl_threshold=25.0):
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

def plot_trait(trait, method_data_dict, out_dir, artifact_dir, title_prefix):
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_score, p_ppl, p_coherence, p_max_ppl = build_pivot(method_data_dict)
    
    n_methods = len(p_score.columns) if not p_score.empty else 1
    fig_w = max(10, n_methods * 1.5 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 13))

    configs = [
        (axes[0], p_score, f"Score [{trait.capitalize()}]",
         "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   f"PPL   [{trait.capitalize()}]",
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
        highlight_safe_cells(ax_obj, p_ppl, p_coherence, p_max_ppl)
        ax_obj.set_title(
            f"{title} (Black Border: Strict Safety Criteria [Max PPL <= 25.0])",
            fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("DLS Method", fontsize=10)
        ax_obj.set_ylabel("Steering Intensity (Alpha)", fontsize=10)

    plt.suptitle(
        f"{title_prefix} DLS 8-Method Comparison: {trait.capitalize()}",
        fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    file_name = f"heatmap_dyn_{trait}.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / f"{title_prefix.lower()}_heatmap_dyn_{trait}.png"
        shutil.copy(out_path, dest)

def plot_summary(all_method_data, out_dir, artifact_dir, title_prefix):
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    score_acc     = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    ppl_acc       = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    coherence_acc = {k: {v: [] for v in VALS} for _, k, _ in METHODS}
    max_ppl_acc   = {k: {v: [] for v in VALS} for _, k, _ in METHODS}

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

    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}
    coherence_rows = {v: {} for v in VALS}
    max_ppl_rows   = {v: {} for v in VALS}
    
    for display_name, loader_key, _ in METHODS:
        for val in VALS:
            scores = score_acc[loader_key][val]
            ppls   = ppl_acc[loader_key][val]
            coherences = coherence_acc[loader_key][val]
            max_ppls   = max_ppl_acc[loader_key][val]
            
            if scores:
                score_rows[val][display_name] = np.mean(scores)
                ppl_rows[val][display_name]   = np.mean(ppls)
                coherence_rows[val][display_name] = np.mean(coherences)
                max_ppl_rows[val][display_name]   = np.mean(max_ppls)

    p_score = pd.DataFrame.from_dict(score_rows, orient="index").reindex(VALS)
    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index").reindex(VALS)
    p_coherence = pd.DataFrame.from_dict(coherence_rows, orient="index").reindex(VALS)
    p_max_ppl = pd.DataFrame.from_dict(max_ppl_rows, orient="index").reindex(VALS)

    cols = [m[0] for m in METHODS if m[0] in p_score.columns]
    p_score = p_score[cols]
    p_ppl = p_ppl[cols]
    p_coherence = p_coherence[cols]
    p_max_ppl = p_max_ppl[cols]

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
        highlight_safe_cells(ax_obj, p_ppl, p_coherence, p_max_ppl)
        ax_obj.set_title(
            f"{title} (Black Border: Strict Safety Criteria [Max PPL <= 25.0])",
            fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("DLS Method", fontsize=10)
        ax_obj.set_ylabel("Steering Intensity (Alpha)", fontsize=10)

    plt.suptitle(
        f"{title_prefix} DLS 8-Method Comparison Summary (All Traits Avg)",
        fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    file_name = "summary_dyn_all_traits.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / f"{title_prefix.lower()}_summary_dyn_all_traits.png"
        shutil.copy(out_path, dest)

def get_max_safe_score(results_dir: Path, trait: str, method: str, max_ppl_threshold=25.0) -> tuple[float, float, float, float]:
    best_score = 0.0
    best_alpha = np.nan
    best_ppl = np.nan
    best_coherence = np.nan
    
    trait_dir = results_dir / trait
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                score_col = "dyn_score" if "dyn_score" in df.columns else df.columns[2]
                ppl_col = "dyn_ppl" if "dyn_ppl" in df.columns else "fusion_ppl"
                reason_col = "dyn_reason" if "dyn_reason" in df.columns else "fusion_reason"
                
                df[score_col] = df[score_col].replace(0, np.nan)
                mean_score = df[score_col].mean()
                mean_ppl = df[ppl_col].mean()
                max_ppl = df[ppl_col].max()
                
                if reason_col in df.columns:
                    coherence_rate = df[reason_col].str.contains("Coherence: Yes", case=False, na=False).mean()
                else:
                    coherence_rate = 1.0
                
                # Strict Safety check (using max_ppl_threshold)
                is_safe = (
                    mean_ppl <= 25.0 and
                    coherence_rate >= 0.8 and
                    max_ppl <= max_ppl_threshold
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

def plot_max_safe_bar(results_dir: Path, out_dir: Path, artifact_dir: Path, title_prefix: str, max_ppl_threshold=25.0):
    data = []
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(results_dir, trait)
        method_results = {}
        for display_name, loader_key, _ in METHODS:
            score, alpha, ppl, coherence = get_max_safe_score(results_dir, trait, loader_key, max_ppl_threshold)
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
        valid_scores = [d[display_name][0] for d in data if d[display_name][0] > 0.0]
        avg_score = np.mean(valid_scores) if valid_scores else 0.0
        avg_results[display_name] = (avg_score, np.nan, np.nan, np.nan)
        
    data.append({
        "trait": "Average",
        "Unsteered Baseline": (avg_ub, np.nan, np.nan, np.nan),
        **avg_results
    })

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
    
    rects_list.append(ax.bar(x + (offset_start * width), [d["Unsteered Baseline"][0] for d in data], width, label="Unsteered Baseline", color=colors["Unsteered Baseline"], zorder=3))
    labels_list.append("Unsteered Baseline")
    
    for i, (display_name, _, _) in enumerate(METHODS):
        rects_list.append(ax.bar(x + ((offset_start + 1 + i) * width), [d[display_name][0] for d in data], width, label=display_name, color=colors[display_name], zorder=3))
        labels_list.append(display_name)
        
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    
    title_text = f"{title_prefix} DLS — Maximum Safe Steering Score Comparison (Max PPL <= {max_ppl_threshold})"
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
    
    file_name = f"max_safe_score_compare.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest_path = artifact_dir / f"{title_prefix.lower()}_max_safe_score_compare.png"
        shutil.copy(out_path, dest_path)

def generate_overall_interval_comparison(
    fixed_dir: Path, int1_dir: Path, int4_dir: Path, int8_dir: Path,
    out_dir: Path, artifact_dir: Path, max_ppl_threshold=25.0
):
    print("Generating overall comparison of update intervals...")
    comparison_records = []
    
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(int1_dir, trait)
        
        # 1. Best Fixed-Layer DLS
        best_fixed_score = 0.0
        best_fixed_method = "None"
        best_fixed_alpha = np.nan
        for _, m_key, _ in METHODS:
            score, alpha, _, _ = get_max_safe_score(fixed_dir, trait, m_key, max_ppl_threshold)
            if score > best_fixed_score:
                best_fixed_score = score
                best_fixed_method = m_key
                best_fixed_alpha = alpha
                
        # 2. Best Interval = 1 DLS
        best_int1_score = 0.0
        best_int1_method = "None"
        best_int1_alpha = np.nan
        for _, m_key, _ in METHODS:
            score, alpha, _, _ = get_max_safe_score(int1_dir, trait, m_key, max_ppl_threshold)
            if score > best_int1_score:
                best_int1_score = score
                best_int1_method = m_key
                best_int1_alpha = alpha
                
        # 3. Best Interval = 4 DLS
        best_int4_score = 0.0
        best_int4_method = "None"
        best_int4_alpha = np.nan
        for _, m_key, _ in METHODS:
            score, alpha, _, _ = get_max_safe_score(int4_dir, trait, m_key, max_ppl_threshold)
            if score > best_int4_score:
                best_int4_score = score
                best_int4_method = m_key
                best_int4_alpha = alpha
                
        # 4. Best Interval = 8 DLS
        best_int8_score = 0.0
        best_int8_method = "None"
        best_int8_alpha = np.nan
        for _, m_key, _ in METHODS:
            score, alpha, _, _ = get_max_safe_score(int8_dir, trait, m_key, max_ppl_threshold)
            if score > best_int8_score:
                best_int8_score = score
                best_int8_method = m_key
                best_int8_alpha = alpha
                
        comparison_records.append({
            "Trait":                TRAIT_LABELS[trait],
            "Unsteered Baseline":   ub_score,
            "Fixed-Layer DLS":      best_fixed_score,
            "Fixed Alpha":          best_fixed_alpha,
            "Fixed Method":         best_fixed_method,
            "Interval 1 (Every)":   best_int1_score,
            "Int 1 Alpha":          best_int1_alpha,
            "Int 1 Method":         best_int1_method,
            "Interval 4 (Block)":   best_int4_score,
            "Int 4 Alpha":          best_int4_alpha,
            "Int 4 Method":         best_int4_method,
            "Interval 8 (Block)":   best_int8_score,
            "Int 8 Alpha":          best_int8_alpha,
            "Int 8 Method":         best_int8_method,
        })
        
    # Calculate Average
    avg_rec = {
        "Trait":                "Average",
        "Unsteered Baseline":   np.mean([r["Unsteered Baseline"] for r in comparison_records]),
        "Fixed-Layer DLS":      np.mean([r["Fixed-Layer DLS"] for r in comparison_records]),
        "Fixed Alpha":          np.nan,
        "Fixed Method":         "",
        "Interval 1 (Every)":   np.mean([r["Interval 1 (Every)"] for r in comparison_records]),
        "Int 1 Alpha":          np.nan,
        "Int 1 Method":         "",
        "Interval 4 (Block)":   np.mean([r["Interval 4 (Block)"] for r in comparison_records]),
        "Int 4 Alpha":          np.nan,
        "Int 4 Method":         "",
        "Interval 8 (Block)":   np.mean([r["Interval 8 (Block)"] for r in comparison_records]),
        "Int 8 Alpha":          np.nan,
        "Int 8 Method":         "",
    }
    comparison_records.append(avg_rec)
    df_comp = pd.DataFrame(comparison_records)
    
    # Save comparison table as Markdown and CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "interval_comparison_table.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to: {csv_path}")
    
    # Generate grouped bar chart comparing the 4 settings
    plt.close("all")
    categories = df_comp["Trait"].tolist()
    x = np.arange(len(categories))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    rects1 = ax.bar(x - 1.5*width, df_comp["Fixed-Layer DLS"], width, label="Fixed-Layer DLS (Optimal)", color="#7f8c8d", zorder=3)
    rects2 = ax.bar(x - 0.5*width, df_comp["Interval 1 (Every)"], width, label="Gen-Time DLS (Interval=1)", color="#e74c3c", zorder=3)
    rects3 = ax.bar(x + 0.5*width, df_comp["Interval 4 (Block)"], width, label="Gen-Time DLS (Interval=4)", color="#3498db", zorder=3)
    rects4 = ax.bar(x + 1.5*width, df_comp["Interval 8 (Block)"], width, label="Gen-Time DLS (Interval=8)", color="#2ecc71", zorder=3)
    
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    
    # Add unsteered baseline as dotted horizontal lines for each trait
    for idx, trait_name in enumerate(categories):
        ub = df_comp.loc[idx, "Unsteered Baseline"]
        ax.plot([idx - 2*width, idx + 2*width], [ub, ub], color="#95a5a6", linestyle=":", linewidth=1.5, zorder=4)
        
    ax.set_title(f"Dynamic Steering Interval Comparison: Fixed vs Interval 1, 4, 8\n(Safety Constraints: Mean PPL <= 25.0, Coherence >= 80%, Max PPL <= {max_ppl_threshold})", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Maximum Safe Steering Score", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color="#bbbbbb", zorder=0)
    
    # Annotate bar scores
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            if h > 0.0 and not np.isnan(h):
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=8, fontweight="bold")
                
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig_name = "interval_comparison_chart.png"
    fig_path = out_dir / fig_name
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart to: {fig_path}")
    
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, artifact_dir / fig_name)
        
    return df_comp

def main():
    ap = argparse.ArgumentParser(description="Analyze and plot update_interval sweeps.")
    ap.add_argument("--fixed_dir", default="exp_steering_dyn_layer_raw/results")
    ap.add_argument("--int1_dir", default="exp_steering_dyn_gen_time_raw/results")
    ap.add_argument("--interval_dir", default="exp_steering_dyn_gen_time_interval_raw")
    ap.add_argument("--out_dir", default="exp_steering_dyn_gen_time_interval_raw/figures")
    ap.add_argument("--artifact_dir", default="/home/s2550009/.gemini/antigravity-ide/brain/316d92fc-a09f-45ab-a84d-a1a4060ccdb9/images")
    ap.add_argument("--max_ppl", type=float, default=25.0, help="Maximum PPL for strict safety filter")
    args = ap.parse_args()

    fixed_dir = Path(args.fixed_dir)
    int1_dir = Path(args.int1_dir)
    interval_dir = Path(args.interval_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Interval 4 results
    print("\n=== Processing Interval 4 ===")
    int4_results_dir = interval_dir / "results_interval4"
    if int4_results_dir.exists():
        all_method_data_int4 = []
        for trait in TRAITS:
            method_data = load_all_methods(int4_results_dir, trait)
            all_method_data_int4.append(method_data)
            plot_trait(trait, method_data, out_dir / "interval4" / trait, artifact_dir, "Int4")
        plot_summary(all_method_data_int4, out_dir / "interval4", artifact_dir, "Int4")
        plot_max_safe_bar(int4_results_dir, out_dir / "interval4", artifact_dir, "Int4", max_ppl_threshold=args.max_ppl)

    # 2. Process Interval 8 results
    print("\n=== Processing Interval 8 ===")
    int8_results_dir = interval_dir / "results_interval8"
    if int8_results_dir.exists():
        all_method_data_int8 = []
        for trait in TRAITS:
            method_data = load_all_methods(int8_results_dir, trait)
            all_method_data_int8.append(method_data)
            plot_trait(trait, method_data, out_dir / "interval8" / trait, artifact_dir, "Int8")
        plot_summary(all_method_data_int8, out_dir / "interval8", artifact_dir, "Int8")
        plot_max_safe_bar(int8_results_dir, out_dir / "interval8", artifact_dir, "Int8", max_ppl_threshold=args.max_ppl)

    # 3. Overall Interval Comparison Chart and Table
    print("\n=== Generating Interval Comparison ===")
    df_comp = generate_overall_interval_comparison(
        fixed_dir, int1_dir, int4_results_dir, int8_results_dir,
        out_dir, artifact_dir, max_ppl_threshold=args.max_ppl
    )
    
    print("\n--- Interval Comparison Summary Table ---")
    print(df_comp[["Trait", "Unsteered Baseline", "Fixed-Layer DLS", "Interval 1 (Every)", "Interval 4 (Block)", "Interval 8 (Block)"]].to_markdown(index=False))

if __name__ == "__main__":
    main()
