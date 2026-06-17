#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/04_dyn_layer/109_plot_dyn_layer_norm_raw.py
#
# Generates heatmaps (Score and PPL) comparing different DLS methods:
#   - DLS_cos_only
#   - DLS_rank_only
#   - DLS_logit_diff
# Rows = alpha (Val) values; Columns = methods.
#
# Outputs:
#   - {out_dir}/{trait}/heatmap_dyn_{trait}.png
#   - {out_dir}/summary_dyn_all_traits.png
#

import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

METHODS = [
    ("DLS_cos_only",    "coral",      "cos_only"),
    ("DLS_rank_only",   "teal",       "rank_only"),
    ("DLS_logit_diff",  "navy",       "logit_diff"),
]


def load_summary(results_dir: Path, axis: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = results_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                records.append({
                    "val":       val,
                    "dyn_score": df["dyn_score"].mean(),
                    "dyn_ppl":   df["dyn_ppl"].mean(),
                })
            except Exception:
                pass
    return pd.DataFrame(records)


def load_all_methods(results_dir: Path, axis: str):
    data_dict = {}
    for display_name, color, loader_key in METHODS:
        data_dict[loader_key] = load_summary(results_dir, axis, loader_key)
    return data_dict


def build_pivot(method_data_dict):
    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}

    for display_name, color, loader_key in METHODS:
        df = method_data_dict.get(loader_key, pd.DataFrame())
        if df.empty:
            continue
        idx = df.set_index("val")
        for val in VALS:
            if val in idx.index:
                score_rows[val][display_name] = idx.loc[val, "dyn_score"]
                ppl_rows[val][display_name]   = idx.loc[val, "dyn_ppl"]

    p_score = pd.DataFrame.from_dict(score_rows, orient="index")
    p_score.index.name = "val"
    p_score = p_score.reindex(VALS)
    p_score.columns = [m[0] for m in METHODS if m[0] in p_score.columns]

    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index")
    p_ppl.index.name = "val"
    p_ppl = p_ppl.reindex(VALS)
    p_ppl.columns = [m[0] for m in METHODS if m[0] in p_ppl.columns]

    return p_score, p_ppl


def highlight_safe_cells(ax, p_ppl, threshold=25.0):
    if p_ppl is None or p_ppl.empty:
        return
    for i in range(len(p_ppl.index)):
        for j in range(len(p_ppl.columns)):
            val = p_ppl.iloc[i, j]
            if not np.isnan(val) and val <= threshold:
                rect = Rectangle((j, i), 1, 1, fill=False,
                                  edgecolor="black", lw=2.5, clip_on=False)
                ax.add_patch(rect)


def draw_separators(ax, p_data):
    cols = list(p_data.columns)
    for display_name, color, _ in METHODS:
        if display_name in cols:
            ax.axvline(x=cols.index(display_name), color=color, linewidth=3.0)


def plot_trait(axis, method_data_dict, out_dir, artifact_dir, title_prefix):
    print(f"\n[{axis}] plotting DLS comparison heatmap...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_score, p_ppl = build_pivot(method_data_dict)
    
    n_methods = len(p_score.columns) if not p_score.empty else 1
    fig_w = max(8, n_methods * 2.0 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 12))

    configs = [
        (axes[0], p_score, p_ppl, f"Score [{axis.capitalize()}]",
         "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   p_ppl, f"PPL   [{axis.capitalize()}]",
         "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        if p_data.empty:
            ax_obj.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_obj.transAxes, fontsize=14)
            continue
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.8, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 10})
        draw_separators(ax_obj, p_data)
        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title} (Black Border: PPL ≤ 25.0)",
            fontsize=11, fontweight="bold")
        ax_obj.set_xlabel("Method", fontsize=10)
        ax_obj.set_ylabel("Val (Steering Intensity / Alpha)", fontsize=10)

    plt.suptitle(
        f"{title_prefix} DLS Comparison: {axis.capitalize()}",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    file_name = f"heatmap_dyn_{axis}.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / f"{title_prefix.lower()}_{file_name}"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")


def plot_summary(all_method_data, out_dir, artifact_dir, title_prefix):
    print(f"\n[Summary] plotting DLS comparison summary heatmap (all traits avg)...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    score_acc = {k: {v: [] for v in VALS} for _, _, k in METHODS}
    ppl_acc   = {k: {v: [] for v in VALS} for _, _, k in METHODS}

    for method_data_dict in all_method_data:
        for _, _, loader_key in METHODS:
            df = method_data_dict.get(loader_key, pd.DataFrame())
            if df.empty:
                continue
            idx = df.set_index("val")
            for val in VALS:
                if val in idx.index:
                    score_acc[loader_key][val].append(idx.loc[val, "dyn_score"])
                    ppl_acc[loader_key][val].append(idx.loc[val, "dyn_ppl"])

    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}
    for display_name, _, loader_key in METHODS:
        for val in VALS:
            scores = score_acc[loader_key][val]
            ppls   = ppl_acc[loader_key][val]
            if scores:
                score_rows[val][display_name] = np.mean(scores)
                ppl_rows[val][display_name]   = np.mean(ppls)

    p_score = pd.DataFrame.from_dict(score_rows, orient="index")
    p_score.index.name = "val"
    p_score = p_score.reindex(VALS)

    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index")
    p_ppl.index.name = "val"
    p_ppl = p_ppl.reindex(VALS)

    n_methods = len(p_score.columns) if not p_score.empty else 1
    fig_w = max(8, n_methods * 2.0 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 12))

    configs = [
        (axes[0], p_score, p_ppl, "Score (All Traits Avg)",
         "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   p_ppl, "PPL   (All Traits Avg)",
         "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        if p_data.empty:
            ax_obj.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_obj.transAxes, fontsize=14)
            continue
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.8, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 11})
        draw_separators(ax_obj, p_data)
        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title} (Black Border: PPL ≤ 25.0)",
            fontsize=11, fontweight="bold")
        ax_obj.set_xlabel("Method", fontsize=10)
        ax_obj.set_ylabel("Val (Steering Intensity / Alpha)", fontsize=10)

    plt.suptitle(
        f"{title_prefix} DLS Comparison Summary (All Traits Avg)",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    file_name = "summary_dyn_all_traits.png"
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / f"{title_prefix.lower()}_{file_name}"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")


def main():
    ap = argparse.ArgumentParser(description="Plot dynamic steering method comparison heatmaps.")
    ap.add_argument("--results_dir", required=True, help="Path to results directory")
    ap.add_argument("--out_dir", required=True, help="Output folder for figures")
    ap.add_argument("--artifact_dir", default=None, help="Folder to copy results for conversation viewing")
    ap.add_argument("--title_prefix", required=True, help="Title prefix (e.g., Norm or Raw)")
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
    print(f"\nHeatmap generation for {title_prefix} finished.")


if __name__ == "__main__":
    main()
