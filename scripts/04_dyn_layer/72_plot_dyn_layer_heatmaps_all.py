#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 72_plot_dyn_layer_heatmaps_all.py
#
# 全層（0〜31）実験結果を反映したヒートマップ可視化スクリプト。
# - レイヤースイープ結果（0, 3, ..., 30層）および全層DLS（Z-score化 logit_diff, Z-score化 anti_alignment, relative anti_alignment）を並べて比較。
# - 各特性ごとの個別ヒートマップおよび全特性サマリーヒートマップを出力。
#

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def load_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{layer}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv"
                if not csv_path.exists():
                    continue
            try:
                df = pd.read_csv(csv_path)
                records.append({
                    "layer": layer,
                    "val":   val,
                    "base_score":  df["base_score"].mean(),
                    "const_score": df["const_score"].mean(),
                    "adapt_score": df["adapt_score"].mean(),
                    "base_ppl":    df["base_ppl"].mean(),
                    "const_ppl":   df["const_ppl"].mean(),
                    "adapt_ppl":   df["adapt_ppl"].mean(),
                })
            except Exception as e:
                print(f"[WARNING] Error reading {csv_path}: {e}")
    return pd.DataFrame(records)

def load_dyn_summary(dyn_dir: Path, axis: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = dyn_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            if not csv_path.exists():
                continue
        try:
            df = pd.read_csv(csv_path)
            records.append({
                "val":       val,
                "dyn_score": df["dyn_score"].mean(),
                "dyn_ppl":   df["dyn_ppl"].mean(),
            })
        except Exception as e:
            print(f"[WARNING] Error reading {csv_path}: {e}")
    return pd.DataFrame(records)

def highlight_safe_cells(ax, p_ppl, threshold=25.0):
    if p_ppl is None or p_ppl.empty:
        return
    for i in range(len(p_ppl.index)):
        for j in range(len(p_ppl.columns)):
            val = p_ppl.iloc[i, j]
            if not np.isnan(val) and val <= threshold:
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2, clip_on=False)
                ax.add_patch(rect)

def append_dls_cols(pivot_df, score_col, logit_df, anti_df, relative_df):
    result = pivot_df.copy()
    if not logit_df.empty and score_col in logit_df.columns:
        result["Bhandari_Zscore_All"] = logit_df.set_index("val")[score_col]
    if not anti_df.empty and score_col in anti_df.columns:
        result["Proposed_Zscore_All"] = anti_df.set_index("val")[score_col]
    if not relative_df.empty and score_col in relative_df.columns:
        result["Proposed_Relative_All"] = relative_df.set_index("val")[score_col]
    return result

def make_empty_pivot(vals):
    return pd.DataFrame(index=pd.Index(vals, name="val"))

def plot_axis(df: pd.DataFrame, axis: str, out_dir: Path, all_layers_dir: Path):
    print(f"\n[{axis}] processing unified heatmap...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load All-Layers DLS results
    logit_df    = load_dyn_summary(all_layers_dir, axis, "logit_diff")
    anti_df     = load_dyn_summary(all_layers_dir, axis, "anti_alignment")
    relative_df = load_dyn_summary(all_layers_dir, axis, "relative_anti_alignment")

    has_layer_data = not df.empty
    has_dls_data   = any(not d.empty for d in [logit_df, anti_df, relative_df])
    
    if not has_layer_data and not has_dls_data:
        print("  No data at all, skipping.")
        return

    if has_layer_data:
        def pivot(col):
            return df.pivot(index="val", columns="layer", values=col)
        p_c_score = pivot("const_score")
        p_a_score = pivot("adapt_score")
        p_c_ppl   = pivot("const_ppl")
        p_a_ppl   = pivot("adapt_ppl")
    else:
        print("  No layer-sweep data; building DLS-only heatmap.")
        all_vals = sorted(set(
            v for d in [logit_df, anti_df, relative_df]
            if not d.empty for v in d["val"].tolist()
        ))
        p_c_score = make_empty_pivot(all_vals)
        p_a_score = make_empty_pivot(all_vals)
        p_c_ppl   = make_empty_pivot(all_vals)
        p_a_ppl   = make_empty_pivot(all_vals)

    p_c_score_dls = append_dls_cols(p_c_score, "dyn_score", logit_df, anti_df, relative_df)
    p_a_score_dls = append_dls_cols(p_a_score, "dyn_score", logit_df, anti_df, relative_df)
    p_c_ppl_dls   = append_dls_cols(p_c_ppl,   "dyn_ppl",   logit_df, anti_df, relative_df)
    p_a_ppl_dls   = append_dls_cols(p_a_ppl,   "dyn_ppl",   logit_df, anti_df, relative_df)

    fig, axes = plt.subplots(2, 2, figsize=(26, 12))
    configs = [
        (axes[0, 0], p_c_score_dls, p_c_ppl_dls, "Constant — Score", "YlGn",     1,   5, ".2f"),
        (axes[0, 1], p_a_score_dls, p_a_ppl_dls, "Adaptive — Score", "YlGn",     1,   5, ".2f"),
        (axes[1, 0], p_c_ppl_dls,   p_c_ppl_dls, "Constant — PPL",   "RdYlGn_r", 1, 100, ".1f"),
        (axes[1, 1], p_a_ppl_dls,   p_a_ppl_dls, "Adaptive — PPL",   "RdYlGn_r", 1, 100, ".1f"),
    ]

    separators = [
        ("Bhandari_Zscore_All", "navy"),
        ("Proposed_Zscore_All", "darkred"),
        ("Proposed_Relative_All", "darkgreen"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        if p_data.empty or p_data.shape[1] == 0:
            ax_obj.set_visible(False)
            continue
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 7})

        cols = list(p_data.columns)
        for col_name, color in separators:
            if col_name in cols:
                ax_obj.axvline(x=cols.index(col_name), color=color, linewidth=2.5)

        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title} [{axis.capitalize()}]"
            f" (Border:PPL<=25 | Navy: Bhandari Z-score | Red: Proposed Z-score | Green: Relative Anti-alignment)",
            fontsize=8, fontweight="bold")
        ax_obj.set_xlabel("Layer / DLS Variants (rightmost)")
        ax_obj.set_ylabel("Val (Steering Intensity)")

    plt.suptitle(f"Layer-Sweep & All-Layer DLS Results: {axis.capitalize()}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = out_dir / f"heatmap_{axis}_unified.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved unified heatmap: {out_path}")

def make_summary_heatmaps(all_df: pd.DataFrame,
                          logit_all_df: pd.DataFrame,
                          anti_all_df: pd.DataFrame,
                          relative_all_df: pd.DataFrame,
                          out_dir: Path):
    all_dyn_dfs = [logit_all_df, anti_all_df, relative_all_df]
    has_layer_data = not all_df.empty
    has_dls_data   = any(not d.empty for d in all_dyn_dfs)
    
    if not has_layer_data and not has_dls_data:
        print("  No data for summary, skipping.")
        return

    def avg_dyn(df):
        if df is None or df.empty:
            return pd.DataFrame()
        return df.groupby("val")[["dyn_score", "dyn_ppl"]].mean().reset_index()

    logit_avg    = avg_dyn(logit_all_df)
    anti_avg     = avg_dyn(anti_all_df)
    relative_avg = avg_dyn(relative_all_df)

    out_dir.mkdir(parents=True, exist_ok=True)

    separators = [
        ("Bhandari_Zscore_All", "navy"),
        ("Proposed_Zscore_All", "darkred"),
        ("Proposed_Relative_All", "darkgreen"),
    ]

    method_dfs = [
        ("Bhandari_Zscore_All",   logit_avg),
        ("Proposed_Zscore_All",   anti_avg),
        ("Proposed_Relative_All", relative_avg),
    ]

    if has_layer_data:
        avg = all_df.groupby(["layer", "val"])[
            ["const_score", "adapt_score", "const_ppl", "adapt_ppl"]
        ].mean().reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(26, 12))
        layer_configs = [
            ("const_score", "Constant — Score (All traits avg)", "YlGn",     1,   5, axes[0, 0]),
            ("adapt_score", "Adaptive — Score (All traits avg)",  "YlGn",     1,   5, axes[0, 1]),
            ("const_ppl",   "Constant — PPL (All traits avg)",    "RdYlGn_r", 1, 100, axes[1, 0]),
            ("adapt_ppl",   "Adaptive — PPL (All traits avg)",    "RdYlGn_r", 1, 100, axes[1, 1]),
        ]
        for col, title, cmap, vmin, vmax, ax_obj in layer_configs:
            p = avg.pivot(index="val", columns="layer", values=col)
            fmt = ".2f" if "score" in col else ".1f"
            ppl_col = col.replace("score", "ppl")
            p_ppl = avg.pivot(index="val", columns="layer", values=ppl_col) if "score" in col else p.copy()
            dls_score_col = "dyn_score" if "score" in col else "dyn_ppl"

            for name, avg_df in method_dfs:
                if avg_df.empty or dls_score_col not in avg_df.columns:
                    continue
                idx = avg_df.set_index("val")
                p[name]     = idx[dls_score_col]
                if "score" in col:
                    p_ppl[name] = idx["dyn_ppl"]
                else:
                    p_ppl[name] = idx[dls_score_col]

            sns.heatmap(p, annot=True, fmt=fmt, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        linewidths=0.4, linecolor="gray",
                        ax=ax_obj, annot_kws={"size": 7})
            cols_list = list(p.columns)
            for col_name, color in separators:
                if col_name in cols_list:
                    ax_obj.axvline(x=cols_list.index(col_name), color=color, linewidth=2.5)
            highlight_safe_cells(ax_obj, p_ppl, threshold=25.0)
            ax_obj.set_title(
                title + " (Border:PPL<=25 | Navy: Bhandari Z | Red: Proposed Z | Green: Relative)",
                fontsize=8, fontweight="bold")
            ax_obj.set_xlabel("Layer / DLS Variants (rightmost)")
            ax_obj.set_ylabel("Val (Steering Intensity)")

        plt.suptitle("Layer-Sweep & All-Layer DLS Summary — All Traits Average",
                     fontsize=14, fontweight="bold", y=1.02)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        configs_dls = [
            ("dyn_score", "Score (All traits avg, all DLS)", "YlGn",     1,   5, axes[0]),
            ("dyn_ppl",   "PPL   (All traits avg, all DLS)", "RdYlGn_r", 1, 100, axes[1]),
        ]
        for dls_col, title, cmap, vmin, vmax, ax_obj in configs_dls:
            fmt = ".2f" if "score" in dls_col else ".1f"
            frames = {}
            ppl_frames = {}
            for name, avg_df in method_dfs:
                if avg_df.empty or dls_col not in avg_df.columns:
                    continue
                idx = avg_df.set_index("val")
                frames[name]     = idx[dls_col]
                ppl_frames[name] = idx["dyn_ppl"]
            if not frames:
                ax_obj.set_visible(False)
                continue
            p     = pd.DataFrame(frames)
            p_ppl = pd.DataFrame(ppl_frames)
            p.index.name = "val"
            sns.heatmap(p, annot=True, fmt=fmt, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        linewidths=0.4, linecolor="gray",
                        ax=ax_obj, annot_kws={"size": 8})
            cols_list = list(p.columns)
            for col_name, color in separators:
                if col_name in cols_list:
                    ax_obj.axvline(x=cols_list.index(col_name), color=color, linewidth=2.5)
            highlight_safe_cells(ax_obj, p_ppl, threshold=25.0)
            ax_obj.set_title(
                title + "\n(Border:PPL<=25 | Navy: Bhandari Z | Red: Proposed Z | Green: Relative)",
                fontsize=8, fontweight="bold")
            ax_obj.set_xlabel("DLS Method")
            ax_obj.set_ylabel("Val (Steering Intensity)")
        plt.suptitle("DLS Summary — All Traits Average",
                     fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    summary_path = out_dir / "summary_all_traits.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved summary: {summary_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",      default="exp_steering_layer_analysis/results")
    ap.add_argument("--all_layers_dir", default="exp_steering_dyn_layer_all_layers/results")
    ap.add_argument("--out_dir",        default="exp_steering_dyn_layer_all_layers/figures")
    args = ap.parse_args()

    input_dir      = Path(args.input_dir)
    all_layers_dir = Path(args.all_layers_dir)
    out_dir        = Path(args.out_dir)

    all_dfs          = []
    all_logit_dfs    = []
    all_anti_dfs     = []
    all_relative_dfs = []

    for axis in TRAITS:
        df = load_summary(input_dir, axis)
        if not df.empty:
            all_dfs.append(df)

        logit_df = load_dyn_summary(all_layers_dir, axis, "logit_diff")
        if not logit_df.empty:
            all_logit_dfs.append(logit_df)

        anti_df = load_dyn_summary(all_layers_dir, axis, "anti_alignment")
        if not anti_df.empty:
            all_anti_dfs.append(anti_df)

        relative_df = load_dyn_summary(all_layers_dir, axis, "relative_anti_alignment")
        if not relative_df.empty:
            all_relative_dfs.append(relative_df)

        plot_axis(df, axis, out_dir / axis, all_layers_dir)

    all_df          = pd.concat(all_dfs,          ignore_index=True) if all_dfs          else pd.DataFrame()
    logit_all_df    = pd.concat(all_logit_dfs,    ignore_index=True) if all_logit_dfs    else pd.DataFrame()
    anti_all_df     = pd.concat(all_anti_dfs,     ignore_index=True) if all_anti_dfs     else pd.DataFrame()
    relative_all_df = pd.concat(all_relative_dfs, ignore_index=True) if all_relative_dfs else pd.DataFrame()

    make_summary_heatmaps(all_df, logit_all_df, anti_all_df, relative_all_df, out_dir)
    print("\nPlotting Done.")

if __name__ == "__main__":
    main()
