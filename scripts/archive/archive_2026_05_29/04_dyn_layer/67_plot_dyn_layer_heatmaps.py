#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 67_plot_dyn_layer_heatmaps.py
#
# Generates heatmaps (Score and PPL) comparing:
#   - Constant & Adaptive background sweeps
#   - Unconstrained DLS (Bhandari, Proposed)
#   - Constrained DLS (Bhandari, Proposed)
#   - Z-score Normalized DLS (Bhandari, Proposed)
#
# Outputs:
#   - exp_steering_dyn_layer_zscore/figures/{trait}/heatmap_{trait}_unified.png
#   - exp_steering_dyn_layer_zscore/figures/summary_all_traits.png

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
        df = pd.read_csv(csv_path)
        records.append({
            "val":       val,
            "dyn_score": df["dyn_score"].mean(),
            "dyn_ppl":   df["dyn_ppl"].mean(),
        })
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


def append_dls_cols(pivot_df, score_col, logit_df, anti_df, logit_cns_df, anti_cns_df, logit_zsc_df, anti_zsc_df, logit_czs_df, anti_czs_df):
    result = pivot_df.copy()
    if not logit_df.empty and score_col in logit_df.columns:
        result["Bhandari"] = logit_df.set_index("val")[score_col]
    if not anti_df.empty and score_col in anti_df.columns:
        result["Proposed"] = anti_df.set_index("val")[score_col]
    if not logit_cns_df.empty and score_col in logit_cns_df.columns:
        result["Bhandari_Cns"] = logit_cns_df.set_index("val")[score_col]
    if not anti_cns_df.empty and score_col in anti_cns_df.columns:
        result["Proposed_Cns"] = anti_cns_df.set_index("val")[score_col]
    if not logit_zsc_df.empty and score_col in logit_zsc_df.columns:
        result["Bhandari_Zsc"] = logit_zsc_df.set_index("val")[score_col]
    if not anti_zsc_df.empty and score_col in anti_zsc_df.columns:
        result["Proposed_Zsc"] = anti_zsc_df.set_index("val")[score_col]
    if not logit_czs_df.empty and score_col in logit_czs_df.columns:
        result["Bhandari_CnsZsc"] = logit_czs_df.set_index("val")[score_col]
    if not anti_czs_df.empty and score_col in anti_czs_df.columns:
        result["Proposed_CnsZsc"] = anti_czs_df.set_index("val")[score_col]
    return result


def plot_axis(df: pd.DataFrame, axis: str, out_dir: Path, dyn_dir: Path, cns_dir: Path, zsc_dir: Path, czs_dir: Path):
    print(f"\n[{axis}] processing unified heatmap...")
    if df.empty:
        print("  No data, skipping.")
        return

    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DLS results
    logit_df = load_dyn_summary(dyn_dir, axis, "logit_diff")
    anti_df  = load_dyn_summary(dyn_dir, axis, "anti_alignment")
    logit_cns_df = load_dyn_summary(cns_dir, axis, "logit_diff")
    anti_cns_df  = load_dyn_summary(cns_dir, axis, "anti_alignment")
    logit_zsc_df = load_dyn_summary(zsc_dir, axis, "logit_diff")
    anti_zsc_df  = load_dyn_summary(zsc_dir, axis, "anti_alignment")
    logit_czs_df = load_dyn_summary(czs_dir, axis, "logit_diff")
    anti_czs_df  = load_dyn_summary(czs_dir, axis, "anti_alignment")

    # Pivot tables
    def pivot(col):
        return df.pivot(index="val", columns="layer", values=col)

    p_c_score = pivot("const_score")
    p_a_score = pivot("adapt_score")
    p_c_ppl   = pivot("const_ppl")
    p_a_ppl   = pivot("adapt_ppl")

    p_c_score_dls = append_dls_cols(p_c_score, "dyn_score", logit_df, anti_df, logit_cns_df, anti_cns_df, logit_zsc_df, anti_zsc_df, logit_czs_df, anti_czs_df)
    p_a_score_dls = append_dls_cols(p_a_score, "dyn_score", logit_df, anti_df, logit_cns_df, anti_cns_df, logit_zsc_df, anti_zsc_df, logit_czs_df, anti_czs_df)
    p_c_ppl_dls   = append_dls_cols(p_c_ppl,   "dyn_ppl",   logit_df, anti_df, logit_cns_df, anti_cns_df, logit_zsc_df, anti_zsc_df, logit_czs_df, anti_czs_df)
    p_a_ppl_dls   = append_dls_cols(p_a_ppl,   "dyn_ppl",   logit_df, anti_df, logit_cns_df, anti_cns_df, logit_zsc_df, anti_zsc_df, logit_czs_df, anti_czs_df)

    # Plot Unified Heatmap (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(24, 11))
    configs = [
        (axes[0, 0], p_c_score_dls, p_c_ppl_dls, "Constant — Score", "YlGn", 1, 5, ".2f"),
        (axes[0, 1], p_a_score_dls, p_a_ppl_dls, "Adaptive — Score", "YlGn", 1, 5, ".2f"),
        (axes[1, 0], p_c_ppl_dls,   p_c_ppl_dls, "Constant — PPL",   "RdYlGn_r", 1, 100, ".1f"),
        (axes[1, 1], p_a_ppl_dls,   p_a_ppl_dls, "Adaptive — PPL",   "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 8})

        # Draw vertical lines to separate DLS variants
        cols = list(p_data.columns)
        separators = [
            ("Bhandari", "navy"), ("Proposed", "darkred"),
            ("Bhandari_Cns", "dodgerblue"), ("Proposed_Cns", "salmon"),
            ("Bhandari_Zsc", "blue"), ("Proposed_Zsc", "magenta"),
            ("Bhandari_CnsZsc", "teal"), ("Proposed_CnsZsc", "purple")
        ]
        for col_name, color in separators:
            if col_name in cols:
                ax_obj.axvline(x=cols.index(col_name), color=color, linewidth=2.5)

        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title} [{axis.capitalize()}] (Border: PPL<=25, Navy/Red: Uncon, Blue/Pink: Cns, Violet/Magenta: Zsc, Teal/Purple: CnsZsc)",
            fontsize=9, fontweight="bold")
        ax_obj.set_xlabel("Layer / DLS Variants (rightmost)")
        ax_obj.set_ylabel("Val (Steering Intensity)")

    plt.suptitle(f"Layer-Sweep & DLS Unified Results: {axis.capitalize()}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = out_dir / f"heatmap_{axis}_unified.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved unified heatmap: {out_path}")


def make_summary_heatmaps(all_df: pd.DataFrame,
                          logit_all_df: pd.DataFrame, anti_all_df: pd.DataFrame,
                          logit_cns_all_df: pd.DataFrame, anti_cns_all_df: pd.DataFrame,
                          logit_zsc_all_df: pd.DataFrame, anti_zsc_all_df: pd.DataFrame,
                          logit_czs_all_df: pd.DataFrame, anti_czs_all_df: pd.DataFrame,
                          out_dir: Path):
    if all_df.empty:
        return
    avg = all_df.groupby(["layer", "val"])[
        ["const_score", "adapt_score", "const_ppl", "adapt_ppl"]
    ].mean().reset_index()

    def avg_dyn(df):
        if df is None or df.empty:
            return pd.DataFrame()
        return df.groupby("val")[["dyn_score", "dyn_ppl"]].mean().reset_index()

    logit_avg     = avg_dyn(logit_all_df)
    anti_avg      = avg_dyn(anti_all_df)
    logit_cns_avg = avg_dyn(logit_cns_all_df)
    anti_cns_avg  = avg_dyn(anti_cns_all_df)
    logit_zsc_avg = avg_dyn(logit_zsc_all_df)
    anti_zsc_avg  = avg_dyn(anti_zsc_all_df)
    logit_czs_avg = avg_dyn(logit_czs_all_df)
    anti_czs_avg  = avg_dyn(anti_czs_all_df)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(24, 11))
    configs = [
        ("const_score", "Constant — Score (All traits avg)", "YlGn",     1,   5, axes[0, 0]),
        ("adapt_score", "Adaptive — Score (All traits avg)",  "YlGn",     1,   5, axes[0, 1]),
        ("const_ppl",   "Constant — PPL (All traits avg)",    "RdYlGn_r", 1, 100, axes[1, 0]),
        ("adapt_ppl",   "Adaptive — PPL (All traits avg)",    "RdYlGn_r", 1, 100, axes[1, 1]),
    ]

    for col, title, cmap, vmin, vmax, ax_obj in configs:
        p = avg.pivot(index="val", columns="layer", values=col)
        fmt = ".2f" if "score" in col else ".1f"
        ppl_col = col.replace("score", "ppl")
        p_ppl = avg.pivot(index="val", columns="layer", values=ppl_col) if "score" in col else p.copy()

        dls_score_col = "dyn_score" if "score" in col else "dyn_ppl"

        def add_col(p, p_ppl, avg_df, col_name):
            if avg_df.empty or dls_score_col not in avg_df.columns:
                return
            idx = avg_df.set_index("val")
            p[col_name] = idx[dls_score_col]
            if "score" in col:
                p_ppl[col_name] = idx["dyn_ppl"]
            else:
                p_ppl[col_name] = idx[dls_score_col]

        add_col(p, p_ppl, logit_avg,     "Bhandari")
        add_col(p, p_ppl, anti_avg,      "Proposed")
        add_col(p, p_ppl, logit_cns_avg, "Bhandari_Cns")
        add_col(p, p_ppl, anti_cns_avg,  "Proposed_Cns")
        add_col(p, p_ppl, logit_zsc_avg, "Bhandari_Zsc")
        add_col(p, p_ppl, anti_zsc_avg,  "Proposed_Zsc")
        add_col(p, p_ppl, logit_czs_avg, "Bhandari_CnsZsc")
        add_col(p, p_ppl, anti_czs_avg,  "Proposed_CnsZsc")

        sns.heatmap(p, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 8})

        cols = list(p.columns)
        separators = [
            ("Bhandari", "navy"), ("Proposed", "darkred"),
            ("Bhandari_Cns", "dodgerblue"), ("Proposed_Cns", "salmon"),
            ("Bhandari_Zsc", "blue"), ("Proposed_Zsc", "magenta"),
            ("Bhandari_CnsZsc", "teal"), ("Proposed_CnsZsc", "purple")
        ]
        for col_name, color in separators:
            if col_name in cols:
                ax_obj.axvline(x=cols.index(col_name), color=color, linewidth=2.5)

        highlight_safe_cells(ax_obj, p_ppl, threshold=25.0)
        ax_obj.set_title(
            title + " (Border: PPL<=25, Navy/Red: Uncon, Blue/Pink: Cns, Violet/Magenta: Zsc, Teal/Purple: CnsZsc)",
            fontsize=9, fontweight="bold")
        ax_obj.set_xlabel("Layer / DLS Variants (rightmost)")
        ax_obj.set_ylabel("Val (Steering Intensity)")

    plt.suptitle("Layer-Sweep & DLS Summary (All Traits Average)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    summary_path = out_dir / "summary_all_traits.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved summary: {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="exp_steering_layer_analysis/results")
    ap.add_argument("--dyn_dir",   default="exp_steering_dyn_layer/results")
    ap.add_argument("--cns_dir",   default="exp_steering_dyn_layer_constrained/results")
    ap.add_argument("--zsc_dir",   default="exp_steering_dyn_layer_zscore/results")
    ap.add_argument("--czs_dir",   default="exp_steering_dyn_layer_CnsZsc/results")
    ap.add_argument("--out_dir",   default="exp_steering_dyn_layer_CnsZsc/figures")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    dyn_dir   = Path(args.dyn_dir)
    cns_dir   = Path(args.cns_dir)
    zsc_dir   = Path(args.zsc_dir)
    czs_dir   = Path(args.czs_dir)
    out_dir   = Path(args.out_dir)

    all_dfs = []
    all_logit_dfs = []
    all_anti_dfs = []
    all_logit_cns_dfs = []
    all_anti_cns_dfs  = []
    all_logit_zsc_dfs = []
    all_anti_zsc_dfs  = []
    all_logit_czs_dfs = []
    all_anti_czs_dfs  = []

    for axis in TRAITS:
        df = load_summary(input_dir, axis)
        if not df.empty:
            all_dfs.append(df)

        logit_df = load_dyn_summary(dyn_dir, axis, "logit_diff")
        if not logit_df.empty:
            all_logit_dfs.append(logit_df)

        anti_df = load_dyn_summary(dyn_dir, axis, "anti_alignment")
        if not anti_df.empty:
            all_anti_dfs.append(anti_df)

        logit_cns_df = load_dyn_summary(cns_dir, axis, "logit_diff")
        if not logit_cns_df.empty:
            all_logit_cns_dfs.append(logit_cns_df)

        anti_cns_df = load_dyn_summary(cns_dir, axis, "anti_alignment")
        if not anti_cns_df.empty:
            all_anti_cns_dfs.append(anti_cns_df)

        logit_zsc_df = load_dyn_summary(zsc_dir, axis, "logit_diff")
        if not logit_zsc_df.empty:
            all_logit_zsc_dfs.append(logit_zsc_df)

        anti_zsc_df = load_dyn_summary(zsc_dir, axis, "anti_alignment")
        if not anti_zsc_df.empty:
            all_anti_zsc_dfs.append(anti_zsc_df)

        logit_czs_df = load_dyn_summary(czs_dir, axis, "logit_diff")
        if not logit_czs_df.empty:
            all_logit_czs_dfs.append(logit_czs_df)

        anti_czs_df = load_dyn_summary(czs_dir, axis, "anti_alignment")
        if not anti_czs_df.empty:
            all_anti_czs_dfs.append(anti_czs_df)

        plot_axis(df, axis, out_dir / axis, dyn_dir, cns_dir, zsc_dir, czs_dir)

    if len(all_dfs) > 1:
        all_df        = pd.concat(all_dfs,           ignore_index=True)
        logit_all_df  = pd.concat(all_logit_dfs,     ignore_index=True) if all_logit_dfs     else pd.DataFrame()
        anti_all_df   = pd.concat(all_anti_dfs,      ignore_index=True) if all_anti_dfs      else pd.DataFrame()
        logit_cns_all = pd.concat(all_logit_cns_dfs, ignore_index=True) if all_logit_cns_dfs else pd.DataFrame()
        anti_cns_all  = pd.concat(all_anti_cns_dfs,  ignore_index=True) if all_anti_cns_dfs  else pd.DataFrame()
        logit_zsc_all = pd.concat(all_logit_zsc_dfs, ignore_index=True) if all_logit_zsc_dfs else pd.DataFrame()
        anti_zsc_all  = pd.concat(all_anti_zsc_dfs,  ignore_index=True) if all_anti_zsc_dfs  else pd.DataFrame()
        logit_czs_all = pd.concat(all_logit_czs_dfs, ignore_index=True) if all_logit_czs_dfs else pd.DataFrame()
        anti_czs_all  = pd.concat(all_anti_czs_dfs,  ignore_index=True) if all_anti_czs_dfs  else pd.DataFrame()

        make_summary_heatmaps(all_df, logit_all_df, anti_all_df, logit_cns_all, anti_cns_all, logit_zsc_all, anti_zsc_all, logit_czs_all, anti_czs_all, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
