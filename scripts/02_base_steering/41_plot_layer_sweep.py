#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 41_plot_layer_sweep.py
#
# Layer-Sweep 実験の結果を可視化する。
#
# 出力:
#   1. ヒートマップ (Layer x Val) - 性格スコア（Constant / Adaptive / DLS）
#   2. ヒートマップ (Layer x Val) - PPL（Constant / Adaptive / DLS）
#   3. スコア-PPLトレードオフプロット (Layer ごとに色付け + DLS)
#
# Usage:
#   python scripts/02_base_steering/41_plot_layer_sweep.py \
#     --input_dir exp_steering_layer_analysis/results \
#     --dyn_dir exp_steering_dyn_layer_compare/results \
#     --out_dir exp_steering_layer_analysis/figures \
#     --axis extraversion

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.5, 1, 2, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40]


def load_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    """指定された特性の全CSVを読み込んで集計DataFrameを返す"""
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            # Note: 40_run_layer_sweep.py now saves with '.0' for integers
            csv_path = trait_dir / f"scores_layer_{layer}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv" # Fallback
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
    """DLS結果CSVを読み込んでval単位の集計DataFrameを返す"""
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
    """PPLが閾値以下のセルを枠線で囲む"""
    if p_ppl is None or p_ppl.empty:
        return
    for i in range(len(p_ppl.index)):
        for j in range(len(p_ppl.columns)):
            val = p_ppl.iloc[i, j]
            if not np.isnan(val) and val <= threshold:
                # Rectangle coordinates: (x, y), width, height
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2, clip_on=False)
                ax.add_patch(rect)


def make_tradeoff_plot(df: pd.DataFrame, axis: str, out_path: Path,
                       logit_df: pd.DataFrame | None = None,
                       anti_df: pd.DataFrame | None = None):
    """Layer を色、Val を大きさにしてスコア vs PPL をプロット。DLS は線とマーカーでオーバーレイ"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    layer_vals = sorted(df["layer"].unique())
    cmap = plt.get_cmap("tab20", len(layer_vals))
    color_map = {l: cmap(i) for i, l in enumerate(layer_vals)}

    for mode, score_col, ppl_col, ax in [
        ("Constant", "const_score", "const_ppl", axes[0]),
        ("Adaptive", "adapt_score", "adapt_ppl", axes[1]),
    ]:
        for layer in layer_vals:
            sub = df[df["layer"] == layer].sort_values("val")
            color = color_map[layer]
            ax.plot(sub[score_col], sub[ppl_col], "-o",
                    color=color, label=f"L{layer}", alpha=0.3, linewidth=1.0)
            ax.scatter(sub[score_col], sub[ppl_col], color=color, s=15, alpha=0.3)
        
        # ベースラインの平均
        base_score = df["base_score"].mean()
        base_ppl  = df["base_ppl"].mean()
        ax.scatter([base_score], [base_ppl], color="black", marker="*", s=250,
                   zorder=5, label="Original Base")

        # Plot Bhandari DLS (Logit Diff)
        if logit_df is not None and not logit_df.empty:
            sub = logit_df.sort_values("val")
            ax.plot(sub["dyn_score"], sub["dyn_ppl"], "-o", color="blue", linewidth=2.5, markersize=8, zorder=6, label="Bhandari (Logit Diff)")
            for _, row in sub.iterrows():
                ax.annotate(f"{row['val']:g}", (row["dyn_score"], row["dyn_ppl"]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="blue", weight='bold')

        # Plot Proposed DLS (Anti Alignment)
        if anti_df is not None and not anti_df.empty:
            sub = anti_df.sort_values("val")
            ax.plot(sub["dyn_score"], sub["dyn_ppl"], "-s", color="red", linewidth=2.5, markersize=8, zorder=7, label="Proposed (Anti-Align)")
            for _, row in sub.iterrows():
                ax.annotate(f"{row['val']:g}", (row["dyn_score"], row["dyn_ppl"]), textcoords="offset points", xytext=(4, -12), fontsize=8, color="red", weight='bold')

        ax.set_title(f"{mode} Steering — {axis.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Personality Score ({axis.capitalize()})", fontsize=11)
        ax.set_ylabel("Perplexity (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_axis(df: pd.DataFrame, axis: str, out_dir: Path, dyn_dir: Path):
    print(f"\n[{axis}] found {len(df)} conditions")
    if df.empty:
        print("  No data, skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DLS results
    logit_df = load_dyn_summary(dyn_dir, axis, "logit_diff")
    anti_df = load_dyn_summary(dyn_dir, axis, "anti_alignment")

    # Pivot tables (index=val, columns=layer)
    def pivot(col):
        return df.pivot(index="val", columns="layer", values=col)

    p_c_score = pivot("const_score")
    p_a_score = pivot("adapt_score")
    p_c_ppl   = pivot("const_ppl")
    p_a_ppl   = pivot("adapt_ppl")

    # Append DLS columns to score/ppl pivots
    def append_dls_cols(pivot_df, score_col, logit_df, anti_df):
        result = pivot_df.copy()
        if not logit_df.empty and score_col in logit_df.columns:
            dls_s = logit_df.set_index("val")[score_col]
            dls_s.name = "Bhandari"
            result["Bhandari"] = dls_s
        if not anti_df.empty and score_col in anti_df.columns:
            dls_s = anti_df.set_index("val")[score_col]
            dls_s.name = "Proposed"
            result["Proposed"] = dls_s
        return result

    p_c_score_dls = append_dls_cols(p_c_score, "dyn_score", logit_df, anti_df)
    p_a_score_dls = append_dls_cols(p_a_score, "dyn_score", logit_df, anti_df)
    p_c_ppl_dls   = append_dls_cols(p_c_ppl,   "dyn_ppl",   logit_df, anti_df)
    p_a_ppl_dls   = append_dls_cols(p_a_ppl,   "dyn_ppl",   logit_df, anti_df)

    # ---- 統合ヒートマップ (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(22, 10))
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
                    ax=ax_obj, annot_kws={"size": 9})

        # DLS 列を視覚的に強調（列境界線を太く）
        if "Bhandari" in p_data.columns:
            idx = list(p_data.columns).index("Bhandari")
            ax_obj.axvline(x=idx, color="navy", linewidth=3)
        if "Proposed" in p_data.columns:
            idx = list(p_data.columns).index("Proposed")
            # Separate the proposed from Bhandari
            ax_obj.axvline(x=idx, color="darkred", linewidth=3)

        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(f"{title} [{axis.capitalize()}] (Border: PPL<=25, Navy: Bhandari, Red: Proposed)",
                         fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer  (rightmost = DLS)")
        ax_obj.set_ylabel("Val")

    plt.suptitle(f"Layer-Sweep Results: {axis.capitalize()}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = out_dir / f"heatmap_{axis}_unified.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved unified heatmap: {out_path}")

    # ---- 3. スコア-PPL トレードオフ ----
    tradeoff_path = out_dir / f"tradeoff_{axis}.png"
    make_tradeoff_plot(df, axis, tradeoff_path, logit_df, anti_df)


def make_summary_heatmaps(all_df: pd.DataFrame, logit_all_df: pd.DataFrame, anti_all_df: pd.DataFrame, out_dir: Path):
    """全特性を平均した Layer x Val ヒートマップ (DLS列付き)"""
    if all_df.empty:
        return
    avg = all_df.groupby(["layer", "val"])[
        ["const_score", "adapt_score", "const_ppl", "adapt_ppl"]
    ].mean().reset_index()

    # DLS も全特性で平均
    logit_avg = pd.DataFrame()
    if not logit_all_df.empty:
        logit_avg = logit_all_df.groupby("val")[["dyn_score", "dyn_ppl"]].mean().reset_index()
    
    anti_avg = pd.DataFrame()
    if not anti_all_df.empty:
        anti_avg = anti_all_df.groupby("val")[["dyn_score", "dyn_ppl"]].mean().reset_index()

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    # レイアウト統一:
    # [0,0]: Const Score  [0,1]: Adapt Score
    # [1,0]: Const PPL    [1,1]: Adapt PPL
    configs = [
        ("const_score", "Constant — Score (All traits avg)", "YlGn", 1, 5, axes[0, 0]),
        ("adapt_score", "Adaptive — Score (All traits avg)",  "YlGn", 1, 5, axes[0, 1]),
        ("const_ppl",   "Constant — PPL (All traits avg)",    "RdYlGn_r", 1, 100, axes[1, 0]),
        ("adapt_ppl",   "Adaptive — PPL (All traits avg)",    "RdYlGn_r", 1, 100, axes[1, 1]),
    ]

    for col, title, cmap, vmin, vmax, ax_obj in configs:
        p = avg.pivot(index="val", columns="layer", values=col)
        fmt = ".2f" if "score" in col else ".1f"
        
        # PPLデータを持ってきて強調表示に使う
        ppl_col = col.replace("score", "ppl")
        p_ppl = avg.pivot(index="val", columns="layer", values=ppl_col) if "score" in col else p

        # DLS を appending する
        dls_score_col = "dyn_score" if "score" in col else "dyn_ppl"
        
        if not logit_avg.empty and dls_score_col in logit_avg.columns:
            p["Bhandari"] = logit_avg.set_index("val")[dls_score_col]
            if "score" in col:
                p_ppl["Bhandari"] = logit_avg.set_index("val")["dyn_ppl"]
            else:
                p_ppl["Bhandari"] = logit_avg.set_index("val")[dls_score_col]
                
        if not anti_avg.empty and dls_score_col in anti_avg.columns:
            p["Proposed"] = anti_avg.set_index("val")[dls_score_col]
            if "score" in col:
                p_ppl["Proposed"] = anti_avg.set_index("val")["dyn_ppl"]
            else:
                p_ppl["Proposed"] = anti_avg.set_index("val")[dls_score_col]

        sns.heatmap(p, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 9})
        
        if "Bhandari" in p.columns:
            idx = list(p.columns).index("Bhandari")
            ax_obj.axvline(x=idx, color="navy", linewidth=3)
        if "Proposed" in p.columns:
            idx = list(p.columns).index("Proposed")
            ax_obj.axvline(x=idx, color="darkred", linewidth=3)

        # PPL <= 25 のセルを囲む
        highlight_safe_cells(ax_obj, p_ppl, threshold=25.0)
        
        ax_obj.set_title(title + " (Border: PPL<=25, Navy: Bhandari, Red: Proposed)", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer (rightmost = DLS)")
        ax_obj.set_ylabel("Val")

    plt.suptitle("Layer-Sweep Summary (All Traits Average)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    summary_path = out_dir / "summary_all_traits.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved summary: {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="exp_steering_layer_analysis/results")
    ap.add_argument("--dyn_dir",   default="exp_steering_dyn_layer_compare/results")
    ap.add_argument("--out_dir",   default="exp_steering_layer_analysis/figures")
    ap.add_argument("--axis",      default=None, help="特定特性のみ。省略で全特性。")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    dyn_dir   = Path(args.dyn_dir)
    out_dir   = Path(args.out_dir)

    target_axes = [args.axis] if args.axis else TRAITS
    all_dfs = []
    all_logit_dfs = []
    all_anti_dfs = []

    for axis in target_axes:
        df = load_summary(input_dir, axis)
        if not df.empty:
            all_dfs.append(df)
            
        logit_df = load_dyn_summary(dyn_dir, axis, "logit_diff")
        if not logit_df.empty:
            all_logit_dfs.append(logit_df)
            
        anti_df = load_dyn_summary(dyn_dir, axis, "anti_alignment")
        if not anti_df.empty:
            all_anti_dfs.append(anti_df)

        plot_axis(df, axis, out_dir / axis, dyn_dir)

    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        logit_all_df = pd.concat(all_logit_dfs, ignore_index=True) if all_logit_dfs else pd.DataFrame()
        anti_all_df = pd.concat(all_anti_dfs, ignore_index=True) if all_anti_dfs else pd.DataFrame()
        
        make_summary_heatmaps(all_df, logit_all_df, anti_all_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
