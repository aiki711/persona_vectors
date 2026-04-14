#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 41_plot_layer_sweep.py
#
# Layer-Sweep 実験の結果を可視化する。
#
# 出力:
#   1. ヒートマップ (Layer x Val) - 性格スコア（Constant / Adaptive）
#   2. ヒートマップ (Layer x Val) - PPL（Constant / Adaptive）
#   3. スコア-PPLトレードオフプロット (Layer ごとに色付け)
#
# Usage:
#   python scripts/41_plot_layer_sweep.py \
#     --input_dir exp_steering_layer_sweep/results \
#     --out_dir exp_steering_layer_sweep/figures \
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

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.03, 0.06, 0.09, 0.12, 0.15]


def load_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    """指定された特性の全CSVを読み込んで集計DataFrameを返す"""
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
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


def make_heatmap(pivot: pd.DataFrame, title: str, out_path: Path,
                 cmap: str = "YlOrRd", fmt: str = ".2f", vmin=None, vmax=None):
    """pivot (index=val, columns=layer) のヒートマップを保存"""
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor="gray",
                ax=ax, annot_kws={"size": 9})
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Steering Strength (Val)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def make_tradeoff_plot(df: pd.DataFrame, axis: str, out_path: Path):
    """Layer を色、Val を大きさにしてスコア vs PPL をプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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
                    color=color, label=f"L{layer}", alpha=0.8, linewidth=1.5)
            for _, row in sub.iterrows():
                ax.annotate(f"{row['val']}", (row[score_col], row[ppl_col]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=color, alpha=0.7)
        
        # ベースラインの平均
        base_score = df["base_score"].mean()
        base_ppl  = df["base_ppl"].mean()
        ax.scatter([base_score], [base_ppl], color="black", marker="*", s=200,
                   zorder=5, label="Baseline")

        ax.set_title(f"{mode} Steering — {axis.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Personality Score ({axis.capitalize()})", fontsize=11)
        ax.set_ylabel("Perplexity (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_axis(df: pd.DataFrame, axis: str, out_dir: Path):
    print(f"\n[{axis}] found {len(df)} conditions")
    if df.empty:
        print("  No data, skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pivot tables (index=val, columns=layer)
    def pivot(col):
        return df.pivot(index="val", columns="layer", values=col)

    # ---- 1. スコアヒートマップ (Constant & Adaptive 並べて) ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax_obj, col, label in [
        (axes[0], "const_score", "Constant Steering — Score"),
        (axes[1], "adapt_score", "Adaptive Steering — Score"),
    ]:
        p = pivot(col)
        sns.heatmap(p, annot=True, fmt=".2f", cmap="YlGn",
                    vmin=1, vmax=5,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 9})
        ax_obj.set_title(f"{label} [{axis.capitalize()}]", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Val")
    plt.tight_layout()
    score_path = out_dir / f"heatmap_{axis}_score.png"
    plt.savefig(score_path, dpi=200)
    plt.close()
    print(f"  Saved: {score_path}")

    # ---- 2. PPLヒートマップ ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax_obj, col, label in [
        (axes[0], "const_ppl", "Constant Steering — PPL"),
        (axes[1], "adapt_ppl", "Adaptive Steering — PPL"),
    ]:
        p = pivot(col)
        sns.heatmap(p, annot=True, fmt=".1f", cmap="YlOrRd",
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 8})
        ax_obj.set_title(f"{label} [{axis.capitalize()}]", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Val")
    plt.tight_layout()
    ppl_path = out_dir / f"heatmap_{axis}_ppl.png"
    plt.savefig(ppl_path, dpi=200)
    plt.close()
    print(f"  Saved: {ppl_path}")

    # ---- 3. スコア-PPL トレードオフ ----
    tradeoff_path = out_dir / f"tradeoff_{axis}.png"
    make_tradeoff_plot(df, axis, tradeoff_path)


def make_summary_heatmaps(all_df: pd.DataFrame, out_dir: Path):
    """全特性を平均した Layer x Val ヒートマップ"""
    if all_df.empty:
        return
    avg = all_df.groupby(["layer", "val"])[
        ["const_score", "adapt_score", "const_ppl", "adapt_ppl"]
    ].mean().reset_index()

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    configs = [
        ("const_score", "Constant — Score (All traits avg)", "YlGn", 1, 5),
        ("adapt_score", "Adaptive — Score (All traits avg)",  "YlGn", 1, 5),
        ("const_ppl",   "Constant — PPL (All traits avg)",    "YlOrRd", None, None),
        ("adapt_ppl",   "Adaptive — PPL (All traits avg)",    "YlOrRd", None, None),
    ]
    for ax_obj, (col, title, cmap, vmin, vmax) in zip(axes.flatten(), configs):
        p = avg.pivot(index="val", columns="layer", values=col)
        fmt = ".2f" if "score" in col else ".1f"
        sns.heatmap(p, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 9})
        ax_obj.set_title(title, fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Val")

    plt.suptitle("Layer-Sweep Summary (All Traits Average)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    summary_path = out_dir / "summary_all_traits.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved summary: {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="exp_steering_layer_sweep/results")
    ap.add_argument("--out_dir",   default="exp_steering_layer_sweep/figures")
    ap.add_argument("--axis",      default=None, help="特定特性のみ。省略で全特性。")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)

    target_axes = [args.axis] if args.axis else TRAITS
    all_dfs = []

    for axis in target_axes:
        df = load_summary(input_dir, axis)
        if not df.empty:
            all_dfs.append(df)
        plot_axis(df, axis, out_dir / axis)

    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        make_summary_heatmaps(all_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
