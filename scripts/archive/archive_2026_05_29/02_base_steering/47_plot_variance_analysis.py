#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 47_plot_variance_analysis.py
#
# 性格スコアの平均(Mean)と標準偏差(StdDev)を並べて可視化し、
# ステアリングの一貫性と安定性を分析する。
#

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.5, 1, 2, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40]

def load_stats_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    """CSVを読み込んで、各条件の平均と標準偏差を計算して返す"""
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            
            # 各スコアの統計
            res = {
                "layer": layer,
                "val":   val,
                "base_score_mean": df["base_score"].mean(),
                "base_score_std":  df["base_score"].std(),
                "const_score_mean": df["const_score"].mean(),
                "const_score_std":  df["const_score"].std(),
                "adapt_score_mean": df["adapt_score"].mean(),
                "adapt_score_std":  df["adapt_score"].std(),
                "base_ppl_mean":    df["base_ppl"].mean(),
                "const_ppl_mean":   df["const_ppl"].mean(),
                "adapt_ppl_mean":   df["adapt_ppl"].mean(),
            }
            records.append(res)
    return pd.DataFrame(records)

def highlight_safe_cells(ax, p_ppl, p_std, ppl_threshold=24.0, std_threshold=1.0):
    """PPLと標準偏差が共に閾値以下のセルを枠線で囲む"""
    rows, cols = p_ppl.shape
    for r in range(rows):
        for c in range(cols):
            ppl_val = p_ppl.iloc[r, c]
            std_val = p_std.iloc[r, c]
            if not np.isnan(ppl_val) and not np.isnan(std_val):
                if ppl_val <= ppl_threshold and std_val <= std_threshold:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", lw=1.8))

def plot_axis_variance(df: pd.DataFrame, axis: str, out_dir: Path):
    """指定された特性の Mean と StdDev のヒートマップを生成"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ピボットテーブル作成
    def get_pivot(col):
        return df.pivot(index="val", columns="layer", values=col)

    p_c_mean = get_pivot("const_score_mean")
    p_c_std  = get_pivot("const_score_std")
    p_a_mean = get_pivot("adapt_score_mean")
    p_a_std  = get_pivot("adapt_score_std")
    p_c_ppl  = get_pivot("const_ppl_mean")
    p_a_ppl  = get_pivot("adapt_ppl_mean")

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # レイアウト:
    # [0,0]: Const Mean    [0,1]: Const StdDev
    # [1,0]: Adapt Mean    [1,1]: Adapt StdDev
    
    s_vmin, s_vmax = 0, 5
    std_vmin, std_vmax = 0, 2.0
    
    cmap_score = "YlGn"
    cmap_std   = "OrRd"

    configs = [
        (axes[0, 0], p_c_mean, p_c_ppl, p_c_std, "Constant — Score Mean", cmap_score, s_vmin, s_vmax, ".2f"),
        (axes[0, 1], p_c_std,  p_c_ppl, p_c_std, "Constant — Score StdDev", cmap_std,   std_vmin, std_vmax, ".2f"),
        (axes[1, 0], p_a_mean, p_a_ppl, p_a_std, "Adaptive — Score Mean", cmap_score, s_vmin, s_vmax, ".2f"),
        (axes[1, 1], p_a_std,  p_a_ppl, p_a_std, "Adaptive — Score StdDev", cmap_std,   std_vmin, std_vmax, ".2f"),
    ]

    for ax_obj, p_data, p_ppl_ref, p_std_ref, title, cmap, vmin, vmax, fmt in configs:
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 8})
        
        # PPL <= 24 AND StdDev <= 1.0 のセルを囲む
        highlight_safe_cells(ax_obj, p_ppl_ref, p_std_ref, ppl_threshold=24.0, std_threshold=1.0)
        
        ax_obj.set_title(f"{title} [{axis.capitalize()}] (Border: PPL<=24 & Std<=1.0)", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Steering Strength (Val)")

    # BaselineのStdDevを参考として表示
    b_std_avg = df["base_score_std"].mean()
    plt.suptitle(f"Steering Stability Analysis: {axis.capitalize()} (Baseline Avg StdDev: {b_std_avg:.2f})", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    out_path = out_dir / f"stability_{axis}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved stability heatmap: {out_path}")

def make_summary_variance(all_df: pd.DataFrame, out_dir: Path):
    """全特性平均の安定性ヒートマップ"""
    if all_df.empty: return
    
    avg = all_df.groupby(["layer", "val"]).mean(numeric_only=True).reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    configs = [
        ("const_score_mean", "Constant — Score Mean (Avg)", "YlGn", 0, 5, axes[0, 0]),
        ("const_score_std",  "Constant — Score StdDev (Avg)", "OrRd", 0, 2, axes[0, 1]),
        ("adapt_score_mean", "Adaptive — Score Mean (Avg)",  "YlGn", 0, 5, axes[1, 0]),
        ("adapt_score_std",  "Adaptive — Score StdDev (Avg)",  "OrRd", 0, 2, axes[1, 1]),
    ]
    
    for col, title, cmap, vmin, vmax, ax_obj in configs:
        p = avg.pivot(index="val", columns="layer", values=col)
        # PPLデータとStdDevデータ
        is_adapt = "adapt" in col
        ppl_col = "adapt_ppl_mean" if is_adapt else "const_ppl_mean"
        std_col = "adapt_score_std" if is_adapt else "const_score_std"
        
        p_ppl = avg.pivot(index="val", columns="layer", values=ppl_col)
        p_std = avg.pivot(index="val", columns="layer", values=std_col)

        sns.heatmap(p, annot=True, fmt=".2f", cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 8})
        
        highlight_safe_cells(ax_obj, p_ppl, p_std, ppl_threshold=24.0, std_threshold=1.0)
        ax_obj.set_title(title + " (Border: PPL<=24 & Std<=1.0)", fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer")
        ax_obj.set_ylabel("Val")

    plt.suptitle("Global Summary: Steering Stability (All Traits Average)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    summary_path = out_dir / "summary_stability_all_traits.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved Global Stability summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="exp_steering_layer_analysis/results")
    parser.add_argument("--out_dir",   default="exp_steering_layer_analysis/figures_stability")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    all_dfs = []
    for trait in TRAITS:
        print(f"Analyzing stability for [{trait}]...")
        df = load_stats_summary(in_dir, trait)
        if df.empty:
            print(f"  No data for {trait}")
            continue
        plot_axis_variance(df, trait, out_dir)
        all_dfs.append(df)
        
    if all_dfs:
        full_df = pd.concat(all_dfs)
        make_summary_variance(full_df, out_dir)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
