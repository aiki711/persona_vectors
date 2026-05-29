#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 48_rank_layer_efficiency.py
#
# 品質予算（PPL）と安定性（StdDev）の制約下で、
# 各レイヤーが達成可能な最大スコア上昇を評価し、精鋭レイヤーを特定する。
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

def load_all_data(input_dir: Path, axis: str) -> pd.DataFrame:
    """全条件のCSVを読み込み、平均スコア、PPL、標準偏差を1つのDataFrameにまとめる"""
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv"
            if not csv_path.exists(): continue
            df = pd.read_csv(csv_path)
            
            records.append({
                "layer": layer,
                "val":   val,
                "base_score":  df["base_score"].mean(),
                "const_score": df["const_score"].mean(),
                "adapt_score": df["adapt_score"].mean(),
                "const_std":   df["const_score"].std(),
                "adapt_std":   df["adapt_score"].std(),
                "base_ppl":    df["base_ppl"].mean(),
                "const_ppl":   df["const_ppl"].mean(),
                "adapt_ppl":   df["adapt_ppl"].mean(),
            })
    return pd.DataFrame(records)

def analyze_efficiency(df: pd.DataFrame, ppl_budget=5.0, std_limit=1.0):
    """各レイヤーにおいて、予算内で最大のスコア上昇（Delta）を計算する"""
    res = []
    for layer in LAYERS:
        sub = df[df["layer"] == layer]
        if sub.empty: continue
        
        # Constant Steering の判定
        # 条件: PPLの悪化がppl_budget以内 かつ 標準偏差がstd_limit以内
        c_safe = sub[(sub["const_ppl"] <= sub["base_ppl"] + ppl_budget) & (sub["const_std"] <= std_limit)]
        c_max_delta = (c_safe["const_score"] - c_safe["base_score"]).max() if not c_safe.empty else 0.0
        
        # Adaptive Steering の判定
        a_safe = sub[(sub["adapt_ppl"] <= sub["base_ppl"] + ppl_budget) & (sub["adapt_std"] <= std_limit)]
        a_max_delta = (a_safe["adapt_score"] - a_safe["base_score"]).max() if not a_safe.empty else 0.0
        
        res.append({
            "layer": layer,
            "const_max_delta": c_max_delta,
            "adapt_max_delta": a_max_delta
        })
    return pd.DataFrame(res)

def plot_efficiency(all_results, out_dir: Path, ppl_budget):
    """レイヤーごとの最大効果をプロットする"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 全特性の平均を計算
    combined = pd.concat(all_results.values())
    avg_eff = combined.groupby("layer").mean(numeric_only=True).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_eff["layer"], avg_eff["const_max_delta"], "o-", label="Constant", color="#3498db", linewidth=2.5, markersize=8)
    ax.plot(avg_eff["layer"], avg_eff["adapt_max_delta"], "s-", label="Adaptive", color="#2ecc71", linewidth=2.5, markersize=8)
    
    ax.set_title(f"Layer Efficiency Ranking\n(Max Delta Score within PPL +{ppl_budget} & Std <= 1.0)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Max Score Increase (Delta)", fontsize=12)
    ax.set_xticks(LAYERS)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    
    # 上位レイヤーの注釈
    top_adapt = avg_eff.sort_values("adapt_max_delta", ascending=False).head(3)
    annotation_text = "Top Efficiency Layers (Adaptive):\n" + ", ".join([f"L{int(l)}" for l in top_adapt["layer"]])
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    summary_path = out_dir / "layer_efficiency_summary.png"
    plt.savefig(summary_path, dpi=200)
    plt.close()
    print(f"  Saved efficiency summary: {summary_path}")
    
    # 特性別のグラフも作成
    for trait, res_df in all_results.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(res_df["layer"], res_df["const_max_delta"], "o-", label="Constant", color="#3498db", alpha=0.7)
        ax.plot(res_df["layer"], res_df["adapt_max_delta"], "s-", label="Adaptive", color="#2ecc71", alpha=0.7)
        ax.set_title(f"Layer Efficiency: {trait.capitalize()}\n(Budget: PPL +{ppl_budget}, Std <= 1.0)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Max Delta Score")
        ax.set_xticks(LAYERS)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        plt.savefig(out_dir / f"efficiency_{trait}.png", dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="exp_steering_layer_analysis/results")
    parser.add_argument("--out_dir",   default="exp_steering_layer_analysis/layer_ranking")
    parser.add_argument("--ppl_budget", type=float, default=5.0)
    parser.add_argument("--std_limit",  type=float, default=1.0)
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    all_results = {}
    for trait in TRAITS:
        print(f"Analyzing layer efficiency for [{trait}]...")
        df = load_all_data(in_dir, trait)
        if df.empty:
            print(f"  No data found for {trait}")
            continue
        eff_df = analyze_efficiency(df, ppl_budget=args.ppl_budget, std_limit=args.std_limit)
        all_results[trait] = eff_df
    
    if all_results:
        plot_efficiency(all_results, out_dir, args.ppl_budget)
        
        # ランキングレポート作成
        summary_df = pd.concat(all_results.values())
        avg_eff = summary_df.groupby("layer").mean(numeric_only=True).reset_index()
        top_layers = avg_eff.sort_values("adapt_max_delta", ascending=False)
        
        report_path = out_dir / "top_layers_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Layer Efficiency Ranking (Multi-trait Average) ===\n")
            f.write(f"Constraints: PPL Increase <= {args.ppl_budget}, StdDev <= {args.std_limit}\n")
            f.write("------------------------------------------------------\n")
            f.write(top_layers[["layer", "adapt_max_delta", "const_max_delta"]].to_string(index=False))
            f.write("\n\nRecommendation:\n")
            f.write("The layers with high 'adapt_max_delta' are the best candidates for single-layer\n")
            f.write("and simultaneous multi-layer steering, as they provide the most change\n")
            f.write("per unit of quality degradation.\n")
            
        print(f"\nRanking Report:\n{top_layers[['layer', 'adapt_max_delta']].head(5)}")
        print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    main()
