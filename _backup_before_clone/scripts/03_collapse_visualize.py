#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_visualize_collapse.py

目的:
- 02_analyze_collapse_from_logs.py で作成した CSV を読み込み、
  α と score_mean / repetition_ratio_mean / toxic_ratio_mean の関係を
  モデル (base / instruct) ごとに折れ線グラフでプロットする。

使い方例:
  python scripts/03_visualize_collapse.py \
    --input_path exp/mistral_7b/results_asst_cluster/collapse_asst_cluster.csv \
    --out_dir    exp/mistral_7b/results_asst_cluster/plots
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TRAIT_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def load_csv(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    if p.is_dir():
        # ディレクトリが来た場合は *.csv を全部読み込んで concat
        csv_files = sorted(p.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {p}")
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {p}")
        df = pd.read_csv(p)

    return df


def plot_per_trait(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    traits = [t for t in TRAIT_ORDER if t in df["trait"].unique()]

    models = sorted(df["model"].dropna().unique())
    colors = {"base": "tab:blue", "instruct": "tab:orange"}

    for trait in traits:
        sub = df[df["trait"] == trait].copy()
        if sub.empty:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
        fig.suptitle(f"Collapse vs Alpha – Trait: {trait}", fontsize=14)

        # 1) score_mean vs alpha_total
        ax = axes[0]
        for m in models:
            msub = sub[sub["model"] == m].sort_values("alpha_total")
            if msub.empty:
                continue
            ax.plot(
                msub["alpha_total"],
                msub["score_mean"],
                marker="o",
                label=m,
                linewidth=2,
            )
        ax.set_ylabel("score_mean")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2) repetition_ratio_mean vs alpha_total
        ax = axes[1]
        for m in models:
            msub = sub[sub["model"] == m].sort_values("alpha_total")
            if msub.empty:
                continue
            ax.plot(
                msub["alpha_total"],
                msub["repetition_ratio_mean"],
                marker="o",
                label=m,
                linewidth=2,
            )
        ax.set_ylabel("repetition_ratio")
        ax.grid(True, alpha=0.3)

        # 3) toxic_ratio_mean vs alpha_total
        ax = axes[2]
        for m in models:
            msub = sub[sub["model"] == m].sort_values("alpha_total")
            if msub.empty:
                continue
            ax.plot(
                msub["alpha_total"],
                msub["toxic_ratio_mean"],
                marker="o",
                label=m,
                linewidth=2,
            )
        ax.set_ylabel("toxic_ratio")
        ax.set_xlabel("alpha_total")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_file = out_dir / f"collapse_vs_alpha_{trait}.png"
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"[SAVE] {out_file}")


def plot_overall(df: pd.DataFrame, out_dir: Path):
    """
    全 trait を平均した α vs 指標 の概要図も 1 枚出しておくと、
    「どの α で崩壊しやすいか」のざっくり把握に便利。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        df.groupby(["model", "alpha_total"])
        .agg(
            score_mean=("score_mean", "mean"),
            repetition_ratio_mean=("repetition_ratio_mean", "mean"),
            toxic_ratio_mean=("toxic_ratio_mean", "mean"),
        )
        .reset_index()
    )

    models = sorted(grouped["model"].unique())

    # score_mean
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    fig.suptitle("Collapse vs Alpha – ALL traits (average)", fontsize=14)

    # score_mean
    ax = axes[0]
    for m in models:
        msub = grouped[grouped["model"] == m].sort_values("alpha_total")
        ax.plot(msub["alpha_total"], msub["score_mean"], marker="o", label=m, linewidth=2)
    ax.set_ylabel("score_mean (avg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # repetition_ratio_mean
    ax = axes[1]
    for m in models:
        msub = grouped[grouped["model"] == m].sort_values("alpha_total")
        ax.plot(
            msub["alpha_total"],
            msub["repetition_ratio_mean"],
            marker="o",
            label=m,
            linewidth=2,
        )
    ax.set_ylabel("repetition_ratio (avg)")
    ax.grid(True, alpha=0.3)

    # toxic_ratio_mean
    ax = axes[2]
    for m in models:
        msub = grouped[grouped["model"] == m].sort_values("alpha_total")
        ax.plot(
            msub["alpha_total"],
            msub["toxic_ratio_mean"],
            marker="o",
            label=m,
            linewidth=2,
        )
    ax.set_ylabel("toxic_ratio (avg)")
    ax.set_xlabel("alpha_total")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = out_dir / "collapse_vs_alpha_ALL_traits.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="collapse metrics CSV (or directory)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = load_csv(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(df.head())

    plot_per_trait(df, out_dir)
    plot_overall(df, out_dir)


if __name__ == "__main__":
    main()
