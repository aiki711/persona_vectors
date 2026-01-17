#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_repetition_vs_alpha.py

04_repetition_only.py が出力した CSV から、
- Δs (ds_avg) vs alpha_total
- max_token_freq_ratio vs alpha_total
を、base / instruct を重ねて trait ごとに 1 画像として可視化する。

使い方:
  python scripts/05_repetition_vs_alpha.py \
    --csv     exp/mistral_7b/results_asst_cluster/repetition_asst_cluster.csv \
    --out_dir exp/mistral_7b/results_asst_cluster/repetition_plots
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 特性の表示順
TRAIT_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def _ensure_numeric(df: pd.DataFrame, cols):
    """指定カラムを数値型に強制変換（失敗したら NaN）"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_joint_for_trait(df_trait: pd.DataFrame, trait: str, out_path: str):
    """
    1 特性分のデータから、
    - 上段: ds_avg (性格スコア変化) vs alpha_total
    - 下段: max_token_freq_ratio (繰り返し度) vs alpha_total
    を base / instruct で重ねて描画。
    """
    if df_trait.empty:
        print(f"[Skip] trait={trait}: no data.")
        return

    # 集約：モデル×α ごとに平均
    agg = (
        df_trait
        .groupby(["model", "alpha_total"], as_index=False)
        .agg(
            ds_avg_mean=("ds_avg", "mean"),
            ds_avg_std=("ds_avg", "std"),
            max_token_freq_ratio_mean=("max_token_freq_ratio", "mean"),
            max_token_freq_ratio_std=("max_token_freq_ratio", "std"),
        )
    )

    # モデルごとに線を引く
    models = sorted(agg["model"].dropna().unique())
    if not models:
        print(f"[Skip] trait={trait}: no model info.")
        return

    print(f"  -> Plotting trait={trait}, models={models}")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6),
        gridspec_kw={"height_ratios": [2, 1.8]}
    )

    # 色をモデルごとに固定したい場合は palette を明示することもできる
    # ここでは matplotlib のデフォルトに任せる
    for m in models:
        sub = agg[agg["model"] == m].sort_values("alpha_total")
        x = sub["alpha_total"]
        y_ds = sub["ds_avg_mean"]
        y_rep = sub["max_token_freq_ratio_mean"]

        # 上段: Δs vs α
        ax1.plot(
            x, y_ds,
            marker="o",
            label=m,
        )

        # 下段: max_token_freq_ratio vs α
        ax2.plot(
            x, y_rep,
            marker="o",
            label=m,
        )

    # 上段の装飾
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Δs (mean)")
    ax1.set_title(f"{trait}: Personality shift vs α and Repetition vs α")

    # 下段の装飾
    ax2.set_xlabel("alpha_total")
    ax2.set_ylabel("max_token_freq_ratio (mean)")

    # 凡例は上段だけに表示（邪魔なら位置を変えても OK）
    ax1.legend(title="model", loc="best")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="04_repetition_only.py が出力した CSV")
    ap.add_argument("--out_dir", required=True, help="画像出力ディレクトリ")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    # 必要なカラムが無ければ終了
    required_cols = {"model", "trait", "alpha_total", "ds_avg", "max_token_freq_ratio"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}")
        return

    # 数値に変換
    df = _ensure_numeric(df, ["alpha_total", "ds_avg", "max_token_freq_ratio"])

    # trait ごとに描画
    print(f"[INFO] Creating plots to {args.out_dir}")
    traits = [t for t in TRAIT_ORDER if t in df["trait"].unique()]
    # 予想外の trait があれば、それも描いておく
    for t in sorted(df["trait"].unique()):
        if t not in traits:
            traits.append(t)

    for trait in traits:
        sub = df[df["trait"] == trait]
        out_path = os.path.join(args.out_dir, f"ds_and_repetition_vs_alpha_{trait}.png")
        plot_joint_for_trait(sub, trait, out_path)

    print("[Done] All plots generated.")


if __name__ == "__main__":
    main()
