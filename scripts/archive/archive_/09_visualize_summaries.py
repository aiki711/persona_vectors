#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize correlation + scaling-dispersion summaries.

- correlation heatmaps: allow dropping weak predictors
- dispersion ratio bar: pos & neg in one plot (no duplicate loops)

No seaborn. Matplotlib only.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------

# 追加: 表示名を差し替えるだけのマップ
DISPLAY_MAP = {
    # predictors
    "rms0_mean": "hidden_scale_mean",
    "rms0_p10": "hidden_scale_p10",

    # norm tags
    "norm_rms0": "norm_hidden_scale",
    "norm_rms0_raw": "norm_hidden_scale_raw",
}

def disp_name(s: str) -> str:
    """列名やタグを、表示用の名前に変換（無ければそのまま）"""
    return DISPLAY_MAP.get(s, s)

def disp_text(s: str) -> str:
    """文字列中のトークンも置換したい場合（境界名など）"""
    # boundary などに rms0 が含まれるケースもまとめて置換できる
    out = s
    # 長いキー優先で置換
    for k in sorted(DISPLAY_MAP.keys(), key=len, reverse=True):
        out = out.replace(k, DISPLAY_MAP[k])
    return out

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)

def parse_corr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide correlation columns to long:
      split, trait, method, boundary, predictor, corr
    from columns like:
      pearson__pos_range_hi__rawnorm_mean
    """
    base_cols = [c for c in ["split", "trait"] if c in df.columns]
    if len(base_cols) < 2:
        raise ValueError("corr_pred_summary.csv must contain columns: split, trait")

    pattern = re.compile(r"^(pearson|spearman)__([^_].*?)__(.+)$")
    records = []
    for col in df.columns:
        m = pattern.match(col)
        if not m:
            continue
        method, boundary, predictor = m.group(1), m.group(2), m.group(3)
        tmp = df[["split", "trait", col]].copy()
        tmp["method"] = method
        tmp["boundary"] = boundary
        tmp["predictor"] = predictor
        tmp = tmp.rename(columns={col: "corr"})
        records.append(tmp)

    if not records:
        raise ValueError("No correlation columns found. Expected columns like pearson__pos_range_hi__rawnorm_mean")

    long_df = pd.concat(records, ignore_index=True)
    return long_df[["split", "trait", "method", "boundary", "predictor", "corr"]]


# ---------------------------
# Plotting: (A) Correlation heatmaps
# ---------------------------

def plot_corr_heatmap(
    long_corr: pd.DataFrame,
    outdir: str,
    split: str,
    method: str,
    boundary: str,
    traits_order: List[str] | None = None,
    predictors_order: List[str] | None = None,
    drop_predictors: set[str] | None = None,
) -> str:
    """
    One figure = one heatmap:
      rows: predictors
      cols: traits
      color: correlation [-1,1]
    """
    sub = long_corr[
        (long_corr["split"] == split) &
        (long_corr["method"] == method) &
        (long_corr["boundary"] == boundary)
    ].copy()

    if drop_predictors:
        sub = sub[~sub["predictor"].isin(drop_predictors)].copy()

    if sub.empty:
        raise ValueError(f"No data for split={split}, method={method}, boundary={boundary} (after filtering predictors)")

    if traits_order is None:
        traits_order = sorted(sub["trait"].unique().tolist())

    if predictors_order is None:
        # stable order; anything else goes to the end
        pref = ["rawnorm_mean", "rawnorm_p10", "rms0_mean", "rms0_p10"]
        preds = sub["predictor"].unique().tolist()
        preds = [p for p in preds if (drop_predictors is None or p not in drop_predictors)]
        predictors_order = [p for p in pref if p in preds] + [p for p in sorted(preds) if p not in pref]

    # if predictors_order became empty (e.g., everything dropped)
    if not predictors_order:
        raise ValueError("No predictors left to plot (check drop_predictors).")

    pivot = (
        sub.pivot_table(index="predictor", columns="trait", values="corr", aggfunc="mean")
           .reindex(index=predictors_order, columns=traits_order)
    )

    fig = plt.figure(figsize=(1.2 + 1.2 * len(traits_order), 1.6 + 0.6 * len(predictors_order)))
    ax = fig.add_subplot(111)

    data = pivot.values.astype(float)
    im = ax.imshow(data, vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(traits_order)))
    ax.set_xticklabels(traits_order, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(predictors_order)))
    ax.set_yticklabels([disp_name(p) for p in predictors_order])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title(f"Correlation heatmap ({method}) | split={split} | boundary={disp_text(boundary)}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("corr")

    fig.tight_layout()
    outpath = os.path.join(outdir, safe_filename(f"heatmap_corr__{split}__{method}__{boundary}.png"))
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


# ---------------------------
# Plotting: (B) Dispersion / normalization
# ---------------------------

def compute_cv(mean: pd.Series, std: pd.Series) -> pd.Series:
    mean = mean.astype(float)
    std = std.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / np.abs(mean)
    return cv.replace([np.inf, -np.inf], np.nan)

def plot_bar_cv_ratio_posneg(
    disp: pd.DataFrame,
    outdir: str,
    split: str,
    normtag: str,         # "norm_rms0" or "norm_rms0_raw"
    category_axis: str = "trait",
) -> str:
    if normtag not in ("norm_rms0", "norm_rms0_raw"):
        raise ValueError("normtag must be norm_rms0 or norm_rms0_raw")

    sub = disp[disp["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"No dispersion rows for split={split}")

    need = [
        "pos_raw__mean", "pos_raw__std", f"pos_{normtag}__mean", f"pos_{normtag}__std",
        "neg_raw__mean", "neg_raw__std", f"neg_{normtag}__mean", f"neg_{normtag}__std",
        category_axis
    ]
    for c in need:
        if c not in sub.columns:
            raise ValueError(f"Missing column in scaling_dispersion.csv: {c}")

    cv_pos_raw  = compute_cv(sub["pos_raw__mean"], sub["pos_raw__std"])
    cv_pos_norm = compute_cv(sub[f"pos_{normtag}__mean"], sub[f"pos_{normtag}__std"])
    cv_neg_raw  = compute_cv(sub["neg_raw__mean"], sub["neg_raw__std"])
    cv_neg_norm = compute_cv(sub[f"neg_{normtag}__mean"], sub[f"neg_{normtag}__std"])

    sub["pos_ratio"] = (cv_pos_norm / cv_pos_raw).replace([np.inf, -np.inf], np.nan)
    sub["neg_ratio"] = (cv_neg_norm / cv_neg_raw).replace([np.inf, -np.inf], np.nan)
    sub["avg_ratio"] = sub[["pos_ratio", "neg_ratio"]].mean(axis=1)

    sub = sub.sort_values("avg_ratio", ascending=True)
    cats = sub[category_axis].tolist()
    y = np.arange(len(cats))
    h = 0.38

    fig = plt.figure(figsize=(10, 0.75 * len(cats) + 2.2))
    ax = fig.add_subplot(111)

    ax.barh(y - h/2, sub["pos_ratio"].values, height=h, label=f"pos: CV({disp_name(normtag)})/CV(raw)")
    ax.barh(y + h/2, sub["neg_ratio"].values, height=h, label=f"neg: CV({disp_name(normtag)})/CV(raw)")

    ax.axvline(1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel(f"CV({disp_name(normtag)}) / CV(raw)   (<1 means reduced dispersion)")
    ax.set_title(f"Dispersion ratio (pos & neg in one plot) | split={split} | norm={disp_name(normtag)}")

    for i, (pr, nr) in enumerate(zip(sub["pos_ratio"].values, sub["neg_ratio"].values)):
        if np.isfinite(pr):
            ax.text(pr, y[i] - h/2, f" {pr:.2f}", va="center", fontsize=9)
        if np.isfinite(nr):
            ax.text(nr, y[i] + h/2, f" {nr:.2f}", va="center", fontsize=9)

    ax.legend()
    fig.tight_layout()

    out = os.path.join(outdir, safe_filename(f"bar_cv_ratio_posneg__{split}__{normtag}.png"))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", type=str, required=True, help="Path to corr_pred_summary.csv")
    ap.add_argument("--disp", type=str, required=True, help="Path to scaling_dispersion.csv")
    ap.add_argument("--outdir", type=str, default="figures", help="Output directory")
    ap.add_argument("--splits", type=str, default="", help="Comma-separated splits to plot (default: all)")
    ap.add_argument("--methods", type=str, default="pearson,spearman", help="Comma-separated methods")
    ap.add_argument("--boundaries", type=str, default="", help="Comma-separated boundaries (default: all found)")

    # ★追加: 弱い説明変数を除外したいとき
    ap.add_argument(
        "--drop_predictors",
        type=str,
        default="rms0_over_rawmean,rms0_over_rawp10",
        help="Comma-separated predictor names to drop from heatmaps (default drops rms0_over_rawmean & rms0_over_rawp10)",
    )

    args = ap.parse_args()
    ensure_dir(args.outdir)

    corr_df = pd.read_csv(args.corr)
    disp_df = pd.read_csv(args.disp)

    long_corr = parse_corr_columns(corr_df)

    drop_set = {s.strip() for s in args.drop_predictors.split(",") if s.strip()}

    splits = args.splits.split(",") if args.splits.strip() else sorted(long_corr["split"].unique().tolist())
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    boundaries = args.boundaries.split(",") if args.boundaries.strip() else sorted(long_corr["boundary"].unique().tolist())

    # (A) correlation heatmaps
    print("== (A) correlation heatmaps ==")
    for sp in splits:
        for m in methods:
            for b in boundaries:
                try:
                    out = plot_corr_heatmap(
                        long_corr, args.outdir,
                        split=sp, method=m, boundary=b,
                        drop_predictors=drop_set
                    )
                    print("saved:", out)
                except Exception as e:
                    print("skip:", sp, m, b, "reason:", e)

    # (B) dispersion ratio (pos & neg in one plot)
    print("== (B) dispersion / normalization ==")
    disp_splits = sorted(disp_df["split"].unique().tolist())
    for sp in disp_splits:
        for normtag in ("norm_rms0", "norm_rms0_raw"):
            try:
                out = plot_bar_cv_ratio_posneg(disp_df, args.outdir, split=sp, normtag=normtag)
                print("saved:", out)
            except Exception as e:
                print("skip:", sp, normtag, "reason:", e)

    print("done.")


if __name__ == "__main__":
    main()
