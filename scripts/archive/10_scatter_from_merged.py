#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scatter plots from merged_metrics.csv:
- marker shape encodes model tag
- pos/neg are overlaid in ONE plot:
    pos: filled marker (range_hi)
    neg: hollow marker (abs_range_lo)
No seaborn. Matplotlib only. No explicit colors.

Outputs:
  figures_scatter/
    scatter_posneg__{split}__x={xcol}__ALL.png
"""

from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:
    raise RuntimeError("scipy が必要です (pip install scipy)") from e


# -----------------------
# utils / feature prep
# -----------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def safe_div(a, b):
    a = safe_num(a); b = safe_num(b)
    out = a / b
    out = out.where(np.isfinite(out))
    return out

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # y boundaries
    df["y_pos"] = safe_num(df.get("range_hi", np.nan))

    # ★修正: abs_range_lo が無ければ range_lo の絶対値を使う
    if "abs_range_lo" in df.columns:
        df["y_neg"] = safe_num(df["abs_range_lo"])
    elif "range_lo" in df.columns:
        df["y_neg"] = safe_num(df["range_lo"]).abs()
    else:
        df["y_neg"] = np.nan

    # rawnorm
    df["rawnorm_mean"] = safe_num(df.get("rawnorm_mean", np.nan))
    df["rawnorm_p10"]  = safe_num(df.get("rawnorm_p10", np.nan))

    # rms0 (near0 RMS before)
    if "mean_rms_near0_before" in df.columns:
        df["rms0_mean"] = safe_num(df["mean_rms_near0_before"])
    else:
        df["rms0_mean"] = safe_num(df.get("mean_rms0_before", np.nan))

    if "p10_rms_near0_before" in df.columns:
        df["rms0_p10"] = safe_num(df["p10_rms_near0_before"])
    else:
        df["rms0_p10"] = safe_num(df.get("p10_rms0_before", np.nan))

    return df

def corr_stats(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    m = np.isfinite(x.values) & np.isfinite(y.values)
    x = x.values[m]; y = y.values[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, int(len(x))
    pr = pearsonr(x, y).statistic
    sr = spearmanr(x, y).statistic
    # OLS: y = b0 + b1*x
    X = np.stack([np.ones_like(x), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope = float(beta[1])
    return float(pr), float(sr), float(slope), float(r2), int(len(x))


# -----------------------
# plotting
# -----------------------

def plot_scatter_posneg_by_model(
    df: pd.DataFrame,
    outdir: str,
    split: str,
    xcol: str,
    max_points_per_tag: int | None = None,
) -> str:
    sub = df[df["split"] == split].copy()
    need = ["tag", "trait", xcol, "y_pos", "y_neg"]
    for c in need:
        if c not in sub.columns:
            raise ValueError(f"missing column: {c}")

    tags = sorted(sub["tag"].dropna().astype(str).unique().tolist())
    if not tags:
        raise ValueError(f"no tags in split={split}")

    # できるだけ「塗り/中抜き」が素直に効くマーカーだけにする（+ や x は中抜きが分かりにくい）
    markers = ["o","s","^","D","P","X","v",">","<","*","h","H","p","d"]
    tag2m = {t: markers[i % len(markers)] for i, t in enumerate(tags)}

    # デフォルトの色サイクルから tag→色を割当（任意の色を手で決めない）
    cycle_colors = plt.rcParams.get("axes.prop_cycle", None)
    colors = cycle_colors.by_key().get("color", []) if cycle_colors is not None else []
    tag2c = {t: (colors[i % len(colors)] if colors else None) for i, t in enumerate(tags)}

    # pooled correlations (ALL models+traits)
    sub_pos = sub[np.isfinite(sub[xcol].astype(float)) & np.isfinite(sub["y_pos"].astype(float))].copy()
    pr_p, sr_p, slope_p, r2_p, n_p = corr_stats(sub_pos[xcol], sub_pos["y_pos"])

    sub_neg = sub[np.isfinite(sub[xcol].astype(float)) & np.isfinite(sub["y_neg"].astype(float))].copy()
    pr_n, sr_n, slope_n, r2_n, n_n = corr_stats(sub_neg[xcol], sub_neg["y_neg"])

    fig = plt.figure(figsize=(9.2, 6.8))
    ax = fig.add_subplot(111)

    # plot per tag
    for t in tags:
        st = sub[sub["tag"].astype(str) == t].copy()

        st_pos = st[np.isfinite(st[xcol].astype(float)) & np.isfinite(st["y_pos"].astype(float))]
        st_neg = st[np.isfinite(st[xcol].astype(float)) & np.isfinite(st["y_neg"].astype(float))]

        if max_points_per_tag is not None:
            if len(st_pos) > max_points_per_tag:
                st_pos = st_pos.sample(n=max_points_per_tag, random_state=0)
            if len(st_neg) > max_points_per_tag:
                st_neg = st_neg.sample(n=max_points_per_tag, random_state=0)

        m = tag2m[t]
        c = tag2c[t]  # デフォルトサイクル色

        # pos: filled
        if not st_pos.empty:
            ax.scatter(
                st_pos[xcol].values, st_pos["y_pos"].values,
                marker=m, s=55, alpha=0.75,
                linewidths=0.8,
                color=c,
                label=f"{t}",
                zorder=3
            )

        # neg: hollow (same marker), ★edgecolors を必ず付ける
        if not st_neg.empty:
            ax.scatter(
                st_neg[xcol].values, st_neg["y_neg"].values,
                marker=m, s=55,
                facecolors="none",
                edgecolors=c if c is not None else "black",
                linewidths=1.6,
                alpha=0.95,
                label=None,
                zorder=4
            )

    # ★保険：y軸を pos/neg 両方から決める（posだけで決まって neg が飛ぶのを防ぐ）
    y_all = np.concatenate([
        sub["y_pos"].to_numpy(dtype=float, copy=False),
        sub["y_neg"].to_numpy(dtype=float, copy=False),
    ])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size:
        ylo, yhi = np.percentile(y_all, [1, 99])
        pad = (yhi - ylo) * 0.05 if yhi > ylo else 1.0
        ax.set_ylim(ylo - pad, yhi + pad)

    ax.set_xlabel(xcol)
    ax.set_ylabel("boundary alpha (pos: range_hi filled, neg: abs_range_lo hollow)")
    ax.set_title(
        f"pos+neg overlaid | split={split} | x={xcol}\n"
        f"POS: n={n_p} pearson={pr_p:.3f} spearman={sr_p:.3f} slope={slope_p:.3g} R2={r2_p:.3f}\n"
        f"NEG: n={n_n} pearson={pr_n:.3f} spearman={sr_n:.3f} slope={slope_n:.3g} R2={r2_n:.3f}"
    )

    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()

    out = os.path.join(outdir, safe_filename(f"scatter_posneg__{split}__x={xcol}__ALL.png"))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------
# main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged_metrics.csv")
    ap.add_argument("--outdir", default="figures_scatter", help="output directory")
    ap.add_argument("--splits", default="", help="comma-separated splits (default: all)")
    ap.add_argument("--xvars", default="rms0_mean,rms0_p10", help="comma-separated x variables")
    ap.add_argument("--max_points_per_tag", type=int, default=0,
                    help="if >0, downsample per tag (pos/neg separately) to this maximum")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.merged)
    df = prepare_features(df)
    
    for c in ["tag", "split", "trait"]:
        if c not in df.columns:
            raise ValueError("merged_metrics.csv must contain at least: tag, split, trait")

    splits = args.splits.split(",") if args.splits.strip() else sorted(df["split"].unique().tolist())
    xvars = [s.strip() for s in args.xvars.split(",") if s.strip()]
    max_per_tag = args.max_points_per_tag if args.max_points_per_tag and args.max_points_per_tag > 0 else None

    print("== scatter pos+neg (marker=tag) ==")
    for sp in splits:
        for xcol in xvars:
            try:
                out = plot_scatter_posneg_by_model(df, args.outdir, sp, xcol, max_points_per_tag=max_per_tag)
                print("saved:", out)
            except Exception as e:
                print("skip:", sp, xcol, "reason:", e)

    print("done.")

if __name__ == "__main__":
    main()
