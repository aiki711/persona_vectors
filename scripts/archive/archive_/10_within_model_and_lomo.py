#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Within-model (tag-wise) correlation/regression summary + LOMO evaluation (Plan B).

Key idea:
- merged_metrics.csv is already aggregated per (tag, split, trait) -> 1 row each.
- Therefore:
  * within-model stats should use "traits as datapoints" inside each (split, tag).
  * LOMO should pool traits inside each split to keep coverage.

Input:
  merged_metrics.csv (must include: tag, split, trait, range_hi, abs_range_lo or range_lo,
                      plus predictor columns like mean_rms_near0_before etc.)

Outputs (default outdir=figures_eval):
  within_model_stats.csv
  lomo_eval.csv
  coverage_report.csv

No seaborn. Matplotlib not required here (CSV only).
"""

from __future__ import annotations
import argparse, os, re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:
    raise RuntimeError("scipy が必要です (pip install scipy)") from e


# -----------------------
# utils
# -----------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def finite_xy(x, y):
    x = pd.Series(x).astype(float).values
    y = pd.Series(y).astype(float).values
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def mae(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[m] - yhat[m])))

def rmse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))

def ols_fit_1d(x, y):
    """
    Fit y = b0 + b1*x by least squares.
    Returns: b0, b1, r2
    """
    x2, y2 = finite_xy(x, y)
    if len(x2) < 3:
        return np.nan, np.nan, np.nan
    X = np.stack([np.ones_like(x2), x2], axis=1)
    beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
    b0, b1 = float(beta[0]), float(beta[1])
    yhat = X @ beta
    ss_res = float(np.sum((y2 - yhat) ** 2))
    ss_tot = float(np.sum((y2 - y2.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b0, b1, float(r2)

def corr_stats_1d(x, y):
    """
    Pearson/Spearman + OLS slope/R2 + MAE/RMSE (in-sample).
    """
    x2, y2 = finite_xy(x, y)
    n = int(len(x2))
    if n < 3:
        return dict(n=n, pearson=np.nan, spearman=np.nan,
                    slope=np.nan, intercept=np.nan, r2=np.nan,
                    mae=np.nan, rmse=np.nan)

    pr = float(pearsonr(x2, y2).statistic)
    sr = float(spearmanr(x2, y2).statistic)
    b0, b1, r2 = ols_fit_1d(x2, y2)
    yhat = b0 + b1 * x2
    return dict(n=n, pearson=pr, spearman=sr,
                slope=float(b1), intercept=float(b0), r2=float(r2),
                mae=mae(y2, yhat), rmse=rmse(y2, yhat))


# -----------------------
# feature prep
# -----------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # targets (alpha boundaries)
    df["y_pos"] = safe_num(df.get("range_hi", np.nan))
    if "abs_range_lo" in df.columns:
        df["y_neg"] = safe_num(df["abs_range_lo"])
    elif "range_lo" in df.columns:
        df["y_neg"] = safe_num(df["range_lo"]).abs()
    else:
        df["y_neg"] = np.nan

    # predictors (common)
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


# -----------------------
# coverage helper
# -----------------------

def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    # how many rows exist per (tag, split, trait)
    c = (
        df.groupby(["tag", "split", "trait"], dropna=False)
          .size()
          .reset_index(name="n_rows")
          .sort_values(["tag", "split", "trait"])
    )
    return c


# -----------------------
# Step 1 & 2: within-model stats (traits are datapoints)
# -----------------------

def compute_within_model_stats_across_traits(
    df: pd.DataFrame,
    xcols: List[str],
    splits: List[str] | None = None,
    traits: List[str] | None = None,
    tags: List[str] | None = None,
) -> pd.DataFrame:
    need = ["tag", "split", "trait", "y_pos", "y_neg"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"merged_metrics.csv missing required column: {c}")

    sub = df.copy()
    if splits is not None:
        sub = sub[sub["split"].isin(splits)]
    if traits is not None:
        sub = sub[sub["trait"].isin(traits)]
    if tags is not None:
        sub = sub[sub["tag"].astype(str).isin(tags)]

    rows = []
    # group by (split, tag) so that traits become points
    for (sp, tg), g in sub.groupby(["split", "tag"], dropna=False):
        for xcol in xcols:
            if xcol not in g.columns:
                continue

            st_pos = corr_stats_1d(g[xcol], g["y_pos"])
            rows.append({
                "split": sp, "tag": str(tg), "boundary": "pos_range_hi", "xcol": xcol,
                **st_pos
            })

            st_neg = corr_stats_1d(g[xcol], g["y_neg"])
            rows.append({
                "split": sp, "tag": str(tg), "boundary": "neg_abs_range_lo", "xcol": xcol,
                **st_neg
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No rows produced. Check filters / xcols.")
    return out


# -----------------------
# Step 3: LOMO (pool traits within each split)
# -----------------------

def lomo_eval_pool_traits(
    df: pd.DataFrame,
    xcols: List[str],
    splits: List[str] | None = None,
    traits: List[str] | None = None,
    tags: List[str] | None = None,
    min_train_points: int = 10,
    min_test_points: int = 3,
) -> pd.DataFrame:
    sub = df.copy()
    if splits is not None:
        sub = sub[sub["split"].isin(splits)]
    if traits is not None:
        sub = sub[sub["trait"].isin(traits)]
    if tags is not None:
        sub = sub[sub["tag"].astype(str).isin(tags)]

    all_tags = sorted(sub["tag"].dropna().astype(str).unique().tolist())
    if len(all_tags) < 2:
        raise ValueError("LOMO needs at least 2 tags/models.")

    all_splits = sorted(sub["split"].dropna().astype(str).unique().tolist())

    rows = []
    for held in all_tags:
        train_df = sub[sub["tag"].astype(str) != held].copy()
        test_df  = sub[sub["tag"].astype(str) == held].copy()

        for sp in all_splits:
            tr_sp = train_df[train_df["split"].astype(str) == sp].copy()
            te_sp = test_df[test_df["split"].astype(str) == sp].copy()
            if te_sp.empty:
                continue

            for xcol in xcols:
                if xcol not in tr_sp.columns or xcol not in te_sp.columns:
                    continue

                # ---- POS ----
                x_tr, y_tr = finite_xy(tr_sp[xcol], tr_sp["y_pos"])
                x_te, y_te = finite_xy(te_sp[xcol], te_sp["y_pos"])
                if len(x_tr) >= min_train_points and len(x_te) >= min_test_points:
                    b0, b1, _r2 = ols_fit_1d(x_tr, y_tr)
                    if np.isfinite(b0) and np.isfinite(b1):
                        yhat = b0 + b1 * x_te
                        # test r2
                        ss_res = float(np.sum((y_te - yhat) ** 2))
                        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
                        r2_te = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                        rows.append({
                            "heldout_tag": held, "split": sp, "boundary": "pos_range_hi", "xcol": xcol,
                            "n_train": int(len(x_tr)), "n_test": int(len(x_te)),
                            "intercept": float(b0), "slope": float(b1),
                            "mae": mae(y_te, yhat), "rmse": rmse(y_te, yhat), "r2": float(r2_te),
                            "pearson_y_yhat": float(pearsonr(y_te, yhat).statistic) if len(y_te) >= 3 else np.nan,
                            "spearman_y_yhat": float(spearmanr(y_te, yhat).statistic) if len(y_te) >= 3 else np.nan,
                        })

                # ---- NEG ----
                x_tr, y_tr = finite_xy(tr_sp[xcol], tr_sp["y_neg"])
                x_te, y_te = finite_xy(te_sp[xcol], te_sp["y_neg"])
                if len(x_tr) >= min_train_points and len(x_te) >= min_test_points:
                    b0, b1, _r2 = ols_fit_1d(x_tr, y_tr)
                    if np.isfinite(b0) and np.isfinite(b1):
                        yhat = b0 + b1 * x_te
                        ss_res = float(np.sum((y_te - yhat) ** 2))
                        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
                        r2_te = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                        rows.append({
                            "heldout_tag": held, "split": sp, "boundary": "neg_abs_range_lo", "xcol": xcol,
                            "n_train": int(len(x_tr)), "n_test": int(len(x_te)),
                            "intercept": float(b0), "slope": float(b1),
                            "mae": mae(y_te, yhat), "rmse": rmse(y_te, yhat), "r2": float(r2_te),
                            "pearson_y_yhat": float(pearsonr(y_te, yhat).statistic) if len(y_te) >= 3 else np.nan,
                            "spearman_y_yhat": float(spearmanr(y_te, yhat).statistic) if len(y_te) >= 3 else np.nan,
                        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "LOMO produced no rows.\n"
            "Try lowering --min_train_points / --min_test_points, or check split coverage across tags."
        )
    return out


# -----------------------
# main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged_metrics.csv")
    ap.add_argument("--outdir", default="figures_eval", help="output directory")
    ap.add_argument("--xcols", default="rms0_mean,rms0_p10,rawnorm_mean,rawnorm_p10",
                    help="comma-separated predictors to evaluate")
    ap.add_argument("--splits", default="", help="comma-separated splits (default: all)")
    ap.add_argument("--traits", default="", help="comma-separated traits (default: all)")
    ap.add_argument("--tags", default="", help="comma-separated tags (default: all)")
    ap.add_argument("--no_lomo", action="store_true", help="skip LOMO")
    ap.add_argument("--min_train_points", type=int, default=10, help="min #train points for LOMO fit")
    ap.add_argument("--min_test_points", type=int, default=3, help="min #test points for LOMO eval")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.merged)
    df = prepare_features(df)

    for c in ["tag", "split", "trait"]:
        if c not in df.columns:
            raise ValueError("merged_metrics.csv must contain at least: tag, split, trait")

    xcols = [s.strip() for s in args.xcols.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()] or None
    traits = [s.strip() for s in args.traits.split(",") if s.strip()] or None
    tags   = [s.strip() for s in args.tags.split(",") if s.strip()] or None

    # coverage
    cov = coverage_report(df)
    cov_out = os.path.join(args.outdir, "coverage_report.csv")
    cov.to_csv(cov_out, index=False)
    print("saved:", cov_out)

    # Step 1 & 2
    within = compute_within_model_stats_across_traits(df, xcols=xcols, splits=splits, traits=traits, tags=tags)
    within_out = os.path.join(args.outdir, "within_model_stats.csv")
    within.to_csv(within_out, index=False)
    print("saved:", within_out)

    # Step 3
    if not args.no_lomo:
        lomo = lomo_eval_pool_traits(
            df, xcols=xcols, splits=splits, traits=traits, tags=tags,
            min_train_points=args.min_train_points,
            min_test_points=args.min_test_points,
        )
        lomo_out = os.path.join(args.outdir, "lomo_eval.csv")
        lomo.to_csv(lomo_out, index=False)
        print("saved:", lomo_out)

    print("done.")

if __name__ == "__main__":
    main()
