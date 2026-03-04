#!/usr/bin/env python3
# scripts/09_corr_and_scaling_from_merged.py
"""
merged_metrics.csv から、目的に沿って
(A) 事前に取れる量（near0系, rawnorm系）と境界(range_hi / abs_range_lo)の相関
(B) 境界が「一定の相対介入強度」で決まるか（正規化でモデル間ばらつきが減るか）
を split/trait ごとに集計する。

出力:
- out/corr_pred_summary.csv : 相関 + 単回帰（傾き）
- out/scaling_dispersion.csv: スケーリング検証（std削減率など）
"""

import argparse, os
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:
    raise RuntimeError("scipy が必要です (pip install scipy)") from e


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def safe_div(a, b):
    a = safe_num(a)
    b = safe_num(b)
    out = a / b
    out = out.where(np.isfinite(out))
    return out

def corr_pair(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    idx = x.index.intersection(y.index)
    x = x.loc[idx].astype(float)
    y = y.loc[idx].astype(float)
    if len(x) < 3:
        return np.nan, np.nan
    pr = pearsonr(x, y).statistic
    sr = spearmanr(x, y).statistic
    return float(pr), float(sr)

def simple_ols_slope(x, y):
    """
    y = b0 + b1*x の b1 と R^2 を返す（最小二乗）
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    idx = x.index.intersection(y.index)
    x = x.loc[idx].astype(float).values
    y = y.loc[idx].astype(float).values
    if len(x) < 3:
        return np.nan, np.nan
    X = np.stack([np.ones_like(x), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(beta[1]), float(r2)

def summarize_disp(group, col):
    s = group[col].dropna().astype(float)
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan}
    mean = s.mean()
    std = s.std(ddof=1) if len(s) > 1 else 0.0
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return {"n": int(len(s)), "mean": float(mean), "std": float(std), "median": float(med), "iqr": float(iqr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="merged_metrics.csv")
    ap.add_argument("--out_dir", default="out_09")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.in_csv)

    # --- 目的変数 ---
    # y_pos: range_hi（=正側境界）
    # y_neg: abs_range_lo（=負側境界の絶対値）
    df["y_pos"] = safe_num(df.get("range_hi", np.nan))
    df["y_neg"] = safe_num(df.get("abs_range_lo", np.nan))

    # --- 事前に取れる説明変数（あなたの方針通り）---
    # rawnorm_mean / p10
    df["rawnorm_mean"] = safe_num(df.get("rawnorm_mean", np.nan))
    df["rawnorm_p10"]  = safe_num(df.get("rawnorm_p10", np.nan))

    # near0 RMS（mean_rms_near0_before が無ければ mean_rms0_before も見る）
    if "mean_rms_near0_before" in df.columns:
        df["rms0_mean"] = safe_num(df["mean_rms_near0_before"])
    else:
        df["rms0_mean"] = safe_num(df.get("mean_rms0_before", np.nan))

    if "p10_rms_near0_before" in df.columns:
        df["rms0_p10"] = safe_num(df["p10_rms_near0_before"])
    else:
        df["rms0_p10"] = safe_num(df.get("p10_rms0_before", np.nan))

    # 派生（あなたが挙げた例）
    df["rms0_over_rawmean"] = safe_div(df["rms0_mean"], df["rawnorm_mean"])
    df["rms0_over_rawp10"]  = safe_div(df["rms0_p10"],  df["rawnorm_p10"])

    # --- スケーリング検証用（near0で作る“相対介入強度”候補）---
    # 「境界は一定の相対介入強度で決まるか？」を見るため、境界αを near0量で正規化
    df["pos_eff_over_rms0"] = safe_div(df["y_pos"], df["rms0_mean"])
    df["neg_eff_over_rms0"] = safe_div(df["y_neg"], df["rms0_mean"])

    df["pos_eff_over_rms0_raw"] = safe_div(df["y_pos"], df["rms0_mean"] * df["rawnorm_mean"])
    df["neg_eff_over_rms0_raw"] = safe_div(df["y_neg"], df["rms0_mean"] * df["rawnorm_mean"])

    # --- (A) 相関 + 回帰（split/traitごと）---
    feats = [
        "rawnorm_mean", "rawnorm_p10",
        "rms0_mean", "rms0_p10",
        "rms0_over_rawmean", "rms0_over_rawp10",
    ]

    corr_rows = []
    for (split, trait), g in df.groupby(["split", "trait"], sort=True):
        row = {"split": split, "trait": trait, "n_models": int(g["tag"].nunique())}

        for ycol, yname in [("y_pos", "pos_range_hi"), ("y_neg", "neg_abs_range_lo")]:
            for f in feats:
                pr, sr = corr_pair(g[f], g[ycol])
                row[f"pearson__{yname}__{f}"] = pr
                row[f"spearman__{yname}__{f}"] = sr

            # 追加: 回帰の傾き（例: range_hi ~ (rms0/rawnorm)）
            slope, r2 = simple_ols_slope(g["rms0_over_rawmean"], g[ycol])
            row[f"slope__{yname}__rms0_over_rawmean"] = slope
            row[f"r2__{yname}__rms0_over_rawmean"] = r2

        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    corr_out = os.path.join(args.out_dir, "corr_pred_summary.csv")
    corr_df.to_csv(corr_out, index=False)

    # --- (B) スケーリング則: 正規化でモデル間ばらつきが減るか？ ---
    disp_rows = []
    for (split, trait), g in df.groupby(["split", "trait"], sort=True):
        base = {"split": split, "trait": trait, "n_models": int(g["tag"].nunique())}

        # raw boundary dispersion
        s_pos = summarize_disp(g, "y_pos")
        s_neg = summarize_disp(g, "y_neg")

        # normalized candidates (near0-based)
        s_pos_rms = summarize_disp(g, "pos_eff_over_rms0")
        s_neg_rms = summarize_disp(g, "neg_eff_over_rms0")

        s_pos_rms_raw = summarize_disp(g, "pos_eff_over_rms0_raw")
        s_neg_rms_raw = summarize_disp(g, "neg_eff_over_rms0_raw")

        def put(prefix, s):
            for k, v in s.items():
                base[f"{prefix}__{k}"] = v

        put("pos_raw", s_pos)
        put("neg_raw", s_neg)
        put("pos_norm_rms0", s_pos_rms)
        put("neg_norm_rms0", s_neg_rms)
        put("pos_norm_rms0_raw", s_pos_rms_raw)
        put("neg_norm_rms0_raw", s_neg_rms_raw)

        # std reduction ratios
        base["pos_std_ratio__norm_rms0"] = (base["pos_norm_rms0__std"] / base["pos_raw__std"]) if base["pos_raw__std"] not in [0, np.nan] else np.nan
        base["pos_std_ratio__norm_rms0_raw"] = (base["pos_norm_rms0_raw__std"] / base["pos_raw__std"]) if base["pos_raw__std"] not in [0, np.nan] else np.nan
        base["neg_std_ratio__norm_rms0"] = (base["neg_norm_rms0__std"] / base["neg_raw__std"]) if base["neg_raw__std"] not in [0, np.nan] else np.nan
        base["neg_std_ratio__norm_rms0_raw"] = (base["neg_norm_rms0_raw__std"] / base["neg_raw__std"]) if base["neg_raw__std"] not in [0, np.nan] else np.nan

        disp_rows.append(base)

    disp_df = pd.DataFrame(disp_rows)
    disp_out = os.path.join(args.out_dir, "scaling_dispersion.csv")
    disp_df.to_csv(disp_out, index=False)

    print("Wrote:")
    print(" -", corr_out)
    print(" -", disp_out)


if __name__ == "__main__":
    main()
