#!/usr/bin/env python3
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def load_probe_jsonl(path: str, model_tag: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            i      = rec["i"]
            trait  = rec["trait"]
            alpha  = float(rec["alpha_total"])
            s_by   = rec["s_by_layer"]  # dict: { "1": score1, ... }
            for L_str, s in s_by.items():
                L = int(L_str)
                rows.append({
                    "model": model_tag,
                    "trait": trait,
                    "sample_id": i,
                    "alpha": alpha,
                    "layer": L,
                    "score": float(s),
                })
    return pd.DataFrame(rows)

def fit_line(x: np.ndarray, y: np.ndarray):
    """return (slope, intercept) or (nan, nan)"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.std(x) == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", required=True)
    ap.add_argument("--instr_json", required=True)
    ap.add_argument("--out_csv", default="probe_slopes_from_logs.csv")

    ap.add_argument("--pooling", choices=["asst", "last", "mean"], default=None,
                    help="Pooling mode used when generating the probe logs (optional meta info)")
    ap.add_argument("--axis_mode", choices=["cluster", "pairwise"], default=None,
                    help="Axis construction mode used when generating the probe logs (optional meta info)")

    # 追加：alpha=0 判定の許容誤差（0.0が float 誤差でズレると baseline が消えるのを防ぐ）
    ap.add_argument("--alpha0_tol", type=float, default=1e-12)

    args = ap.parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_base  = load_probe_jsonl(args.base_json,  model_tag="base")
    df_instr = load_probe_jsonl(args.instr_json, model_tag="instruct")
    df = pd.concat([df_base, df_instr], ignore_index=True)

    # 1) baseline (alpha≈0) を取ってくる
    is_a0 = np.isclose(df["alpha"].to_numpy(dtype=float), 0.0, atol=args.alpha0_tol)
    df0 = df[is_a0].rename(columns={"score": "score0"})

    key_cols = ["model", "trait", "sample_id", "layer"]
    df_merged = pd.merge(
        df,
        df0[key_cols + ["score0"]],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )

    # baseline が取れない行は delta が NaN になるので落とす（または warn したいならここで集計）
    df_merged["delta_score"] = df_merged["score"] - df_merged["score0"]
    df_merged = df_merged[np.isfinite(df_merged["delta_score"].to_numpy(dtype=float))].copy()

    records = []
    for (model, trait, layer), df_grp in df_merged.groupby(["model", "trait", "layer"]):
        # α ごとに平均Δscore
        df_alpha = df_grp.groupby("alpha", as_index=False)["delta_score"].mean()
        df_alpha = df_alpha.sort_values("alpha")

        alphas = df_alpha["alpha"].to_numpy(dtype=float)
        deltas = df_alpha["delta_score"].to_numpy(dtype=float)

        num_alpha = int(len(df_alpha))
        if num_alpha < 2:
            slope_raw, intercept_raw = np.nan, np.nan
            slope_a01, intercept_a01 = np.nan, np.nan
            a_min, a_max, a_rng = np.nan, np.nan, np.nan
        else:
            a_min = float(np.min(alphas))
            a_max = float(np.max(alphas))
            a_rng = float(a_max - a_min)

            # 既存：raw slope（delta ~ alpha）
            slope_raw, intercept_raw = fit_line(alphas, deltas)

            # 追加：alpha を [0,1] 正規化して slope（delta ~ alpha01）
            if a_rng > 0 and np.isfinite(a_rng):
                alpha01 = (alphas - a_min) / a_rng
                slope_a01, intercept_a01 = fit_line(alpha01, deltas)
            else:
                slope_a01, intercept_a01 = np.nan, np.nan

        rec = {
            "model": model,
            "trait": trait,
            "layer": int(layer),

            # 既存（維持）
            "slope_delta_score_vs_alpha": slope_raw,
            "intercept_delta_score_vs_alpha": intercept_raw,
            "num_alpha": num_alpha,

            # 追加（αスケール差を吸収）
            "alpha_min": a_min,
            "alpha_max": a_max,
            "alpha_range": a_rng,
            "slope_delta_score_vs_alpha01": slope_a01,
            "intercept_delta_score_vs_alpha01": intercept_a01,
        }

        if args.pooling is not None:
            rec["pooling"] = args.pooling
        if args.axis_mode is not None:
            rec["axis_mode"] = args.axis_mode

        records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    print(f"Saved slopes to {out_path} (rows={len(out_df)})")

if __name__ == "__main__":
    main()
