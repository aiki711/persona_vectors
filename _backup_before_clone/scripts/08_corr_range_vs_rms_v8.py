#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_corr_range_vs_rms_v8.py

境界α(=range_lo/hi付近)のRMS と、α=0基準RMS の両方を併記して集計・相関比較する版。

目的:
- (診断) 境界αでのRMS: 「そのαが崩壊につながりそうか」をモニタする材料
- (予測) α=0 RMS: 「介入前に安全域を見積もれるか」を検証

入力:
- range_summary.csv (scripts/07 の出力; kind=range の行)
- probe *with_rms.jsonl (scripts/01_run_probe_with_rms.py の出力)

出力:
- merged_metrics.csv
  - boundary: inv_mean_rms_before_at_{lo,hi}, inv_p10_rms_before_at_{lo,hi}
  - alpha0 : inv_mean_rms0_before, inv_p10_rms0_before
- corr_summary.csv
  - POS: range_hi vs inv_* (boundary@hi / alpha0)
  - NEG: abs(range_lo) vs inv_* (boundary@lo / alpha0)
  - mean/p10 それぞれ Pearson/Spearman

実行例:
python scripts/08_corr_range_vs_rms_v8.py \
  --range_csv exp/mistral_7b/asst_pairwise_results/selected_range/_summary/range_summary.csv \
  --probe_jsonl_glob "exp/mistral_7b/asst_pairwise_results/*with_rms.jsonl" \
  --out_dir exp/mistral_7b/asst_pairwise_results/selected_range/_corr \
  --corr_group split \
  --min_n 3 \
  --make_plots

corr_group:
- split_trait (default): split×trait
- split: traitを混ぜて splitごと（tagが1種類のときに相関が出る）
- all: 全部混ぜて1つ
"""

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KNOWN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
KNOWN_SPLITS = ["base", "instruct"]

def load_rawnorm_summaries(npz_paths: List[str], known_tags: List[str]) -> Dict[tuple, Dict[str, float]]:
    """
    rawnorm npz（key: "L|trait", value: scalar）を読み、
    (tag, split, trait) -> {"rawnorm_mean":..., "rawnorm_p10":...} を返す。
    集計は L>0 のみ（embedding相当のL=0を除外）。
    """
    out: Dict[tuple, Dict[str, float]] = {}
    for p in npz_paths:
        meta = infer_meta_from_path(p, known_tags)
        tag, split = meta["tag"], meta["split"]
        try:
            bank = np.load(p)
        except Exception:
            continue

        # trait -> list of norms across layers
        tmp: Dict[str, List[float]] = {t: [] for t in KNOWN_TRAITS}
        for k in bank.files:
            try:
                Ls, tr = k.split("|")
                L = int(Ls)
            except Exception:
                continue
            if tr not in tmp:
                continue
            if L <= 0:
                continue
            val = float(np.array(bank[k]).reshape(-1)[0])
            if np.isfinite(val):
                tmp[tr].append(val)

        for tr, vals in tmp.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            key = (tag, split, tr)
            out[key] = {
                "rawnorm_mean": float(np.mean(arr)),
                "rawnorm_p10": float(np.percentile(arr, 10)),
            }
    return out

def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def infer_meta_from_path(path_str: str, known_tags: List[str]) -> Dict[str, str]:
    p = Path(path_str)
    name = p.name.lower()

    split = "unknown"
    # 1) ファイル名に "axes_base_" / "axes_instruct_" がある場合は最優先
    if "axes_base_" in name or "axes_base-" in name:
        split = "base"
    elif "axes_instruct_" in name or "axes_instruct-" in name:
        split = "instruct"
    else:
        # 2) 既存の境界正規表現フォールバック
        for s in KNOWN_SPLITS:
            if re.search(rf"(^|[^a-z0-9]){s}([^a-z0-9]|$)", name):
                split = s
                break


    trait = "unknown"
    for t in KNOWN_TRAITS:
        if t in name:
            trait = t
            break

    tag = "unknown"
    for cand in sorted(known_tags, key=lambda z: -len(z)):
        if cand.lower() in name:
            tag = cand
            break

    if tag == "unknown":
        parts_lower = [x.lower() for x in p.parts]
        if "exp" in parts_lower:
            i = parts_lower.index("exp")
            if i + 1 < len(p.parts):
                tag = p.parts[i + 1]

    return {"tag": tag, "split": split, "trait": trait}

def _aligned_layer_keys(row: Dict[str, Any]) -> List[str]:
    """rms_by_layer のキーを layers と整合するように並べた rms_key のリストを返す。"""
    rms_by_layer = row.get("rms_by_layer", {}) or {}
    if not isinstance(rms_by_layer, dict) or not rms_by_layer:
        return []

    rms_keys = sorted([str(k) for k in rms_by_layer.keys()], key=lambda x: int(x) if str(x).isdigit() else x)
    rms_key_ints = [int(k) for k in rms_keys if k.isdigit()]
    if not rms_key_ints:
        return []

    layers = row.get("layers", None)
    if not layers:
        return rms_keys

    try:
        layer_ints = [int(x) for x in layers]
    except Exception:
        return rms_keys

    min_layer, max_layer = min(layer_ints), max(layer_ints)
    min_rms, max_rms = min(rms_key_ints), max(rms_key_ints)

    # layers=1..N, rms=0..N-1
    if min_layer == 1 and min_rms == 0 and (max_layer - 1) == max_rms:
        out = []
        for L in layer_ints:
            rk = str(L - 1)
            if rk in rms_by_layer:
                out.append(rk)
        return out

    # layers=0..N-1, rms=0..N-1
    if min_layer == 0 and min_rms == 0 and max_layer == max_rms:
        out = []
        for L in layer_ints:
            rk = str(L)
            if rk in rms_by_layer:
                out.append(rk)
        return out

    # otherwise: intersection (fallback)
    layer_set = set(str(x) for x in layer_ints)
    rms_set = set(rms_keys)
    common = sorted(layer_set & rms_set, key=lambda x: int(x) if x.isdigit() else x)
    return common if common else rms_keys

def extract_layer_rms_list(row: Dict[str, Any], which: str = "before") -> Optional[List[float]]:
    """rms_by_layer から層ごとの rms_{which}.mean のリストを取り出す（finite & >0 のみ）。"""
    rms_by_layer = row.get("rms_by_layer", None)
    if not isinstance(rms_by_layer, dict) or not rms_by_layer:
        return None

    key = "rms_before" if which == "before" else "rms_after"
    rms_keys = _aligned_layer_keys(row)

    vals: List[float] = []
    for rk in rms_keys:
        stats = rms_by_layer.get(rk, {})
        if key in stats and isinstance(stats[key], dict) and "mean" in stats[key]:
            try:
                v = float(stats[key]["mean"])
            except Exception:
                continue
            # ★ここが重要：NaN/inf/<=0 を落とす
            if math.isfinite(v) and v > 0:
                vals.append(v)

    return vals if vals else None

def summarize_rms_features(row: Dict[str, Any], which: str = "before") -> Dict[str, float]:
    """1レコードから mean / p10 / n_valid を作る（finite のみで計算）。"""
    vals = extract_layer_rms_list(row, which=which)

    # scalar fallback（昔のログ互換）
    if not vals:
        if "rms" in row:
            try:
                x = float(row["rms"])
                if math.isfinite(x) and x > 0:
                    return {"mean": x, "p10": x, "n": 1.0}
            except Exception:
                pass
        return {}

    arr = np.asarray(vals, dtype=float)
    # 念のためここでも finite のみにする（二重安全）
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return {}

    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "n": float(arr.size),
    }

def _alpha_bucket(rows: List[Dict[str, Any]]) -> Dict[float, List[Dict[str, Any]]]:
    out: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        a = safe_float(r.get("alpha_total", None))  # ★0をデフォルトにしない
        if not math.isfinite(a):                    # ★NaN/infは捨てる（増殖防止）
            continue
        a = float(round(a, 12))                     # ★任意：浮動小数の誤差対策
        out.setdefault(a, []).append(r)
    return out

def _avg_feature_per_alpha(
    alpha_rows: Dict[float, List[Dict[str, Any]]],
    which: str
) -> Tuple[Dict[float, float], Dict[float, float], Dict[float, float]]:
    """alpha -> avg(meanRMS), alpha -> avg(p10RMS), alpha -> avg(n_valid_layers)"""
    mean_map: Dict[float, List[float]] = {}
    p10_map: Dict[float, List[float]] = {}
    n_map: Dict[float, List[float]] = {}

    for a, rs in alpha_rows.items():
        for r in rs:
            feats = summarize_rms_features(r, which=which)
            if not feats:
                continue

            m = float(feats["mean"])
            p = float(feats["p10"])
            n = float(feats.get("n", 0))

            # ★NaN/inf を混ぜない
            if math.isfinite(m) and m > 0:
                mean_map.setdefault(a, []).append(m)
            if math.isfinite(p) and p > 0:
                p10_map.setdefault(a, []).append(p)
            if math.isfinite(n) and n > 0:
                n_map.setdefault(a, []).append(n)

    avg_mean = {a: float(np.mean(v)) for a, v in mean_map.items() if v}
    avg_p10  = {a: float(np.mean(v)) for a, v in p10_map.items() if v}
    avg_n    = {a: float(np.mean(v)) for a, v in n_map.items() if v}
    return avg_mean, avg_p10, avg_n

def process_jsonl_rms(fpath: str, target_lo: float, target_hi: float) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return {}

    alpha_vals = [safe_float(r.get("alpha_total", None)) for r in rows]
    n_total = len(alpha_vals)
    n_bad = sum([not math.isfinite(a) for a in alpha_vals])
    alpha_fin = sorted(set([round(a,12) for a in alpha_vals if math.isfinite(a)]))

    print("[DBG]", fpath)
    print(" total_rows:", n_total, "bad_alpha:", n_bad, "unique_alpha:", len(alpha_fin))
    print(" min/max:", (alpha_fin[0], alpha_fin[-1]) if alpha_fin else None)
    print(" has_exact0:", any(abs(a-0.0)<1e-9 for a in alpha_fin))
    print(" head:", alpha_fin[:10])

    alpha_rows = _alpha_bucket(rows)
    print(" bucket_keys:", sorted(alpha_rows.keys())[:10], "...")
    print(" count_at_0:", len(alpha_rows.get(0.0, [])))

    avg_mean_before, avg_p10_before, avg_n_before = _avg_feature_per_alpha(alpha_rows, which="before")
    if not avg_mean_before:
        return {}

    alphas = sorted(avg_mean_before.keys())
    if not alphas:
        return {}

    def closest(target: float) -> float:
        return min(alphas, key=lambda x: abs(x - target))

    found_lo = closest(target_lo)
    found_hi = closest(target_hi)
    print(" target_lo/hi:", target_lo, target_hi,
      " found_lo/hi:", found_lo, found_hi,
      " delta:", found_lo-target_lo, found_hi-target_hi)


    # --- 0は「厳密に存在するか」を優先 ---
    tol = 1e-9
    has_exact0 = any(abs(a - 0.0) < tol for a in alphas)
    found_near0 = closest(0.0)

    # 厳密0の値（なければNaN）
    found_0 = 0.0 if has_exact0 else float("nan")

    eps = 1e-12

    def safe_inv(x: float) -> float:
        return float(1.0 / (x + eps)) if (x == x and x > 0) else float("nan")

    mean_lo = avg_mean_before.get(found_lo, float("nan"))
    mean_hi = avg_mean_before.get(found_hi, float("nan"))
    p10_lo  = avg_p10_before.get(found_lo, float("nan"))
    p10_hi  = avg_p10_before.get(found_hi, float("nan"))

    mean_0 = avg_mean_before.get(found_0, float("nan")) if has_exact0 else float("nan")
    p10_0  = avg_p10_before.get(found_0, float("nan"))  if has_exact0 else float("nan")

    mean_near0 = avg_mean_before.get(found_near0, float("nan"))
    p10_near0  = avg_p10_before.get(found_near0, float("nan"))

    return {
        "actual_alpha_lo": float(found_lo),
        "actual_alpha_hi": float(found_hi),
        "delta_alpha_lo": float(found_lo - target_lo),
        "delta_alpha_hi": float(found_hi - target_hi),

        # 0について：厳密/近傍を分離
        "has_alpha0": float(1.0 if has_exact0 else 0.0),
        "actual_alpha0": float(found_0) if has_exact0 else float("nan"),
        "actual_alpha_near0": float(found_near0),
        "delta_alpha_near0": float(found_near0 - 0.0),

        # boundary
        "mean_rms_before_at_lo": float(mean_lo),
        "mean_rms_before_at_hi": float(mean_hi),
        "p10_rms_before_at_lo": float(p10_lo),
        "p10_rms_before_at_hi": float(p10_hi),
        "inv_mean_rms_before_at_lo": safe_inv(mean_lo),
        "inv_mean_rms_before_at_hi": safe_inv(mean_hi),
        "inv_p10_rms_before_at_lo": safe_inv(p10_lo),
        "inv_p10_rms_before_at_hi": safe_inv(p10_hi),

        # alpha0 (厳密0)
        "mean_rms0_before": float(mean_0),
        "p10_rms0_before": float(p10_0),
        "inv_mean_rms0_before": safe_inv(mean_0),
        "inv_p10_rms0_before": safe_inv(p10_0),

        # near0（必要ならこちらで比較）
        "mean_rms_near0_before": float(mean_near0),
        "p10_rms_near0_before": float(p10_near0),
        "inv_mean_rms_near0_before": safe_inv(mean_near0),
        "inv_p10_rms_near0_before": safe_inv(p10_near0),
    }

def build_merged_metrics(range_csv_path: str, jsonl_paths: List[str], rawnorm_map: Optional[Dict[tuple, Dict[str, float]]] = None) -> pd.DataFrame:
    df_range = pd.read_csv(range_csv_path)
    if "kind" in df_range.columns:
        df_range = df_range[df_range["kind"].astype(str).str.lower() == "range"].copy()

    for c in ["range_lo", "range_hi", "recommended", "rec_median", "rec_mean", "rec_p_pos"]:
        if c in df_range.columns:
            df_range[c] = pd.to_numeric(df_range[c], errors="coerce")

    known_tags = sorted(df_range["tag"].dropna().unique().tolist())
    if not known_tags:
        raise SystemExit("[ERROR] No tags found in range_summary.csv")

    range_map: Dict[tuple, Any] = {}
    for _, row in df_range.iterrows():
        key = (str(row.get("tag")), str(row.get("split")), str(row.get("trait")))
        range_map[key] = row

    out_rows = []
    print(f"[INFO] Processing {len(jsonl_paths)} probe files...")

    for fpath in jsonl_paths:
        meta = infer_meta_from_path(fpath, known_tags)
        tag, split, trait = meta["tag"], meta["split"], meta["trait"]
        key = (tag, split, trait)

        if key not in range_map:
            cands = [k for k in range_map.keys() if k[1] == split and k[2] == trait]
            if len(cands) == 1:
                key = cands[0]
                tag = key[0]
            else:
                continue

        rr = range_map[key]
        range_lo = safe_float(rr.get("range_lo"))
        range_hi = safe_float(rr.get("range_hi"))
        if math.isnan(range_lo) or math.isnan(range_hi):
            continue

        rms_stats = process_jsonl_rms(str(fpath), range_lo, range_hi)
        if not rms_stats:
            continue

        merged = dict(rr)
        merged.update(rms_stats)
        if rawnorm_map is not None:
            k2 = (tag, split, trait)
            rn = rawnorm_map.get(k2, {})
            merged["rawnorm_mean"] = rn.get("rawnorm_mean", float("nan"))
            merged["rawnorm_p10"]  = rn.get("rawnorm_p10",  float("nan"))
        else:
            merged["rawnorm_mean"] = float("nan")
            merged["rawnorm_p10"]  = float("nan")
        
        merged["path"] = str(fpath)
        merged["tag"] = tag
        merged["split"] = split
        merged["trait"] = trait
        merged["abs_range_lo"] = abs(range_lo)
        merged["range_width"] = range_hi - range_lo
        out_rows.append(merged)

                # =========================
        # alpha normalization (new)
        # =========================
        # NOTE:
        # - range_hi は POS 側
        # - abs_range_lo は NEG 側
        # - rawnorm_map が無いと NaN になります（--rawnorm_npz_glob を渡す）
        rn_mean = safe_float(merged.get("rawnorm_mean"))
        rn_p10  = safe_float(merged.get("rawnorm_p10"))

        inv_hi_mean = safe_float(merged.get("inv_mean_rms_before_at_hi"))
        inv_hi_p10  = safe_float(merged.get("inv_p10_rms_before_at_hi"))
        inv_lo_mean = safe_float(merged.get("inv_mean_rms_before_at_lo"))
        inv_lo_p10  = safe_float(merged.get("inv_p10_rms_before_at_lo"))

        # POS: range_hi の正規化
        merged["range_hi_eff_rawnorm_mean"] = float(merged["range_hi"] * rn_mean) if math.isfinite(rn_mean) else float("nan")
        merged["range_hi_eff_rawnorm_p10"]  = float(merged["range_hi"] * rn_p10)  if math.isfinite(rn_p10)  else float("nan")

        merged["range_hi_eff_rms_hi_mean"]  = float(merged["range_hi"] * inv_hi_mean) if math.isfinite(inv_hi_mean) else float("nan")
        merged["range_hi_eff_rms_hi_p10"]   = float(merged["range_hi"] * inv_hi_p10)  if math.isfinite(inv_hi_p10)  else float("nan")

        # NEG: abs(range_lo) の正規化
        merged["abs_range_lo_eff_rawnorm_mean"] = float(merged["abs_range_lo"] * rn_mean) if math.isfinite(rn_mean) else float("nan")
        merged["abs_range_lo_eff_rawnorm_p10"]  = float(merged["abs_range_lo"] * rn_p10)  if math.isfinite(rn_p10)  else float("nan")

        merged["abs_range_lo_eff_rms_lo_mean"]  = float(merged["abs_range_lo"] * inv_lo_mean) if math.isfinite(inv_lo_mean) else float("nan")
        merged["abs_range_lo_eff_rms_lo_p10"]   = float(merged["abs_range_lo"] * inv_lo_p10)  if math.isfinite(inv_lo_p10)  else float("nan")

    return pd.DataFrame(out_rows)

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 2:
        return float("nan")
    xr = pd.Series(x[m]).rank(method="average").to_numpy()
    yr = pd.Series(y[m]).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])

def corr_summary(df: pd.DataFrame, corr_group: str, min_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if corr_group == "split_trait":
        gb_cols = ["split", "trait"]
    elif corr_group == "split":
        gb_cols = ["split"]
    elif corr_group == "all":
        gb_cols = []
    else:
        raise ValueError(f"Unknown corr_group: {corr_group}")

    grouped = [("all", df)] if not gb_cols else list(df.groupby(gb_cols))

    recs = []
    for key, sub in grouped:
        sub = sub.copy()
        n = len(sub)

        row = {"n_samples": int(n)}
        if gb_cols:
            if len(gb_cols) == 1:
                row[gb_cols[0]] = key
            else:
                for c, v in zip(gb_cols, key):
                    row[c] = v
        else:
            row["group"] = "all"

        if n < min_n:
            recs.append(row)
            continue

        def col(name: str) -> np.ndarray:
            # 列がなければ NaN ベクトル
            if name not in sub.columns:
                return np.full(n, np.nan, dtype=float)
            return sub[name].to_numpy(float)

        y_pos = col("range_hi")
        y_neg = col("abs_range_lo")

        # boundary
        xb_pos_mean = col("inv_mean_rms_before_at_hi")
        xb_neg_mean = col("inv_mean_rms_before_at_lo")
        xb_pos_p10  = col("inv_p10_rms_before_at_hi")
        xb_neg_p10  = col("inv_p10_rms_before_at_lo")

        # alpha0
        x0_mean = col("inv_mean_rms0_before")
        x0_p10  = col("inv_p10_rms0_before")

        # --- rawnorm (単体) ---
        x_rn_mean = col("rawnorm_mean")
        x_rn_p10  = col("rawnorm_p10")

        # ===== 正規化α（eff alpha）を目的変数にした相関 =====
        y_pos_rn = col("range_hi_eff_rawnorm_mean")
        y_neg_rn = col("abs_range_lo_eff_rawnorm_mean")
        y_pos_rms = col("range_hi_eff_rms_hi_mean")
        y_neg_rms = col("abs_range_lo_eff_rms_lo_mean")

        row.update({
            # ===== 元の相関（boundary / alpha0） =====
            "pearson_pos_boundary_mean": _pearson(y_pos, xb_pos_mean),
            "spearman_pos_boundary_mean": _spearman(y_pos, xb_pos_mean),
            "pearson_pos_boundary_p10": _pearson(y_pos, xb_pos_p10),
            "spearman_pos_boundary_p10": _spearman(y_pos, xb_pos_p10),

            "pearson_neg_boundary_mean": _pearson(y_neg, xb_neg_mean),
            "spearman_neg_boundary_mean": _spearman(y_neg, xb_neg_mean),
            "pearson_neg_boundary_p10": _pearson(y_neg, xb_neg_p10),
            "spearman_neg_boundary_p10": _spearman(y_neg, xb_neg_p10),

            "pearson_pos_alpha0_mean": _pearson(y_pos, x0_mean),
            "spearman_pos_alpha0_mean": _spearman(y_pos, x0_mean),
            "pearson_pos_alpha0_p10": _pearson(y_pos, x0_p10),
            "spearman_pos_alpha0_p10": _spearman(y_pos, x0_p10),

            "pearson_neg_alpha0_mean": _pearson(y_neg, x0_mean),
            "spearman_neg_alpha0_mean": _spearman(y_neg, x0_mean),
            "pearson_neg_alpha0_p10": _pearson(y_neg, x0_p10),
            "spearman_neg_alpha0_p10": _spearman(y_neg, x0_p10),

            # POS: range_hi vs rawnorm_*
            "pearson_pos_rawnorm_mean": _pearson(y_pos, x_rn_mean),
            "spearman_pos_rawnorm_mean": _spearman(y_pos, x_rn_mean),
            "pearson_pos_rawnorm_p10": _pearson(y_pos, x_rn_p10),
            "spearman_pos_rawnorm_p10": _spearman(y_pos, x_rn_p10),

            # NEG: abs(range_lo) vs rawnorm_*
            "pearson_neg_rawnorm_mean": _pearson(y_neg, x_rn_mean),
            "spearman_neg_rawnorm_mean": _spearman(y_neg, x_rn_mean),
            "pearson_neg_rawnorm_p10": _pearson(y_neg, x_rn_p10),
            "spearman_neg_rawnorm_p10": _spearman(y_neg, x_rn_p10),

            # (A) POS: 正規化α(rawnorm) vs boundary
            "pearson_pos_eff_rawnorm_vs_boundary_mean": _pearson(y_pos_rn, xb_pos_mean),
            "spearman_pos_eff_rawnorm_vs_boundary_mean": _spearman(y_pos_rn, xb_pos_mean),

            # (B) NEG: 正規化α(rawnorm) vs boundary
            "pearson_neg_eff_rawnorm_vs_boundary_mean": _pearson(y_neg_rn, xb_neg_mean),
            "spearman_neg_eff_rawnorm_vs_boundary_mean": _spearman(y_neg_rn, xb_neg_mean),

            # (C) POS: 正規化α(RMS) vs rawnorm
            "pearson_pos_eff_rms_vs_rawnorm_mean": _pearson(y_pos_rms, x_rn_mean),
            "spearman_pos_eff_rms_vs_rawnorm_mean": _spearman(y_pos_rms, x_rn_mean),

            # (D) NEG: 正規化α(RMS) vs rawnorm
            "pearson_neg_eff_rms_vs_rawnorm_mean": _pearson(y_neg_rms, x_rn_mean),
            "spearman_neg_eff_rms_vs_rawnorm_mean": _spearman(y_neg_rms, x_rn_mean),
        })
        recs.append(row)

    return pd.DataFrame(recs)

def maybe_make_plots(df: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    for (split, trait), sub in df.groupby(["split", "trait"]):
        # POS
        for mode, xcol, fname in [
            ("boundary_mean", "inv_mean_rms_before_at_hi", f"scatter_pos_boundary_mean_{split}_{trait}.png"),
            ("alpha0_mean", "inv_mean_rms0_before", f"scatter_pos_alpha0_mean_{split}_{trait}.png"),
            ("boundary_p10", "inv_p10_rms_before_at_hi", f"scatter_pos_boundary_p10_{split}_{trait}.png"),
            ("alpha0_p10", "inv_p10_rms0_before", f"scatter_pos_alpha0_p10_{split}_{trait}.png"),
        ]:
            fig = plt.figure()
            plt.scatter(sub[xcol].to_numpy(float), sub["range_hi"].to_numpy(float))
            plt.xlabel(xcol)
            plt.ylabel("range_hi")
            plt.title(f"POS {mode}: {split}-{trait}")
            for _, r in sub.iterrows():
                plt.annotate(str(r["tag"]), (float(r[xcol]), float(r["range_hi"])), fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=200)
            plt.close(fig)

        # NEG
        for mode, xcol, fname in [
            ("boundary_mean", "inv_mean_rms_before_at_lo", f"scatter_neg_boundary_mean_{split}_{trait}.png"),
            ("alpha0_mean", "inv_mean_rms0_before", f"scatter_neg_alpha0_mean_{split}_{trait}.png"),
            ("boundary_p10", "inv_p10_rms_before_at_lo", f"scatter_neg_boundary_p10_{split}_{trait}.png"),
            ("alpha0_p10", "inv_p10_rms0_before", f"scatter_neg_alpha0_p10_{split}_{trait}.png"),
        ]:
            fig = plt.figure()
            plt.scatter(sub[xcol].to_numpy(float), sub["abs_range_lo"].to_numpy(float))
            plt.xlabel(xcol)
            plt.ylabel("abs(range_lo)")
            plt.title(f"NEG {mode}: {split}-{trait}")
            for _, r in sub.iterrows():
                plt.annotate(str(r["tag"]), (float(r[xcol]), float(r["abs_range_lo"])), fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=200)
            plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range_csv", required=True)
    ap.add_argument("--probe_jsonl_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--make_plots", action="store_true")
    ap.add_argument("--corr_group", choices=["split_trait", "split", "all"], default="split_trait")
    ap.add_argument("--min_n", type=int, default=3)
    ap.add_argument("--rawnorm_npz_glob", default=None,
                help="optional glob for *_rawnorms.npz (from 00_prepare_vectors.py)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.probe_jsonl_glob, recursive=True))
    if not paths:
        raise SystemExit("[ERROR] No files matched the glob pattern.")

    rawnorm_map = None
    if args.rawnorm_npz_glob:
        raw_paths = sorted(glob.glob(args.rawnorm_npz_glob, recursive=True))
        if raw_paths:
            # range_csvから known_tags を読むために一度 df_range を読む
            df_range_tmp = pd.read_csv(args.range_csv)
            if "kind" in df_range_tmp.columns:
                df_range_tmp = df_range_tmp[df_range_tmp["kind"].astype(str).str.lower() == "range"].copy()
            known_tags = sorted(df_range_tmp["tag"].dropna().unique().tolist())
            rawnorm_map = load_rawnorm_summaries(raw_paths, known_tags)
            print(f"[INFO] loaded rawnorm summaries: {len(rawnorm_map)} entries from {len(raw_paths)} files")
        else:
            print("[WARN] rawnorm_npz_glob matched no files. Continue without rawnorm.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_merged = build_merged_metrics(args.range_csv, paths, rawnorm_map=rawnorm_map)
    if df_merged.empty:
        raise SystemExit("[ERROR] Merged DataFrame is empty.")

    merged_path = out_dir / "merged_metrics.csv"
    df_merged.to_csv(merged_path, index=False)
    print(f"[OK] Saved merged metrics: {merged_path}")

    sizes = df_merged.groupby(["split", "trait"]).size().reset_index(name="n")
    print("\n[INFO] group sizes (split×trait):")
    print(sizes.to_string(index=False))

    df_corr = corr_summary(df_merged, args.corr_group, args.min_n)
    corr_path = out_dir / "corr_summary.csv"
    df_corr.to_csv(corr_path, index=False)
    print(f"\n[OK] Saved correlation summary: {corr_path}")
    print("\n--- Correlation Results (boundary vs alpha0; mean vs p10) ---")
    print(df_corr.to_string(index=False))

    if args.make_plots:
        maybe_make_plots(df_merged, out_dir)
        print("[OK] Saved scatter plots: scatter_(pos|neg)_(boundary|alpha0)_(mean|p10)_*.png")

if __name__ == "__main__":
    main()
