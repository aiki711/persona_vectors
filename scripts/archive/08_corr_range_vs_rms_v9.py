#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_corr_range_vs_rms_v9.py

目的:
- (A) モデル単位: range_summary.csv(kind=range) と *_with_rms.jsonl を結合して相関を見る (従来v8)
- (B) プロンプト単位: alpha_range_*_per_prompt.jsonl と *_with_rms.jsonl を sid (= i) で結合して相関を見る (新)

入力:
(A) model-mode:
  --range_csv                 : range_summary.csv (kind=range を想定)
  --probe_jsonl_glob          : exp/<tag>/.../*with_rms.jsonl
  --rawnorm_npz_glob (optional): exp/<tag>/axes_*_rawnorms.npz

(B) per-prompt-mode:
  --per_prompt_jsonl_glob     : exp/<tag>/.../alpha_range_*_per_prompt.jsonl
  --probe_jsonl_glob          : exp/<tag>/.../*with_rms.jsonl

出力:
  out_dir/
    model/
      merged_metrics_model.csv
      corr_summary_model.csv
      scatter_*.png (optional)
    per_prompt/
      merged_metrics_per_prompt.csv
      corr_summary_per_prompt.csv
      scatter_*.png (optional)

注意:
- per_prompt_summary.csv は sid を持たないので、このスクリプトでは使いません。
  per-prompt は alpha_range_*_per_prompt.jsonl を直接読むのが正解です。
"""

from __future__ import annotations

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

# -------------------------
# Utilities
# -------------------------
def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))

def infer_meta_from_path(path_str: str, known_tags: Optional[List[str]] = None) -> Dict[str, str]:
    """
    path/filename から tag, split, trait を推定する。
    - tag: exp/<tag>/... があればそれ
    - split: base/instruct
    - trait: openness/... が含まれていれば
    - known_tags があれば、それを優先して filename からも拾う
    """
    p = Path(path_str)
    name = p.name.lower()

    split = "unknown"
    if "axes_base_" in name or "axes_base-" in name:
        split = "base"
    elif "axes_instruct_" in name or "axes_instruct-" in name:
        split = "instruct"
    else:
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
    parts_lower = [x.lower() for x in p.parts]
    if "exp" in parts_lower:
        i = parts_lower.index("exp")
        if i + 1 < len(p.parts):
            tag = p.parts[i + 1]

    # filenameに含まれるtag優先（known_tagsがある場合）
    if known_tags:
        for cand in sorted(known_tags, key=lambda z: -len(z)):
            if cand.lower() in name:
                tag = cand
                break

    return {"tag": tag, "split": split, "trait": trait}

# -------------------------
# RMS extraction (same spirit as v8)
# -------------------------
def _aligned_layer_keys(row: Dict[str, Any]) -> List[str]:
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

    # fallback: intersection
    layer_set = set(str(x) for x in layer_ints)
    rms_set = set(rms_keys)
    common = sorted(layer_set & rms_set, key=lambda x: int(x) if x.isdigit() else x)
    return common if common else rms_keys

def extract_layer_rms_list(row: Dict[str, Any], which: str = "before") -> Optional[List[float]]:
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
            if math.isfinite(v) and v > 0:
                vals.append(v)

    return vals if vals else None

def summarize_rms_features_from_row(row: Dict[str, Any], which: str = "before") -> Dict[str, float]:
    vals = extract_layer_rms_list(row, which=which)

    # scalar fallback
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
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return {}

    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "n": float(arr.size),
    }

def inv(x: float, eps: float = 1e-12) -> float:
    if not is_finite(x) or x <= 0:
        return float("nan")
    return float(1.0 / (x + eps))

# -------------------------
# Probe -> sid(i) -> alpha0 features
# -------------------------
def load_probe_alpha0_features(probe_path: str) -> Dict[str, Dict[str, float]]:
    """
    1つの *_with_rms.jsonl から:
      sid (= i) ごとに、alpha=0（厳密優先、なければ近傍）の rms_before 特徴量を作る
    return: sid -> feature dict
    """
    rows: List[Dict[str, Any]] = []
    with open(probe_path, "r", encoding="utf-8") as f:
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

    # sid -> list of rows
    by_sid: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = r.get("i", None)
        if sid is None:
            continue
        sid = str(sid)
        by_sid.setdefault(sid, []).append(r)

    out: Dict[str, Dict[str, float]] = {}

    for sid, rs in by_sid.items():
        # available alphas
        alphas = []
        for r in rs:
            a = safe_float(r.get("alpha_total", None))
            if is_finite(a):
                alphas.append(float(round(a, 12)))
        if not alphas:
            continue
        uniq = sorted(set(alphas))

        tol = 1e-9
        has_exact0 = any(abs(a - 0.0) < tol for a in uniq)
        target = 0.0 if has_exact0 else min(uniq, key=lambda x: abs(x - 0.0))

        # rows at target alpha
        tgt_rows = [r for r in rs if is_finite(safe_float(r.get("alpha_total", None))) and abs(float(round(safe_float(r.get("alpha_total")), 12)) - target) < tol]
        if not tgt_rows:
            # fallback: closest by absolute diff
            def _dist(r):
                a = safe_float(r.get("alpha_total", None))
                return abs(a - target) if is_finite(a) else 1e18
            tgt_rows = [min(rs, key=_dist)]

        # aggregate across possibly multiple rows (just in case)
        means, p10s, ns = [], [], []
        for r in tgt_rows:
            feats = summarize_rms_features_from_row(r, which="before")
            if not feats:
                continue
            means.append(feats["mean"])
            p10s.append(feats["p10"])
            ns.append(feats.get("n", float("nan")))

        if not means:
            continue

        m = float(np.mean(means))
        p = float(np.mean(p10s)) if p10s else float("nan")
        n = float(np.mean([x for x in ns if is_finite(x)])) if any(is_finite(x) for x in ns) else float("nan")

        out[sid] = {
            "alpha0_used": float(target),
            "has_exact_alpha0": float(1.0 if has_exact0 else 0.0),
            "mean_rms0_before": m,
            "p10_rms0_before": p,
            "inv_mean_rms0_before": inv(m),
            "inv_p10_rms0_before": inv(p),
            "n_valid_layers_mean": n,
        }

    return out

# -------------------------
# Per-prompt range loader
# -------------------------
def load_per_prompt_range_rows(path: str) -> List[Dict[str, Any]]:
    """
    alpha_range_*_per_prompt.jsonl
      - 期待: 1行=1 sid の dict (tag/split/trait/sid/alpha_lo/alpha_hi/...)
      - もし将来まとめ形式になっても耐えるように、軽くフォールバック
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # case1: per line record
            if isinstance(obj, dict) and "sid" in obj and ("alpha_lo" in obj or "alpha_hi" in obj):
                rows.append(obj)
                continue

            # case2: bundled list
            if isinstance(obj, dict):
                for k in ["per_prompt", "per_prompt_ranges", "rows"]:
                    if k in obj and isinstance(obj[k], list):
                        for r in obj[k]:
                            if isinstance(r, dict) and "sid" in r:
                                # inherit tag/split/trait if missing
                                if "tag" not in r and "tag" in obj:
                                    r["tag"] = obj["tag"]
                                if "split" not in r and "split" in obj:
                                    r["split"] = obj["split"]
                                if "trait" not in r and "trait" in obj:
                                    r["trait"] = obj["trait"]
                                rows.append(r)
                        break

    return rows

# -------------------------
# Correlation helpers
# -------------------------
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

# -------------------------
# Model-mode (v8 compatible)
# -------------------------
def load_rawnorm_summaries(npz_paths: List[str], known_tags: List[str]) -> Dict[tuple, Dict[str, float]]:
    out: Dict[tuple, Dict[str, float]] = {}
    for p in npz_paths:
        meta = infer_meta_from_path(p, known_tags)
        tag, split = meta["tag"], meta["split"]
        try:
            bank = np.load(p)
        except Exception:
            continue

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

def build_merged_metrics_model(range_csv_path: str, jsonl_paths: List[str], rawnorm_map: Optional[Dict[tuple, Dict[str, float]]] = None) -> pd.DataFrame:
    df_range = pd.read_csv(range_csv_path)

    # kind=range があるならそれを優先（v8踏襲）
    if "kind" in df_range.columns:
        kinds = set(df_range["kind"].astype(str).str.lower().unique().tolist())
        if "range" in kinds:
            df_range = df_range[df_range["kind"].astype(str).str.lower() == "range"].copy()
        else:
            # rangeが無いなら model-mode として成立しない（per_prompt_summary 等）
            raise SystemExit(
                "[ERROR] range_csv has no kind=range rows. "
                "If you want per-prompt correlation, pass --per_prompt_jsonl_glob (alpha_range_*_per_prompt.jsonl)."
            )

    for c in ["range_lo", "range_hi"]:
        if c in df_range.columns:
            df_range[c] = pd.to_numeric(df_range[c], errors="coerce")

    known_tags = sorted(df_range["tag"].dropna().unique().tolist()) if "tag" in df_range.columns else []
    if not known_tags:
        raise SystemExit("[ERROR] No tags found in range_summary.csv (after filtering kind=range).")

    range_map: Dict[tuple, Any] = {}
    for _, row in df_range.iterrows():
        key = (str(row.get("tag")), str(row.get("split")), str(row.get("trait")))
        range_map[key] = row

    out_rows = []
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
        if not is_finite(range_lo) or not is_finite(range_hi):
            continue

        # v8: boundary alphaでのRMS、alpha0のRMS…が必要だったが、
        # ここは「境界alphaでのRMS」用途なので、v8互換の簡易版を残す:
        # -> per alpha bucketして closest(range_lo/hi) の meanRMS を取る
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
            continue

        # alpha -> list of rows
        alpha_rows: Dict[float, List[Dict[str, Any]]] = {}
        for r in rows:
            a = safe_float(r.get("alpha_total", None))
            if not is_finite(a):
                continue
            a = float(round(a, 12))
            alpha_rows.setdefault(a, []).append(r)

        if not alpha_rows:
            continue

        alphas = sorted(alpha_rows.keys())
        def closest(target: float) -> float:
            return min(alphas, key=lambda x: abs(x - target))

        a_lo = closest(range_lo)
        a_hi = closest(range_hi)

        # aggregate mean rms at lo/hi
        def agg_mean_rms(a: float) -> float:
            vals = []
            for r in alpha_rows.get(a, []):
                feats = summarize_rms_features_from_row(r, which="before")
                if feats and is_finite(feats["mean"]):
                    vals.append(feats["mean"])
            return float(np.mean(vals)) if vals else float("nan")

        mean_lo = agg_mean_rms(a_lo)
        mean_hi = agg_mean_rms(a_hi)

        merged = dict(rr)
        merged.update({
            "actual_alpha_lo": a_lo,
            "actual_alpha_hi": a_hi,
            "mean_rms_before_at_lo": mean_lo,
            "mean_rms_before_at_hi": mean_hi,
            "inv_mean_rms_before_at_lo": inv(mean_lo),
            "inv_mean_rms_before_at_hi": inv(mean_hi),
        })

        # alpha0 features (sid無視して全体で近傍0) を一応残す
        # ただし model-mode は従来通り「事前に取れる量」を使う想定なので、
        # ここは「ファイル全体の alpha0 平均」として扱う。
        # (必要なら後で改良可)
        a0 = 0.0 if any(abs(a-0.0) < 1e-9 for a in alphas) else closest(0.0)
        vals0 = []
        for r in alpha_rows.get(a0, []):
            feats = summarize_rms_features_from_row(r, which="before")
            if feats and is_finite(feats["mean"]):
                vals0.append(feats["mean"])
        mean0 = float(np.mean(vals0)) if vals0 else float("nan")
        merged["mean_rms0_before"] = mean0
        merged["inv_mean_rms0_before"] = inv(mean0)

        if rawnorm_map is not None:
            rn = rawnorm_map.get((tag, split, trait), {})
            merged["rawnorm_mean"] = rn.get("rawnorm_mean", float("nan"))
            merged["rawnorm_p10"] = rn.get("rawnorm_p10", float("nan"))
        else:
            merged["rawnorm_mean"] = float("nan")
            merged["rawnorm_p10"] = float("nan")

        merged["tag"] = tag
        merged["split"] = split
        merged["trait"] = trait
        merged["abs_range_lo"] = abs(range_lo)
        merged["range_width"] = range_hi - range_lo
        merged["path"] = str(fpath)
        out_rows.append(merged)

    return pd.DataFrame(out_rows)

def corr_summary_model(df: pd.DataFrame, corr_group: str, min_n: int) -> pd.DataFrame:
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

        y_pos = sub["range_hi"].to_numpy(float) if "range_hi" in sub.columns else np.full(n, np.nan)
        y_neg = sub["abs_range_lo"].to_numpy(float) if "abs_range_lo" in sub.columns else np.full(n, np.nan)

        xb_pos = sub["inv_mean_rms_before_at_hi"].to_numpy(float) if "inv_mean_rms_before_at_hi" in sub.columns else np.full(n, np.nan)
        xb_neg = sub["inv_mean_rms_before_at_lo"].to_numpy(float) if "inv_mean_rms_before_at_lo" in sub.columns else np.full(n, np.nan)

        x0 = sub["inv_mean_rms0_before"].to_numpy(float) if "inv_mean_rms0_before" in sub.columns else np.full(n, np.nan)

        row.update({
            "pearson_pos_boundary": _pearson(y_pos, xb_pos),
            "spearman_pos_boundary": _spearman(y_pos, xb_pos),
            "pearson_neg_boundary": _pearson(y_neg, xb_neg),
            "spearman_neg_boundary": _spearman(y_neg, xb_neg),
            "pearson_pos_alpha0": _pearson(y_pos, x0),
            "spearman_pos_alpha0": _spearman(y_pos, x0),
            "pearson_neg_alpha0": _pearson(y_neg, x0),
            "spearman_neg_alpha0": _spearman(y_neg, x0),
        })
        recs.append(row)

    return pd.DataFrame(recs)

# -------------------------
# Per-prompt mode
# -------------------------
def build_merged_metrics_per_prompt(per_prompt_jsonl_paths: List[str], probe_jsonl_paths: List[str]) -> pd.DataFrame:
    """
    per_prompt_jsonl_paths: alpha_range_*_per_prompt.jsonl の集合
    probe_jsonl_paths     : *_with_rms.jsonl の集合

    各 (tag, split, trait) で
      sid -> boundary(hi/lo)
      sid -> alpha0 rms features
    を結合して、12点/グループの行にする
    """
    # probe features map: (tag, split, trait) -> sid -> feats
    probe_map: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]] = {}
    for p in probe_jsonl_paths:
        meta = infer_meta_from_path(p, known_tags=None)
        key = (meta["tag"], meta["split"], meta["trait"])
        feats_by_sid = load_probe_alpha0_features(p)
        if feats_by_sid:
            probe_map[key] = feats_by_sid

    out_rows: List[Dict[str, Any]] = []

    for pp in per_prompt_jsonl_paths:
        rows = load_per_prompt_range_rows(pp)
        if not rows:
            continue

        # group meta from first row
        tag = str(rows[0].get("tag", "unknown"))
        split = str(rows[0].get("split", "unknown"))
        trait = str(rows[0].get("trait", "unknown"))
        key = (tag, split, trait)

        feats_by_sid = probe_map.get(key, {})
        if not feats_by_sid:
            # たまに trait/split 推定ズレがあるので、fallbackで「tag一致 & split/trait一致」を探す
            candidates = [k for k in probe_map.keys() if k[0] == tag and k[1] == split and k[2] == trait]
            if len(candidates) == 1:
                feats_by_sid = probe_map[candidates[0]]

        for r in rows:
            sid = r.get("sid", None)
            if sid is None:
                continue
            sid = str(sid)

            alpha_lo = safe_float(r.get("alpha_lo", None))
            alpha_hi = safe_float(r.get("alpha_hi", None))
            if not is_finite(alpha_lo) or not is_finite(alpha_hi):
                continue

            y_pos = alpha_hi
            y_neg = abs(alpha_lo)

            feats = feats_by_sid.get(sid, None)
            if feats is None:
                # sidが一致しない（今回はintersection=12なので基本起きない）
                continue

            merged = {
                "tag": tag,
                "split": split,
                "trait": trait,
                "sid": sid,
                "alpha_lo": alpha_lo,
                "alpha_hi": alpha_hi,
                "abs_alpha_lo": y_neg,
                "range_width": alpha_hi - alpha_lo,
            }
            merged.update(feats)  # alpha0 rms features
            out_rows.append(merged)

    return pd.DataFrame(out_rows)

def corr_summary_per_prompt(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """
    12点（sid）での within-model 相関を、(tag, split, trait) ごとに出す
    """
    if df.empty:
        return pd.DataFrame()

    recs = []
    for (tag, split, trait), sub in df.groupby(["tag", "split", "trait"]):
        n = len(sub)
        row = {"tag": tag, "split": split, "trait": trait, "n_prompts": int(n)}
        if n < min_n:
            recs.append(row)
            continue

        y_pos = sub["alpha_hi"].to_numpy(float)
        y_neg = sub["abs_alpha_lo"].to_numpy(float)

        x0m = sub["inv_mean_rms0_before"].to_numpy(float) if "inv_mean_rms0_before" in sub.columns else np.full(n, np.nan)
        x0p = sub["inv_p10_rms0_before"].to_numpy(float) if "inv_p10_rms0_before" in sub.columns else np.full(n, np.nan)

        row.update({
            "pearson_pos_alpha0_mean": _pearson(y_pos, x0m),
            "spearman_pos_alpha0_mean": _spearman(y_pos, x0m),
            "pearson_neg_alpha0_mean": _pearson(y_neg, x0m),
            "spearman_neg_alpha0_mean": _spearman(y_neg, x0m),

            "pearson_pos_alpha0_p10": _pearson(y_pos, x0p),
            "spearman_pos_alpha0_p10": _spearman(y_pos, x0p),
            "pearson_neg_alpha0_p10": _pearson(y_neg, x0p),
            "spearman_neg_alpha0_p10": _spearman(y_neg, x0p),

            "alpha0_exact_ratio": float(np.mean(sub["has_exact_alpha0"].to_numpy(float))) if "has_exact_alpha0" in sub.columns else float("nan"),
        })
        recs.append(row)

    return pd.DataFrame(recs).sort_values(["tag", "split", "trait"]).reset_index(drop=True)

# -------------------------
# Plotters (optional)
# -------------------------
def maybe_make_plots_per_prompt(df: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    for (tag, split, trait), sub in df.groupby(["tag", "split", "trait"]):
        # POS
        for xcol, ycol, fname in [
            ("inv_mean_rms0_before", "alpha_hi", f"scatter_pp_pos_alpha0mean_{tag}_{split}_{trait}.png"),
            ("inv_p10_rms0_before", "alpha_hi", f"scatter_pp_pos_alpha0p10_{tag}_{split}_{trait}.png"),
        ]:
            if xcol not in sub.columns:
                continue
            fig = plt.figure()
            plt.scatter(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float))
            plt.xlabel(xcol)
            plt.ylabel(ycol)
            plt.title(f"Per-prompt POS: {tag}-{split}-{trait}")
            # annotate sid lightly
            for _, r in sub.iterrows():
                plt.annotate(str(r["sid"]), (float(r[xcol]), float(r[ycol])), fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=200)
            plt.close(fig)

        # NEG
        for xcol, ycol, fname in [
            ("inv_mean_rms0_before", "abs_alpha_lo", f"scatter_pp_neg_alpha0mean_{tag}_{split}_{trait}.png"),
            ("inv_p10_rms0_before", "abs_alpha_lo", f"scatter_pp_neg_alpha0p10_{tag}_{split}_{trait}.png"),
        ]:
            if xcol not in sub.columns:
                continue
            fig = plt.figure()
            plt.scatter(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float))
            plt.xlabel(xcol)
            plt.ylabel(ycol)
            plt.title(f"Per-prompt NEG: {tag}-{split}-{trait}")
            for _, r in sub.iterrows():
                plt.annotate(str(r["sid"]), (float(r[xcol]), float(r[ycol])), fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=200)
            plt.close(fig)

def maybe_make_plots_model(df: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    if "split" not in df.columns or "trait" not in df.columns:
        return

    for (split, trait), sub in df.groupby(["split", "trait"]):
        if "inv_mean_rms0_before" in sub.columns and "range_hi" in sub.columns:
            fig = plt.figure()
            plt.scatter(sub["inv_mean_rms0_before"].to_numpy(float), sub["range_hi"].to_numpy(float))
            plt.xlabel("inv_mean_rms0_before")
            plt.ylabel("range_hi")
            plt.title(f"Model POS alpha0: {split}-{trait}")
            for _, r in sub.iterrows():
                plt.annotate(str(r.get("tag", "")), (float(r["inv_mean_rms0_before"]), float(r["range_hi"])), fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"scatter_model_pos_alpha0_{split}_{trait}.png", dpi=200)
            plt.close(fig)

        if "inv_mean_rms0_before" in sub.columns and "abs_range_lo" in sub.columns:
            fig = plt.figure()
            plt.scatter(sub["inv_mean_rms0_before"].to_numpy(float), sub["abs_range_lo"].to_numpy(float))
            plt.xlabel("inv_mean_rms0_before")
            plt.ylabel("abs(range_lo)")
            plt.title(f"Model NEG alpha0: {split}-{trait}")
            for _, r in sub.iterrows():
                plt.annotate(str(r.get("tag", "")), (float(r["inv_mean_rms0_before"]), float(r["abs_range_lo"])), fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"scatter_model_neg_alpha0_{split}_{trait}.png", dpi=200)
            plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_jsonl_glob", required=True, help="glob for *_with_rms.jsonl")
    ap.add_argument("--out_dir", required=True)

    # model-mode
    ap.add_argument("--range_csv", default=None, help="range_summary.csv (kind=range) for model-level correlation")
    ap.add_argument("--corr_group", choices=["split_trait", "split", "all"], default="split_trait")
    ap.add_argument("--rawnorm_npz_glob", default=None, help="optional glob for *_rawnorms.npz")

    # per-prompt-mode
    ap.add_argument("--per_prompt_jsonl_glob", default=None, help="glob for alpha_range_*_per_prompt.jsonl")

    ap.add_argument("--min_n", type=int, default=3)
    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    probe_paths = sorted(glob.glob(args.probe_jsonl_glob, recursive=True))
    if not probe_paths:
        raise SystemExit("[ERROR] No probe jsonl matched --probe_jsonl_glob")

    did_any = False

    # -------------------------
    # (B) per-prompt-mode
    # -------------------------
    if args.per_prompt_jsonl_glob:
        pp_paths = sorted(glob.glob(args.per_prompt_jsonl_glob, recursive=True))
        if not pp_paths:
            raise SystemExit("[ERROR] No per_prompt jsonl matched --per_prompt_jsonl_glob")

        out_dir = out_root / "per_prompt"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_pp = build_merged_metrics_per_prompt(pp_paths, probe_paths)
        if df_pp.empty:
            raise SystemExit(
                "[ERROR] per-prompt merged df is empty. "
                "Check meta matching (tag/split/trait) and sid=i join."
            )

        merged_path = out_dir / "merged_metrics_per_prompt.csv"
        df_pp.to_csv(merged_path, index=False)
        print(f"[OK] Saved: {merged_path}")

        df_corr = corr_summary_per_prompt(df_pp, min_n=args.min_n)
        corr_path = out_dir / "corr_summary_per_prompt.csv"
        df_corr.to_csv(corr_path, index=False)
        print(f"[OK] Saved: {corr_path}")
        print(df_corr.to_string(index=False))

        if args.make_plots:
            maybe_make_plots_per_prompt(df_pp, out_dir)
            print(f"[OK] Saved per-prompt plots to: {out_dir}")

        did_any = True

    # -------------------------
    # (A) model-mode
    # -------------------------
    if args.range_csv:
        out_dir = out_root / "model"
        out_dir.mkdir(parents=True, exist_ok=True)

        rawnorm_map = None
        if args.rawnorm_npz_glob:
            raw_paths = sorted(glob.glob(args.rawnorm_npz_glob, recursive=True))
            if raw_paths:
                df_range_tmp = pd.read_csv(args.range_csv)
                if "kind" in df_range_tmp.columns:
                    df_range_tmp = df_range_tmp[df_range_tmp["kind"].astype(str).str.lower() == "range"].copy()
                known_tags = sorted(df_range_tmp["tag"].dropna().unique().tolist()) if "tag" in df_range_tmp.columns else []
                if known_tags:
                    rawnorm_map = load_rawnorm_summaries(raw_paths, known_tags)
                    print(f"[INFO] loaded rawnorm summaries: {len(rawnorm_map)} entries from {len(raw_paths)} files")
                else:
                    print("[WARN] could not infer known_tags from range_csv; skip rawnorm_map")
            else:
                print("[WARN] rawnorm_npz_glob matched no files; continue without rawnorm")

        df_model = build_merged_metrics_model(args.range_csv, probe_paths, rawnorm_map=rawnorm_map)
        if df_model.empty:
            raise SystemExit("[ERROR] model merged df is empty. Check range_csv and probe_jsonl_glob.")

        merged_path = out_dir / "merged_metrics_model.csv"
        df_model.to_csv(merged_path, index=False)
        print(f"[OK] Saved: {merged_path}")

        df_corr = corr_summary_model(df_model, corr_group=args.corr_group, min_n=args.min_n)
        corr_path = out_dir / "corr_summary_model.csv"
        df_corr.to_csv(corr_path, index=False)
        print(f"[OK] Saved: {corr_path}")
        print(df_corr.to_string(index=False))

        if args.make_plots:
            maybe_make_plots_model(df_model, out_dir)
            print(f"[OK] Saved model plots to: {out_dir}")

        did_any = True

    if not did_any:
        raise SystemExit(
            "[ERROR] Nothing to do. Provide at least one of:\n"
            "  --per_prompt_jsonl_glob (per-prompt)\n"
            "  --range_csv (model-level)\n"
        )

if __name__ == "__main__":
    main()
