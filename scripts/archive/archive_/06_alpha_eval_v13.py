#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_alpha_eval_v14.py

[目的]
v13:
- 12プロンプト全体での安全範囲 (alpha_lo/alpha_hi) を推定
- 制約ごとの合格率/失敗要因/境界決定要因を出力

v14:
- v13互換を維持しつつ、--per_prompt で「プロンプト(sid)別のα境界」を算出できるようにする。
  sid は基本 r["i"]（無ければ r["x"]）で識別。
  各sidごとに baseline(|alpha|最小) の y を y0 として、alpha スイープの is_safe を作り
  0付近から連続合格区間を拡張して lo/hi を決める。

入力:
  --in  <..._with_rms.jsonl>  （行は、i, alpha_total, y, ds_avg などを含む想定）

出力:
  v13(デフォルト): out_root/range/alpha_range_{split}_{trait}.jsonl （1行）
  v14(--per_prompt): out_root/range/alpha_range_{split}_{trait}_per_prompt.jsonl （sidごとに複数行）
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

# scikit-learn (TF-IDF計算用)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    ENGLISH_STOP_WORDS = frozenset({
        "a","an","the","and","or","but","if","then","else","when","at","by","for","with","about","against",
        "between","into","through","during","before","after","above","below","to","from","up","down","in","out",
        "on","off","over","under","again","further","then","once","here","there","when","where","why","how",
        "all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own",
        "same","so","than","too","very","s","t","can","will","just","don","should","now"
    })


# ------------------------------
# I/O Helper
# ------------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

def get_sid(r: Dict[str, Any]) -> str:
    # prompt識別子：基本 i、なければ x、さらに無ければ unknown
    if "i" in r and r["i"] is not None:
        return str(r["i"])
    if "x" in r and r["x"] is not None:
        return str(r["x"])
    return "unknown"

@dataclass
class InferredMeta:
    tag: str
    split: str
    trait: str

def infer_meta_from_path(in_path: str) -> InferredMeta:
    p = Path(in_path)
    parts = list(p.parts)

    tag = "unknown_tag"
    if "exp" in parts:
        try:
            idx = parts.index("exp")
            if idx + 1 < len(parts):
                tag = parts[idx + 1]
        except Exception:
            pass
    if tag == "unknown_tag":
        mtag = re.match(r"([a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*)_(base|instruct)_", p.name)
        if mtag:
            tag = mtag.group(1)

    base = p.name.lower()
    known_traits = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]

    m1 = re.search(r"probe_(base|instruct)_([a-z0-9_]+)\.jsonl$", base)
    if m1:
        split = m1.group(1)
        trait = m1.group(2)
        return InferredMeta(tag=tag, split=split, trait=trait)

    m2 = re.search(r"(base|instruct)_", base)
    split = "unknown_split"
    trait = "unknown_trait"
    if m2:
        split = m2.group(1)
        for kt in known_traits:
            if kt in base:
                trait = kt
                break

    return InferredMeta(tag=tag, split=split, trait=trait)


# ------------------------------
# Metrics
# ------------------------------
_STRIP_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")
_SENT_SPLIT_RE = re.compile(r"[.!?]+|\n+")

def simple_tokenize(text: str) -> List[str]:
    return [t for t in (text or "").strip().split() if t]

def _normalize_token(t: str) -> str:
    return _STRIP_EDGE_PUNCT_RE.sub("", (t or "").lower().strip())

def _normalize_phrase(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _max_consecutive_run(seq: List[str]) -> int:
    best = 0
    cur = 0
    prev = None
    for x in seq:
        if not x:
            continue
        if x == prev:
            cur += 1
        else:
            prev = x
            cur = 1
        if cur > best:
            best = cur
    return best

def max_run_token(text: str) -> int:
    toks = simple_tokenize(text)
    norm = []
    for t in toks:
        nt = _normalize_token(t)
        if nt:
            norm.append(nt)
    return _max_consecutive_run(norm)

def max_run_phrase(text: str, min_phrase_tokens: int = 3) -> int:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text or "") if p.strip()]
    norm_phrases = []
    for p in parts:
        np = _normalize_phrase(p)
        if not np:
            continue
        if len(np.split()) < min_phrase_tokens:
            continue
        norm_phrases.append(np)
    return _max_consecutive_run(norm_phrases)

def distinct2_ratio(tokens_norm: List[str]) -> float:
    if len(tokens_norm) < 2:
        return 0.0
    grams = [tuple(tokens_norm[i:i+2]) for i in range(len(tokens_norm)-1)]
    return (len(set(grams)) / len(grams)) if grams else 0.0

def tfidf_char_cosine(a: str, b: str, ngram_range=(3, 5)) -> float:
    if not SKLEARN_AVAILABLE:
        return 0.0
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=True)
        X = vec.fit_transform([a, b])
        v0, v1 = X[0], X[1]
        denom = (v0.multiply(v0).sum() ** 0.5) * (v1.multiply(v1).sum() ** 0.5)
        return float(v0.multiply(v1).sum() / denom) if denom > 0 else 0.0
    except Exception:
        return 0.0

def jaccard_similarity(a: str, b: str) -> float:
    ta = {_normalize_token(t) for t in simple_tokenize(a) if _normalize_token(t)}
    tb = {_normalize_token(t) for t in simple_tokenize(b) if _normalize_token(t)}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def compute_metrics(y0: str, y: str, use_tfidf: bool, min_phrase_tokens: int = 3) -> Dict[str, float]:
    t0 = simple_tokenize(y0)
    t1 = simple_tokenize(y)

    sem = tfidf_char_cosine(y0, y) if use_tfidf and SKLEARN_AVAILABLE else jaccard_similarity(y0, y)

    t1_norm = []
    for t in t1:
        nt = _normalize_token(t)
        if nt:
            t1_norm.append(nt)

    distinct2_raw = distinct2_ratio(t1_norm)
    t1_cont = [t for t in t1_norm if t not in ENGLISH_STOP_WORDS]
    distinct2_content = distinct2_ratio(t1_cont)

    non_space = [c for c in (y or "") if not c.isspace()]
    punct_ratio = sum(1 for c in non_space if not c.isalnum()) / len(non_space) if non_space else 0.0

    len_ratio = len(t1) / max(1, len(t0))

    mrun_tok = max_run_token(y)
    mrun_phr = max_run_phrase(y, min_phrase_tokens=min_phrase_tokens)

    return {
        "sem_score": sem,
        "len_ratio": len_ratio,
        "distinct2_raw": distinct2_raw,
        "distinct2_content": distinct2_content,
        "punct_ratio": punct_ratio,
        "max_run_token": mrun_tok,
        "max_run_phrase": mrun_phr,
    }


# ------------------------------
# Safety Logic
# ------------------------------
@dataclass
class SafetyConstraints:
    sem_min: float = 0.35
    len_ratio_min: float = 0.2
    len_ratio_max: float = 4.0
    distinct2_min: float = 0.05
    punct_ratio_max: float = 0.85

    max_run_token_max: int = 10
    max_run_phrase_max: int = 3
    min_phrase_tokens: int = 3

def check_safety_details(m: Dict[str, float], c: SafetyConstraints) -> Tuple[bool, Dict[str, bool]]:
    results = {}
    results["sem"] = (m["sem_score"] >= c.sem_min)
    results["len"] = (c.len_ratio_min <= m["len_ratio"] <= c.len_ratio_max)
    results["distinct"] = (m["distinct2_raw"] >= c.distinct2_min)
    results["punct"] = (m["punct_ratio"] <= c.punct_ratio_max)
    results["max_run_token"] = (m["max_run_token"] <= c.max_run_token_max)
    results["max_run_phrase"] = (m["max_run_phrase"] <= c.max_run_phrase_max)

    is_safe = all(results.values())
    return is_safe, results

def primary_fail_reason(details: Dict[str, bool]) -> Optional[str]:
    """
    v13互換の優先順位で primary failure を決める
    """
    if details.get("max_run_phrase") is False:
        return "max_run_phrase"
    if details.get("max_run_token") is False:
        return "max_run_token"
    if details.get("distinct") is False:
        return "distinct"
    if details.get("punct") is False:
        return "punct"
    if details.get("len") is False:
        return "len"
    if details.get("sem") is False:
        return "sem"
    return None

def analyze_safety_range_v13(per_alpha_stats: List[Dict[str, Any]], pass_rate_min: float) -> Dict[str, Any]:
    """
    v13: 12プロンプト全体の pass_rate>=min を safe として連続合格区間を決める
    """
    if not per_alpha_stats:
        return {"lo": None, "hi": None, "rec": None, "reason_pos": None, "reason_neg": None}

    stats = sorted(per_alpha_stats, key=lambda s: float(s["alpha_total"]))
    alphas = [float(s["alpha_total"]) for s in stats]

    safe_map = {float(s["alpha_total"]): (float(s.get("pass_rate", 0.0)) >= pass_rate_min) for s in stats}

    base_alpha = min(alphas, key=abs)

    if not safe_map[base_alpha]:
        return {"lo": None, "hi": None, "rec": None, "reason_pos": "base_failed", "reason_neg": "base_failed"}

    base_idx = alphas.index(base_alpha)

    def find_reason(stat_dict: Dict[str, Any]) -> str:
        candidates = []
        for k in ["sem", "len", "distinct", "punct", "max_run_token", "max_run_phrase"]:
            pr_key = f"pass_rate_{k}" if k in ["sem", "len", "distinct", "punct"] else None
            if pr_key is not None:
                if stat_dict.get(pr_key, 1.0) < pass_rate_min:
                    candidates.append((k, stat_dict.get(pr_key, 1.0)))
        if not candidates:
            return "unknown_mixed"
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    curr_lo = base_idx
    reason_neg = "limit_reached"
    while curr_lo > 0:
        prev_idx = curr_lo - 1
        prev_a = alphas[prev_idx]
        if safe_map[prev_a]:
            curr_lo -= 1
        else:
            reason_neg = find_reason(stats[prev_idx])
            break

    curr_hi = base_idx
    reason_pos = "limit_reached"
    while curr_hi < len(alphas) - 1:
        next_idx = curr_hi + 1
        next_a = alphas[next_idx]
        if safe_map[next_a]:
            curr_hi += 1
        else:
            reason_pos = find_reason(stats[next_idx])
            break

    range_lo = alphas[curr_lo]
    range_hi = alphas[curr_hi]
    rec = range_hi if abs(range_hi) > abs(range_lo) else range_lo

    return {"lo": range_lo, "hi": range_hi, "rec": rec, "reason_pos": reason_pos, "reason_neg": reason_neg}

def analyze_range_from_is_safe(per_alpha_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    v14: sid単位の is_safe 列から連続合格区間を決める
    """
    if not per_alpha_stats:
        return {"lo": None, "hi": None, "rec": None, "reason_pos": None, "reason_neg": None}

    stats = sorted(per_alpha_stats, key=lambda s: float(s["alpha_total"]))
    alphas = [float(s["alpha_total"]) for s in stats]
    safe_map = {float(s["alpha_total"]): bool(s.get("is_safe", False)) for s in stats}

    base_alpha = min(alphas, key=abs)
    if not safe_map[base_alpha]:
        return {"lo": None, "hi": None, "rec": None, "reason_pos": "base_failed", "reason_neg": "base_failed"}

    base_idx = alphas.index(base_alpha)

    def reason_at(idx: int) -> str:
        return stats[idx].get("fail_primary") or "unknown_mixed"

    curr_lo = base_idx
    reason_neg = "limit_reached"
    while curr_lo > 0:
        prev_idx = curr_lo - 1
        prev_a = alphas[prev_idx]
        if safe_map[prev_a]:
            curr_lo -= 1
        else:
            reason_neg = reason_at(prev_idx)
            break

    curr_hi = base_idx
    reason_pos = "limit_reached"
    while curr_hi < len(alphas) - 1:
        next_idx = curr_hi + 1
        next_a = alphas[next_idx]
        if safe_map[next_a]:
            curr_hi += 1
        else:
            reason_pos = reason_at(next_idx)
            break

    range_lo = alphas[curr_lo]
    range_hi = alphas[curr_hi]
    rec = range_hi if abs(range_hi) > abs(range_lo) else range_lo

    return {"lo": range_lo, "hi": range_hi, "rec": rec, "reason_pos": reason_pos, "reason_neg": reason_neg}


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl (alpha sweep with outputs)")
    ap.add_argument("--out_root", required=True, help="output root directory")

    # v14
    ap.add_argument("--per_prompt", action="store_true",
                    help="if set, compute alpha boundary per prompt(sid) instead of pooled over all prompts")
    ap.add_argument("--no_per_alpha_stats", action="store_true",
                    help="if set, drop per_alpha_stats to reduce output size")

    # Safety Config
    ap.add_argument("--pass_rate_min", type=float, default=0.5,
                    help="(pooled mode) safe if pass_rate>=this. (per_prompt mode) mostly irrelevant.")
    ap.add_argument("--sem_min", type=float, default=0.20)
    ap.add_argument("--distinct2_min", type=float, default=0.05)
    ap.add_argument("--len_ratio_min", type=float, default=0.2)
    ap.add_argument("--len_ratio_max", type=float, default=5.0)

    # legacy / keep
    ap.add_argument("--direction", default="increase")
    ap.add_argument("--prefer_small_abs_alpha", action="store_true")
    ap.add_argument("--punct_ratio_max", type=float, default=0.85)
    ap.add_argument("--max_run_token_max", type=int, default=10)
    ap.add_argument("--max_run_phrase_max", type=int, default=3)
    ap.add_argument("--min_phrase_tokens", type=int, default=3)

    args = ap.parse_args()

    #if not SKLEARN_AVAILABLE:
    #    print("[WARN] scikit-learn not found. Using Jaccard instead of TF-IDF.")
    #else:
    #    print("[INFO] Using TF-IDF for semantic check.")

    c = SafetyConstraints(
        sem_min=args.sem_min,
        distinct2_min=args.distinct2_min,
        len_ratio_min=args.len_ratio_min,
        len_ratio_max=args.len_ratio_max,
        punct_ratio_max=args.punct_ratio_max,
        max_run_token_max=args.max_run_token_max,
        max_run_phrase_max=args.max_run_phrase_max,
        min_phrase_tokens=args.min_phrase_tokens,
    )

    meta = infer_meta_from_path(args.inp)
    rows = read_jsonl(args.inp)

    # group by alpha and by sid
    by_alpha = defaultdict(list)
    by_sid = defaultdict(lambda: defaultdict(list))

    for r in rows:
        a = safe_float(r.get("alpha_total"))
        if a is None:
            continue
        sid = get_sid(r)
        by_alpha[a].append(r)
        by_sid[sid][a].append(r)

    if not by_alpha:
        raise ValueError("No valid rows with alpha_total found in input.")

    out_dir = Path(args.out_root) / "range"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------
    # Mode A: pooled (v13-compatible)
    # ------------------------------
    if not args.per_prompt:
        # Baseline: alpha closest to 0
        base_alpha = min(by_alpha.keys(), key=abs)

        # baseline y per sample (sid)
        by_sample_base = {}
        for r in by_alpha[base_alpha]:
            sid = get_sid(r)
            by_sample_base[sid] = r.get("y", "")

        per_alpha_stats = []
        for a, group in sorted(by_alpha.items(), key=lambda kv: float(kv[0])):
            n_total = len(group)
            n_pass = 0

            n_pass_sem = 0
            n_pass_len = 0
            n_pass_distinct = 0
            n_pass_punct = 0
            n_pass_mrt = 0
            n_pass_mrp = 0

            fail_primary = Counter()
            fail_any = Counter()

            ds_values = []

            for r in group:
                sid = get_sid(r)
                y = r.get("y", "")
                y0 = by_sample_base.get(sid, "")

                m = compute_metrics(y0, y, use_tfidf=SKLEARN_AVAILABLE, min_phrase_tokens=c.min_phrase_tokens)
                is_safe, details = check_safety_details(m, c)

                if is_safe:
                    n_pass += 1

                if details["sem"]:
                    n_pass_sem += 1
                if details["len"]:
                    n_pass_len += 1
                if details["distinct"]:
                    n_pass_distinct += 1
                if details["punct"]:
                    n_pass_punct += 1
                if details["max_run_token"]:
                    n_pass_mrt += 1
                if details["max_run_phrase"]:
                    n_pass_mrp += 1

                if not is_safe:
                    failed_keys = [k for k, v in details.items() if not v]
                    for k in failed_keys:
                        fail_any[k] += 1

                    pr = primary_fail_reason(details)
                    if pr is not None:
                        fail_primary[pr] += 1
                    else:
                        fail_primary["unknown"] += 1

                ds = safe_float(r.get("ds_avg"))
                if ds is not None:
                    ds_values.append(ds)

            stat_dict = {
                "alpha_total": float(a),
                "n_total": int(n_total),
                "pass_rate": (n_pass / n_total) if n_total > 0 else 0.0,

                "pass_rate_sem": (n_pass_sem / n_total) if n_total > 0 else 0.0,
                "pass_rate_len": (n_pass_len / n_total) if n_total > 0 else 0.0,
                "pass_rate_distinct": (n_pass_distinct / n_total) if n_total > 0 else 0.0,
                "pass_rate_punct": (n_pass_punct / n_total) if n_total > 0 else 0.0,
                "pass_rate_mrt": (n_pass_mrt / n_total) if n_total > 0 else 0.0,
                "pass_rate_mrp": (n_pass_mrp / n_total) if n_total > 0 else 0.0,

                "fail_primary": dict(fail_primary),
                "fail_any": dict(fail_any),

                "mean": mean(ds_values) if ds_values else 0.0,
                "median": median(ds_values) if ds_values else 0.0,
                "p_pos": (sum(1 for v in ds_values if v > 0) / len(ds_values)) if ds_values else 0.0,
            }
            per_alpha_stats.append(stat_dict)

        res = analyze_safety_range_v13(per_alpha_stats, args.pass_rate_min)

        out_path = out_dir / f"alpha_range_{meta.split}_{meta.trait}.jsonl"
        out_obj = {
            "tag": meta.tag,
            "split": meta.split,
            "trait": meta.trait,
            "constraints": vars(c),
            "pass_rate_min": args.pass_rate_min,

            "alpha_lo": res["lo"],
            "alpha_hi": res["hi"],
            "alpha_recommended": res["rec"],
            "limiting_reason_pos": res["reason_pos"],
            "limiting_reason_neg": res["reason_neg"],
        }
        if not args.no_per_alpha_stats:
            out_obj["per_alpha_stats"] = per_alpha_stats

        write_jsonl(str(out_path), [out_obj])

        print(f"[OK] Safety Analysis v14 (pooled=v13): {out_path}")
        if res["lo"] is not None:
            print(f"     Range: {res['lo']:.3f} ~ {res['hi']:.3f}")
            print(f"     Reason Pos: {res['reason_pos']}")
            print(f"     Reason Neg: {res['reason_neg']}")
        else:
            print("     [WARN] No safe range found.")
        return

    # ------------------------------
    # Mode B: per-prompt (sid-wise)
    # ------------------------------
    out_rows = []
    sids = sorted(by_sid.keys(), key=lambda s: (s == "unknown", s))

    for sid in sids:
        alpha_map = by_sid[sid]
        if not alpha_map:
            continue

        # baseline alpha per sid = alpha closest to 0
        base_alpha = min(alpha_map.keys(), key=abs)

        # baseline y0: if multiple rows exist at base alpha, take first (should be 1)
        base_rows = alpha_map[base_alpha]
        y0 = base_rows[0].get("y", "") if base_rows else ""

        per_alpha_stats = []
        for a in sorted(alpha_map.keys(), key=float):
            rs = alpha_map[a]
            if not rs:
                continue
            # if multiple entries at same (sid, alpha), treat them as multiple trials -> compute "pass_rate" too
            # but default is 1 entry.
            n_total = len(rs)
            n_pass = 0
            n_pass_sem = n_pass_len = n_pass_distinct = n_pass_punct = n_pass_mrt = n_pass_mrp = 0
            fail_primary = Counter()
            fail_any = Counter()

            ds_values = []

            # For per-prompt: evaluate each trial row
            for r in rs:
                y = r.get("y", "")
                m = compute_metrics(y0, y, use_tfidf=SKLEARN_AVAILABLE, min_phrase_tokens=c.min_phrase_tokens)
                is_safe, details = check_safety_details(m, c)

                if is_safe:
                    n_pass += 1
                if details["sem"]:
                    n_pass_sem += 1
                if details["len"]:
                    n_pass_len += 1
                if details["distinct"]:
                    n_pass_distinct += 1
                if details["punct"]:
                    n_pass_punct += 1
                if details["max_run_token"]:
                    n_pass_mrt += 1
                if details["max_run_phrase"]:
                    n_pass_mrp += 1

                if not is_safe:
                    failed_keys = [k for k, v in details.items() if not v]
                    for k in failed_keys:
                        fail_any[k] += 1
                    pr = primary_fail_reason(details)
                    if pr is not None:
                        fail_primary[pr] += 1
                    else:
                        fail_primary["unknown"] += 1

                ds = safe_float(r.get("ds_avg"))
                if ds is not None:
                    ds_values.append(ds)

            pass_rate = (n_pass / n_total) if n_total > 0 else 0.0
            is_safe_final = pass_rate >= args.pass_rate_min  # trialsが複数ある場合にも対応

            # 代表の fail_primary（最頻）
            fail_primary_mode = None
            if not is_safe_final and fail_primary:
                fail_primary_mode = fail_primary.most_common(1)[0][0]

            stat = {
                "alpha_total": float(a),
                "n_total": int(n_total),
                "pass_rate": float(pass_rate),
                "is_safe": bool(is_safe_final),
                "fail_primary": fail_primary_mode,

                "pass_rate_sem": (n_pass_sem / n_total) if n_total > 0 else 0.0,
                "pass_rate_len": (n_pass_len / n_total) if n_total > 0 else 0.0,
                "pass_rate_distinct": (n_pass_distinct / n_total) if n_total > 0 else 0.0,
                "pass_rate_punct": (n_pass_punct / n_total) if n_total > 0 else 0.0,
                "pass_rate_mrt": (n_pass_mrt / n_total) if n_total > 0 else 0.0,
                "pass_rate_mrp": (n_pass_mrp / n_total) if n_total > 0 else 0.0,

                "fail_primary_counts": dict(fail_primary),
                "fail_any_counts": dict(fail_any),

                "mean": mean(ds_values) if ds_values else 0.0,
                "median": median(ds_values) if ds_values else 0.0,
                "p_pos": (sum(1 for v in ds_values if v > 0) / len(ds_values)) if ds_values else 0.0,
            }
            if not args.no_per_alpha_stats:
                # 代表1本目の metrics/details を残したい場合はここで
                # ただし巨大になるのでデフォルトは落とす運用もあり
                pass
            per_alpha_stats.append(stat)

        # sid-wise boundary from is_safe
        res = analyze_range_from_is_safe(per_alpha_stats)

        out_obj = {
            "tag": meta.tag,
            "split": meta.split,
            "trait": meta.trait,
            "sid": sid,

            "constraints": vars(c),
            "pass_rate_min": args.pass_rate_min,  # trials複数のとき意味が出る

            "alpha_lo": res["lo"],
            "alpha_hi": res["hi"],
            "alpha_recommended": res["rec"],
            "limiting_reason_pos": res["reason_pos"],
            "limiting_reason_neg": res["reason_neg"],
        }

        if not args.no_per_alpha_stats:
            out_obj["per_alpha_stats"] = per_alpha_stats

        out_rows.append(out_obj)

    out_path = out_dir / f"alpha_range_{meta.split}_{meta.trait}_per_prompt.jsonl"
    write_jsonl(str(out_path), out_rows)

    #print(f"[OK] Safety Analysis v14 (per_prompt): {out_path}")
    #print(f"     prompts(sid): {len(out_rows)}")
    # quick summary
    vals_hi = [r["alpha_hi"] for r in out_rows if r.get("alpha_hi") is not None]
    vals_lo = [r["alpha_lo"] for r in out_rows if r.get("alpha_lo") is not None]
    if vals_hi and vals_lo:
        def _stats(v):
            v = sorted(float(x) for x in v)
            return (min(v), max(v), sum(v)/len(v))
        lo_min, lo_max, lo_mean = _stats(vals_lo)
        hi_min, hi_max, hi_mean = _stats(vals_hi)
        print(f"     alpha_lo: min={lo_min:.3f} max={lo_max:.3f} mean={lo_mean:.3f}")
        print(f"     alpha_hi: min={hi_min:.3f} max={hi_max:.3f} mean={hi_mean:.3f}")
    else:
        print("     [WARN] Some prompts had no safe range (base_failed etc.).")

if __name__ == "__main__":
    main()
