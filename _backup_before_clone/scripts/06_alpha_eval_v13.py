#!/usr/bin/env python3
"""
06_alpha_eval_v13.py

[目的]
v12の「安全範囲(Safety Range)の特定」に加え、
「なぜその範囲に制限されたのか（何が原因で不合格になったか）」を詳細に分析する。

[v13の追加機能]
1. 制約ごとの合格率 (pass_rate_sem, pass_rate_len, ...)
2. 失敗要因のカウント (fail_primary, fail_any)
3. 境界決定要因の推定 (limiting_reason_pos, limiting_reason_neg)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

# scikit-learn (TF-IDF計算用)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback stopwords
    ENGLISH_STOP_WORDS = frozenset({"a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"})

# ------------------------------
# I/O Helper
# ------------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_float(x: Any) -> Optional[float]:
    try: return float(x) if x is not None else None
    except: return None

@dataclass
class InferredMeta:
    tag: str
    split: str
    trait: str

def infer_meta_from_path(in_path: str) -> InferredMeta:
    p = Path(in_path)
    parts = list(p.parts)

    # tag: exp/<tag>/... があればそこ、なければ filename 先頭
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

    # 1) probe_<split>_<trait>.jsonl
    m1 = re.search(r"probe_(base|instruct)_([a-z0-9_]+)\.jsonl$", base)
    if m1:
        split = m1.group(1)
        trait = m1.group(2)
        return InferredMeta(tag=tag, split=split, trait=trait)

    # 2) <tag>_<split>_<trait>_with_rms.jsonl など
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

def _normalize_phrase(s: str) -> str:
    s = (s or "").lower()
    # 記号を空白へ（英数字と空白以外を落とす）
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
        # 短すぎる文は除外（誤爆防止）
        if len(np.split()) < min_phrase_tokens:
            continue
        norm_phrases.append(np)
    return _max_consecutive_run(norm_phrases)

def simple_tokenize(text: str) -> List[str]:
    return [t for t in (text or "").strip().split() if t]

def _normalize_token(t: str) -> str:
    return _STRIP_EDGE_PUNCT_RE.sub("", (t or "").lower().strip())

def content_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        nt = _normalize_token(t)
        if nt and nt not in ENGLISH_STOP_WORDS: out.append(nt)
    return out

def tfidf_char_cosine(a: str, b: str, ngram_range=(3, 5)) -> float:
    if not SKLEARN_AVAILABLE: return 0.0
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b: return 0.0
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=True)
        X = vec.fit_transform([a, b])
        v0, v1 = X[0], X[1]
        denom = (v0.multiply(v0).sum() ** 0.5) * (v1.multiply(v1).sum() ** 0.5)
        return float(v0.multiply(v1).sum() / denom) if denom > 0 else 0.0
    except: return 0.0

def jaccard_similarity(a: str, b: str) -> float:
    ta = {_normalize_token(t) for t in simple_tokenize(a) if _normalize_token(t)}
    tb = {_normalize_token(t) for t in simple_tokenize(b) if _normalize_token(t)}
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def distinct2_ratio(tokens_norm: List[str]) -> float:
    # stopword を落とさない版でも、短すぎると判定不能なので 0.0 に寄せる
    if len(tokens_norm) < 2:
        return 0.0
    grams = [tuple(tokens_norm[i:i+2]) for i in range(len(tokens_norm)-1)]
    return (len(set(grams)) / len(grams)) if grams else 0.0

def compute_metrics(y0: str, y: str, use_tfidf: bool, min_phrase_tokens: int = 3) -> Dict[str, float]:
    t0 = simple_tokenize(y0)
    t1 = simple_tokenize(y)

    # 1. Meaning Preservation
    sem = tfidf_char_cosine(y0, y) if use_tfidf and SKLEARN_AVAILABLE else jaccard_similarity(y0, y)

    # normalize tokens once
    t1_norm = []
    for t in t1:
        nt = _normalize_token(t)
        if nt:
            t1_norm.append(nt)

    # 2-a. Distinct-2 (RAW: stopword KEEP)  ← 判定に使う
    distinct2_raw = distinct2_ratio(t1_norm)

    # 2-b. Distinct-2 (CONTENT: stopword DROP) ← 参考値として保持
    t1_cont = [t for t in t1_norm if t not in ENGLISH_STOP_WORDS]
    distinct2_content = distinct2_ratio(t1_cont)

    # 3. Punctuation Ratio
    non_space = [c for c in (y or "") if not c.isspace()]
    punct_ratio = sum(1 for c in non_space if not c.isalnum()) / len(non_space) if non_space else 0.0

    # 4. Length Ratio
    len_ratio = len(t1) / max(1, len(t0))

    # 5. Repetition
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
    """
    各制約ごとの合否(True/False)と、総合合否を返す
    """
    results = {}
    
    # 1. Sem
    results["sem"] = (m["sem_score"] >= c.sem_min)
    
    # 2. Len
    results["len"] = (c.len_ratio_min <= m["len_ratio"] <= c.len_ratio_max)
    
    # 3. Distinct
    results["distinct"] = (m["distinct2_raw"] >= c.distinct2_min)
    
    # 4. Punct
    results["punct"] = (m["punct_ratio"] <= c.punct_ratio_max)

    # 5. Max Run Token
    results["max_run_token"] = (m["max_run_token"] <= c.max_run_token_max)
    
    # 6. Max Run Phrase
    results["max_run_phrase"] = (m["max_run_phrase"] <= c.max_run_phrase_max)
    
    is_safe = all(results.values())
    return is_safe, results

def analyze_safety_range_v13(
    per_alpha_stats: List[Dict[str, Any]], 
    pass_rate_min: float
) -> Dict[str, Any]:
    """
    連続合格区間を探し、なぜそこで止まったか(limiting_reason)を推定する
    """
    if not per_alpha_stats: 
        return {"lo": None, "hi": None, "rec": None, "reason_pos": None, "reason_neg": None}
    
    # Sort
    stats = sorted(per_alpha_stats, key=lambda s: float(s["alpha_total"]))
    alphas = [float(s["alpha_total"]) for s in stats]
    
    # Pass map (overall)
    safe_map = {
        s["alpha_total"]: (s["pass_rate"] >= pass_rate_min)
        for s in stats
    }
    
    # Find base (closest to 0)
    base_alpha = min(alphas, key=abs)
    
    # If base is failed
    if not safe_map[base_alpha]:
        return {"lo": None, "hi": None, "rec": None, "reason_pos": "base_failed", "reason_neg": "base_failed"}
    
    base_idx = alphas.index(base_alpha)
    
    # --- Helper: Identify limiting reason ---
    def find_reason(stat_dict):
        # どの pass_rate が min を下回ったか？
        # 優先順位: sem -> len -> distinct -> punct (あるいは一番低いもの)
        # ここでは「pass_rate_min を下回っているものの中で、最も合格率が低いもの」を主因とする
        candidates = []
        for k in ["sem", "len", "distinct", "punct"]:
            pr_key = f"pass_rate_{k}"
            if stat_dict.get(pr_key, 1.0) < pass_rate_min:
                candidates.append((k, stat_dict.get(pr_key, 1.0)))
        
        if not candidates:
            return "unknown_mixed" # 全体では落ちたが個別に低いのがない？（稀）
        
        # 最も低いものを選ぶ
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0] # "sem" etc.

    # --- Negative Search (Lower Bound) ---
    curr_lo = base_idx
    reason_neg = "limit_reached" # 端まで行った場合
    
    while curr_lo > 0:
        prev_idx = curr_lo - 1
        prev_a = alphas[prev_idx]
        if safe_map[prev_a]:
            curr_lo -= 1
        else:
            # 失敗した地点(prev_a)の理由を特定
            reason_neg = find_reason(stats[prev_idx])
            break
            
    # --- Positive Search (Upper Bound) ---
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
    
    return {
        "lo": range_lo,
        "hi": range_hi,
        "rec": rec,
        "reason_pos": reason_pos,
        "reason_neg": reason_neg
    }

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out_root", required=True)
    
    # Safety Config
    ap.add_argument("--pass_rate_min", type=float, default=0.5)
    ap.add_argument("--sem_min", type=float, default=0.20)
    ap.add_argument("--distinct2_min", type=float, default=0.05)
    ap.add_argument("--len_ratio_min", type=float, default=0.2)
    ap.add_argument("--len_ratio_max", type=float, default=5.0)
    
    # Ignore legacy args
    ap.add_argument("--direction", default="increase")
    ap.add_argument("--prefer_small_abs_alpha", action="store_true")
    ap.add_argument("--punct_ratio_max", type=float, default=0.85)
    ap.add_argument("--max_run_token_max", type=int, default=10)
    ap.add_argument("--max_run_phrase_max", type=int, default=3)
    ap.add_argument("--min_phrase_tokens", type=int, default=3)

    args = ap.parse_args()
    
    if not SKLEARN_AVAILABLE:
        print("[WARN] scikit-learn not found. Using Jaccard instead of TF-IDF.")
    else:
        print("[INFO] Using TF-IDF for semantic check.")
        
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
    
    # Group by Alpha
    by_alpha = defaultdict(list)
    for r in rows:
        a = safe_float(r.get("alpha_total"))
        if a is None: continue
        by_alpha[a].append(r)
        
    # Baseline
    base_alpha = min(by_alpha.keys(), key=abs)
    by_sample_base = {}
    for r in by_alpha[base_alpha]:
        sid = str(r.get("i", r.get("x", "unknown")))
        by_sample_base[sid] = r.get("y", "")

    per_alpha_stats = []
    
    for a, group in sorted(by_alpha.items()):
        n_total = len(group)
        n_pass = 0
        
        # Detail Counters
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
            sid = str(r.get("i", r.get("x", "unknown")))
            y = r.get("y", "")
            y0 = by_sample_base.get(sid, "")
            
            m = compute_metrics(y0, y, use_tfidf=SKLEARN_AVAILABLE)
            is_safe, details = check_safety_details(m, c)
            
            if is_safe: n_pass += 1
            
            # Count detailed pass
            if details["sem"]: n_pass_sem += 1
            if details["len"]: n_pass_len += 1
            if details["distinct"]: n_pass_distinct += 1
            if details["punct"]: n_pass_punct += 1
            if details["max_run_token"]: n_pass_mrt += 1
            if details["max_run_phrase"]: n_pass_mrp += 1
            
            # Count fails
            if not is_safe:
                # Any count
                failed_keys = [k for k, v in details.items() if not v]
                for k in failed_keys: fail_any[k] += 1
                
                # Primary count (Priority: Sem > Len > Distinct > Punct)
                if not details["max_run_phrase"]:
                    fail_primary["max_run_phrase"] += 1
                elif not details["max_run_token"]:
                    fail_primary["max_run_token"] += 1
                elif not details["distinct"]:
                    fail_primary["distinct"] += 1
                elif not details["punct"]:
                    fail_primary["punct"] += 1
                elif not details["len"]:
                    fail_primary["len"] += 1                
                elif not details["sem"]:
                    fail_primary["sem"] += 1

            ds = safe_float(r.get("ds_avg"))
            if ds is not None: ds_values.append(ds)
            
        # Stats Aggregation
        stat_dict = {
            "alpha_total": a,
            "n_total": n_total,
            "pass_rate": n_pass / n_total if n_total > 0 else 0.0,
            
            # New Metrics
            "pass_rate_sem": n_pass_sem / n_total if n_total > 0 else 0.0,
            "pass_rate_len": n_pass_len / n_total if n_total > 0 else 0.0,
            "pass_rate_distinct": n_pass_distinct / n_total if n_total > 0 else 0.0,
            "pass_rate_punct": n_pass_punct / n_total if n_total > 0 else 0.0,
            "pass_rate_mrt": n_pass_mrt / n_total if n_total > 0 else 0.0,
            "pass_rate_mrp": n_pass_mrp / n_total if n_total > 0 else 0.0,
            
            "fail_primary": dict(fail_primary),
            "fail_any": dict(fail_any),
            
            # Legacy fields
            "mean": mean(ds_values) if ds_values else 0.0,
            "median": median(ds_values) if ds_values else 0.0,
            "p_pos": sum(1 for v in ds_values if v > 0)/len(ds_values) if ds_values else 0.0
        }
        per_alpha_stats.append(stat_dict)
        
    # Analyze Range with Reasons
    res = analyze_safety_range_v13(per_alpha_stats, args.pass_rate_min)
    
    # Output
    out_dir = Path(args.out_root) / "range"
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
        
        "per_alpha_stats": per_alpha_stats
    }
    
    write_jsonl(str(out_path), [out_obj])
    print(f"[OK] Safety Analysis v13: {out_path}")
    if res["lo"] is not None:
        print(f"     Range: {res['lo']:.1f} ~ {res['hi']:.1f}")
        print(f"     Reason Pos: {res['reason_pos']}")
        print(f"     Reason Neg: {res['reason_neg']}")
    else:
        print("     [WARN] No safe range found.")

if __name__ == "__main__":
    main()