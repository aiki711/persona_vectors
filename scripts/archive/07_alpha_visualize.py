#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_alpha_visualize_per_prompt.py

- *_per_prompt.jsonl だけを対象に、prompt(sid)ごとの alpha range を集計＆可視化
- seaborn無し / matplotlibのみ
- base/instruct で色を固定
- recommended は描かない
- n_prompts=0 問題を sid 対応で解消
"""

from __future__ import annotations
import argparse, glob, json, os, re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIT_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
COLORS = {"base": "#1f77b4", "instruct": "#ff7f0e"}

# per_prompt に限定して拾う
FILENAME_RE = re.compile(r"alpha_range_(base|instruct)_(\w+)_per_prompt\.jsonl$")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except:
        return None

def parse_filename_meta(path: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(path)
    m = FILENAME_RE.match(base)
    if not m:
        return None
    split = m.group(1).lower()
    trait = m.group(2).lower()
    return split, trait

def infer_tag_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if "exp" in parts:
        i = parts.index("exp")
        if i + 1 < len(parts):
            return parts[i + 1]
    # fallback
    return "unknown"

def get_prompt_uid(r: dict) -> str | None:
    for k in ["sid", "prompt_id", "prompt_idx", "sample_id", "i", "x"]:
        v = r.get(k, None)
        if v is not None and str(v).strip() != "":
            return str(v)
    if r.get("prompt"):
        return str(r["prompt"])
    return None

def has_range(r: dict) -> bool:
    lo = r.get("alpha_lo", None)
    hi = r.get("alpha_hi", None)
    return (lo is not None) and (hi is not None)

def summarize_per_prompt_range(tag: str, split: str, trait: str, fpath: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # split 正規化
    split = (split or "").lower()

    # prompt数（dropna前に数える）
    uids = [get_prompt_uid(r) for r in rows]
    uids = [u for u in uids if u is not None]
    n_prompts_total = len(set(uids))

    uids_ok = [get_prompt_uid(r) for r in rows if has_range(r)]
    uids_ok = [u for u in uids_ok if u is not None]
    n_prompts_with_range = len(set(uids_ok))

    # range統計（rangeがあるものだけで）
    lo_list = [to_float(r.get("alpha_lo")) for r in rows]
    hi_list = [to_float(r.get("alpha_hi")) for r in rows]
    lo_list = [v for v in lo_list if v is not None]
    hi_list = [v for v in hi_list if v is not None]

    return {
        "tag": tag,
        "kind": "range_per_prompt",
        "split": split,
        "trait": trait,
        "n_rows": len(rows),
        "n_prompts_total": n_prompts_total,
        "n_prompts_with_range": n_prompts_with_range,
        "alpha_lo_median": float(pd.Series(lo_list).median()) if lo_list else None,
        "alpha_hi_median": float(pd.Series(hi_list).median()) if hi_list else None,
        "alpha_lo_mean": float(pd.Series(lo_list).mean()) if lo_list else None,
        "alpha_hi_mean": float(pd.Series(hi_list).mean()) if hi_list else None,
    }

def plot_range_per_prompt(df: pd.DataFrame, tag: str, out_path: str):
    """
    traitごとに、base/instruct の (lo~hi) の代表値(ここではmedian) を線で描く
    recommended点は描かない
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    y_positions = range(len(TRAIT_ORDER))
    offsets = {"base": -0.15, "instruct": 0.15}

    has_any = False
    for i, trait in enumerate(TRAIT_ORDER):
        for split in ["base", "instruct"]:
            row = df[(df["trait"] == trait) & (df["split"] == split)]
            if row.empty:
                continue
            lo = row.iloc[0]["alpha_lo_median"]
            hi = row.iloc[0]["alpha_hi_median"]
            if pd.isna(lo) or pd.isna(hi):
                continue

            has_any = True
            y = i + offsets[split]
            color = COLORS[split]

            ax.hlines(y, lo, hi, colors=color, linewidth=3, alpha=0.75)
            ax.vlines(lo, y - 0.1, y + 0.1, colors=color, linewidth=2)
            ax.vlines(hi, y - 0.1, y + 0.1, colors=color, linewidth=2)

    if not has_any:
        plt.close(fig)
        return

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([t.capitalize() for t in TRAIT_ORDER], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Alpha (median lo~hi across prompts)", fontsize=12)
    ax.set_title(f"Per-prompt Alpha Range (median)\nModel: {tag}", fontsize=14)

    # 凡例（線の色）
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLORS["base"], lw=3, label="base"),
        Line2D([0], [0], color=COLORS["instruct"], lw=3, label="instruct"),
    ]
    ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--globs", nargs="+", required=True, help="e.g. exp/**/alpha_range_*_per_prompt.jsonl")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    files = []
    for g in args.globs:
        files.extend(glob.glob(g, recursive=True))
    files = sorted(set(files))
    if not files:
        raise SystemExit("[ERROR] No files matched.")

    rows_out = []
    for fpath in files:
        meta = parse_filename_meta(fpath)
        if meta is None:
            continue
        split, trait = meta
        if trait not in TRAIT_ORDER:
            continue

        rows = read_jsonl(fpath)
        if not rows:
            continue

        tag = rows[0].get("tag") or infer_tag_from_path(fpath)
        rows_out.append(summarize_per_prompt_range(tag, split, trait, fpath, rows))

    df = pd.DataFrame(rows_out)
    if df.empty:
        raise SystemExit("[ERROR] No valid data extracted.")

    df["split"] = df["split"].astype(str).str.lower()
    df["trait"] = pd.Categorical(df["trait"], categories=TRAIT_ORDER, ordered=True)
    df = df.sort_values(["tag", "split", "trait"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] saved csv: {args.out_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    for tag in sorted(df["tag"].unique()):
        df_tag = df[df["tag"] == tag]
        out_path = os.path.join(args.out_dir, f"per_prompt_range_{tag}.png")
        plot_range_per_prompt(df_tag, tag, out_path)

    print("[OK] done.")

if __name__ == "__main__":
    main()
