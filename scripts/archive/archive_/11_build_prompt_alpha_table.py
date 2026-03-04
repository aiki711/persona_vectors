#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a per-(prompt, alpha) table from *_with_rms.jsonl files.

Expected json line fields (at least):
  - i (prompt index)
  - trait
  - alpha_total
  - alpha_per_layer (optional)
  - rms_by_layer: {layer: {rms_before:{mean,var,n}, rms_after:{mean,var,n}, alpha_over_rms_before_mean}}
  - s_avg, s0_avg, ds_avg (optional)

We aggregate over layers (mean) to produce scalar predictors.

Output columns (main):
  tag, split, trait, prompt_id, alpha_total,
  rms_before_mean_over_layers, rms_after_mean_over_layers,
  rms_before_var_over_layers,  rms_after_var_over_layers,
  alpha_over_rms_before_mean_over_layers,
  s_avg, s0_avg, ds_avg
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# -----------------------
# helpers
# -----------------------

def safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return np.nan
    except Exception:
        return np.nan

def parse_path_for_meta(path: str) -> Dict[str, str]:
    """
    Try to parse tag/split/trait from path like:
      exp/mistral_7b/asst_pairwise_results/mistral_7b_base_agreeableness_with_rms.jsonl

    Returns dict with keys: tag, split, trait (may be '').
    """
    base = os.path.basename(path)
    # tag_split_trait_with_rms.jsonl
    m = re.match(r"^(?P<tag>.+?)_(?P<split>base|instruct)_(?P<trait>openness|conscientiousness|extraversion|agreeableness|neuroticism).*?_with_rms\.jsonl$", base)
    if m:
        return {k: m.group(k) for k in ["tag", "split", "trait"]}

    # fallback: get tag from parent dir exp/{tag}/...
    tag = ""
    split = ""
    trait = ""
    parts = path.replace("\\", "/").split("/")
    # find exp/{tag}/...
    for i in range(len(parts) - 1):
        if parts[i] == "exp" and i + 1 < len(parts):
            tag = parts[i + 1]
            break

    return {"tag": tag, "split": split, "trait": trait}

def layer_dict_items(d: Any):
    if isinstance(d, dict):
        return d.items()
    return []

def agg_layer_stats(rms_by_layer: Dict[str, Any]) -> Dict[str, float]:
    """
    Aggregate per-layer rms stats by simple mean over layers (ignoring NaNs).
    Returns scalars:
      rms_before_mean_over_layers, rms_after_mean_over_layers,
      rms_before_var_over_layers,  rms_after_var_over_layers,
      alpha_over_rms_before_mean_over_layers
    """
    before_means: List[float] = []
    after_means: List[float] = []
    before_vars: List[float] = []
    after_vars: List[float] = []
    a_over: List[float] = []

    for _, v in layer_dict_items(rms_by_layer):
        # v is like {"rms_before": {...}, "rms_after": {...}, "alpha_over_rms_before_mean": ...}
        if not isinstance(v, dict):
            continue

        rb = v.get("rms_before", {})
        ra = v.get("rms_after", {})

        before_means.append(safe_float(rb.get("mean", np.nan)))
        after_means.append(safe_float(ra.get("mean", np.nan)))
        before_vars.append(safe_float(rb.get("var", np.nan)))
        after_vars.append(safe_float(ra.get("var", np.nan)))
        a_over.append(safe_float(v.get("alpha_over_rms_before_mean", np.nan)))

    def nanmean(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else np.nan

    return {
        "rms_before_mean_over_layers": nanmean(before_means),
        "rms_after_mean_over_layers": nanmean(after_means),
        "rms_before_var_over_layers": nanmean(before_vars),
        "rms_after_var_over_layers": nanmean(after_vars),
        "alpha_over_rms_before_mean_over_layers": nanmean(a_over),
        "n_layers_used": int(np.isfinite(np.asarray(before_means, dtype=float)).sum()),
    }


# -----------------------
# main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_glob", required=True, help="glob for *_with_rms.jsonl, e.g. 'exp/*/asst_pairwise_results/*_with_rms.jsonl'")
    ap.add_argument("--out_csv", required=True, help="output csv path")
    ap.add_argument("--keep_text", action="store_true", help="also store prompt text x and generated y (can be huge)")
    ap.add_argument("--max_lines_per_file", type=int, default=0, help="if >0, read at most this many lines per file (debug)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.jsonl_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.jsonl_glob}")

    rows: List[Dict[str, Any]] = []
    for path in paths:
        meta = parse_path_for_meta(path)
        tag0 = meta.get("tag", "")
        split0 = meta.get("split", "")
        trait0 = meta.get("trait", "")

        with open(path, "r", encoding="utf-8") as f:
            for li, line in enumerate(f):
                if args.max_lines_per_file and li >= args.max_lines_per_file:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    # skip malformed line
                    continue

                prompt_id = obj.get("i", obj.get("prompt_id", None))
                trait = obj.get("trait", trait0)
                alpha_total = obj.get("alpha_total", obj.get("alpha", None))
                alpha_per_layer = obj.get("alpha_per_layer", np.nan)

                # rms_by_layer can be stored under "rms_by_layer" or nested
                rms_by_layer = obj.get("rms_by_layer", None)
                if rms_by_layer is None:
                    # try nested under something else if needed
                    rms_by_layer = {}

                agg = agg_layer_stats(rms_by_layer if isinstance(rms_by_layer, dict) else {})

                row = {
                    "source_path": path,
                    "tag": str(tag0),
                    "split": str(split0),
                    "trait": str(trait),
                    "prompt_id": int(prompt_id) if prompt_id is not None and str(prompt_id).isdigit() else prompt_id,
                    "alpha_total": safe_float(alpha_total),
                    "alpha_per_layer": safe_float(alpha_per_layer),
                    # optional steering signals
                    "s_avg": safe_float(obj.get("s_avg", np.nan)),
                    "s0_avg": safe_float(obj.get("s0_avg", np.nan)),
                    "ds_avg": safe_float(obj.get("ds_avg", np.nan)),
                    # aggregated rms stats
                    **agg,
                }

                if args.keep_text:
                    row["x_prompt"] = obj.get("x", "")
                    row["y_text"] = obj.get("y", "")

                rows.append(row)

    df = pd.DataFrame(rows)

    # Light sanity: attempt to fill missing split/trait if path parse failed but json has them
    # (tag is usually from path)
    if "split" in df.columns:
        df["split"] = df["split"].replace("", np.nan)
    if "trait" in df.columns:
        df["trait"] = df["trait"].replace("", np.nan)

    # Save
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Quick coverage report (printed)
    print("saved:", args.out_csv)
    print("rows:", len(df))
    if len(df):
        print("coverage (rows per tag/split/trait):")
        cov = df.groupby(["tag", "split", "trait"]).size().sort_values(ascending=False)
        print(cov.head(50).to_string())
        print("unique prompts per group (approx):")
        up = df.groupby(["tag", "split", "trait"])["prompt_id"].nunique().sort_values(ascending=False)
        print(up.head(50).to_string())

if __name__ == "__main__":
    main()
