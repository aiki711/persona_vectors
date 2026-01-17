#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Join boundaries from merged_metrics.csv onto the per-(prompt,alpha) table,
then add is_safe label:

  safe if  -abs_range_lo <= alpha_total <= range_hi

Inputs:
  - prompt_alpha_table.csv  (from 11_build_prompt_alpha_table.py)
  - merged_metrics.csv      (must include: tag, split, trait, range_hi, abs_range_lo or range_lo)

Output:
  - prompt_alpha_labeled.csv
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="prompt_alpha_table.csv")
    ap.add_argument("--merged", required=True, help="merged_metrics.csv (boundaries)")
    ap.add_argument("--out_csv", required=True, help="output labeled csv")
    args = ap.parse_args()

    tab = pd.read_csv(args.table)
    mer = pd.read_csv(args.merged)

    # Prepare boundaries
    mer = mer.copy()
    if "abs_range_lo" not in mer.columns:
        if "range_lo" in mer.columns:
            mer["abs_range_lo"] = safe_num(mer["range_lo"]).abs()
        else:
            raise ValueError("merged_metrics.csv must have abs_range_lo or range_lo")

    need = ["tag", "split", "trait", "range_hi", "abs_range_lo"]
    for c in need:
        if c not in mer.columns:
            raise ValueError(f"merged_metrics.csv missing: {c}")

    # Keep only boundary columns (deduplicate if multiple rows exist; take first)
    bnd = mer[["tag","split","trait","range_hi","abs_range_lo"]].copy()
    bnd["range_hi"] = safe_num(bnd["range_hi"])
    bnd["abs_range_lo"] = safe_num(bnd["abs_range_lo"])
    bnd = bnd.drop_duplicates(subset=["tag","split","trait"], keep="first")

    # Join
    out = tab.merge(bnd, on=["tag","split","trait"], how="left")

    # Label
    out["alpha_total"] = safe_num(out.get("alpha_total", np.nan))
    out["y_pos"] = safe_num(out.get("range_hi", np.nan))
    out["y_neg"] = safe_num(out.get("abs_range_lo", np.nan))  # positive magnitude
    out["is_safe"] = np.where(
        np.isfinite(out["alpha_total"]) & np.isfinite(out["y_pos"]) & np.isfinite(out["y_neg"]) &
        (out["alpha_total"] >= -out["y_neg"]) & (out["alpha_total"] <= out["y_pos"]),
        1, 0
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)
    print("rows:", len(out))

    # quick sanity
    miss = out["y_pos"].isna().mean()
    print("missing-boundary ratio (y_pos NaN):", float(miss))
    print("is_safe distribution:", out["is_safe"].value_counts(dropna=False).to_dict())

if __name__ == "__main__":
    main()
