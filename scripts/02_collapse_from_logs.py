#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_analyze_collapse_from_logs.py

目的:
- 01_run_probe.py などが出した JSONL (probe_base_alltraits.jsonl,
  probe_instruct_alltraits.jsonl) から、
  αごと × trait ごと × model ごとの以下の指標を集計して CSV に出す。

  ・s_avg の平均 (score_mean)
  ・token 長の平均 (len_tokens_mean)
  ・語彙の重複率 (repetition_ratio_mean)
      repetition_ratio = 1 - (unique_tokens / total_tokens)
  ・簡易 toxic 語彙の割合 (toxic_ratio_mean)

使い方例:
  python scripts/02_analyze_collapse_from_logs.py \
    --base_json  exp/mistral_7b/results_asst_cluster/probe_base_alltraits.jsonl \
    --instr_json exp/mistral_7b/results_asst_cluster/probe_instruct_alltraits.jsonl \
    --out_csv    exp/mistral_7b/results_asst_cluster/collapse_asst_cluster.csv \
    --pooling asst --axis_mode cluster
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# 単純な toxic 語彙リスト（必要なら増やしてOK）
TOXIC_WORDS = {
    "fuck", "fucking", "shit", "damn", "bitch", "bastard",
    "idiot", "stupid", "moron", "dumb", "retard",
    "asshole", "loser", "suck", "piss", "crap",
    "garbage", "trash", "hate", "kill", "die",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")  # 英単語ベースでざっくり


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


def calc_collapse_metrics(text: str):
    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        return {
            "len_tokens": 0,
            "repetition_ratio": 0.0,
            "toxic_ratio": 0.0,
        }

    unique = set(tokens)
    repetition_ratio = 1.0 - len(unique) / float(n)

    toxic_cnt = sum(1 for t in tokens if t in TOXIC_WORDS)
    toxic_ratio = toxic_cnt / float(n)

    return {
        "len_tokens": n,
        "repetition_ratio": repetition_ratio,
        "toxic_ratio": toxic_ratio,
    }


def load_jsonl_with_model(path: Path, model_label: str):
    """
    JSONL を読み込んで、(model, trait, alpha_total) ごとに
    s_avg と崩壊度指標を集計できるような dict を返す。
    """
    groups = defaultdict(lambda: {
        "s_avgs": [],
        "len_tokens": [],
        "repetition_ratio": [],
        "toxic_ratio": [],
    })

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            trait = rec.get("trait")
            alpha = rec.get("alpha_total")
            text = rec.get("y", "")

            if trait is None or alpha is None:
                continue

            key = (model_label, trait, float(alpha))

            s_avg = rec.get("s_avg")
            if s_avg is not None:
                groups[key]["s_avgs"].append(float(s_avg))

            m = calc_collapse_metrics(text)
            groups[key]["len_tokens"].append(m["len_tokens"])
            groups[key]["repetition_ratio"].append(m["repetition_ratio"])
            groups[key]["toxic_ratio"].append(m["toxic_ratio"])

    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", required=True)
    ap.add_argument("--instr_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--pooling", default="asst")
    ap.add_argument("--axis_mode", default="cluster")
    args = ap.parse_args()

    base_path = Path(args.base_json)
    instr_path = Path(args.instr_json)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Base / Instruct それぞれ読み込み
    base_groups = load_jsonl_with_model(base_path, model_label="base")
    instr_groups = load_jsonl_with_model(instr_path, model_label="instruct")

    # 結合
    all_groups = {}
    all_groups.update(base_groups)
    all_groups.update(instr_groups)

    rows = []
    for (model, trait, alpha), stats in all_groups.items():
        s_arr = np.array(stats["s_avgs"], dtype=float) if stats["s_avgs"] else np.array([np.nan])
        len_arr = np.array(stats["len_tokens"], dtype=float)
        rep_arr = np.array(stats["repetition_ratio"], dtype=float)
        tox_arr = np.array(stats["toxic_ratio"], dtype=float)

        rows.append({
            "model": model,
            "trait": trait,
            "alpha_total": alpha,
            "score_mean": float(np.nanmean(s_arr)),
            "len_tokens_mean": float(np.mean(len_arr)),
            "repetition_ratio_mean": float(np.mean(rep_arr)),
            "toxic_ratio_mean": float(np.mean(tox_arr)),
            "n_samples": int(len_arr.size),
            "pooling": args.pooling,
            "axis_mode": args.axis_mode,
        })

    df = pd.DataFrame(rows)
    df.sort_values(["trait", "model", "alpha_total"], inplace=True)
    df.to_csv(out_path, index=False)
    print(f"[Done] Saved collapse metrics CSV to: {out_path}")


if __name__ == "__main__":
    main()
