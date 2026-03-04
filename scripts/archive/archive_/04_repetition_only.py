#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_repetition_only.py

01_run_probe.py が出力した JSONL（base / instruct）から、
- 性格スコアの変化 ds_avg
- 繰り返し度 (distinct1, distinct2, max_token_freq_ratio)
- 有害語カウント (toxic_count, toxic_ratio)

だけを抜き出して CSV にまとめるスクリプト。

使い方:
  python scripts/04_repetition_only.py \
    --base_json  exp/mistral_7b/results_asst_cluster/probe_base_alltraits.jsonl \
    --instr_json exp/mistral_7b/results_asst_cluster/probe_instruct_alltraits.jsonl \
    --out_csv    exp/mistral_7b/results_asst_cluster/repetition_asst_cluster.csv
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm


# ごく簡単な有害語リスト（必要に応じて拡張）
TOXIC_WORDS = {
    "fuck", "fucking", "shit", "damn", "bitch", "idiot",
    "stupid", "dumb", "piss", "asshole", "bastard",
}


def simple_tokenize(text: str):
    """とりあえず空白区切りのシンプルなトークナイザ."""
    return text.strip().split()


def compute_repetition_features(tokens):
    """
    - distinct1_ratio: 異なる単語数 / 総トークン数
    - distinct2_ratio: 異なるビグラム数 / 総ビグラム数
    - max_token_freq_ratio: 最頻出トークン頻度 / 総トークン数
      → これが大きいと「同じトークンの繰り返し」が酷い
    """
    n = len(tokens)
    if n == 0:
        return {
            "distinct1_ratio": 0.0,
            "distinct2_ratio": 0.0,
            "max_token_freq_ratio": 0.0,
        }

    c = Counter(tokens)
    distinct1 = len(c) / n
    max_freq_ratio = max(c.values()) / n

    bigrams = list(zip(tokens, tokens[1:]))
    if bigrams:
        distinct2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct2 = 0.0

    return {
        "distinct1_ratio": float(distinct1),
        "distinct2_ratio": float(distinct2),
        "max_token_freq_ratio": float(max_freq_ratio),
    }


def compute_toxic_features(tokens):
    """簡易な有害語カウント."""
    n = len(tokens)
    if n == 0:
        return {"toxic_count": 0, "toxic_ratio": 0.0}
    cnt = sum(1 for t in tokens if t.lower() in TOXIC_WORDS)
    return {
        "toxic_count": int(cnt),
        "toxic_ratio": float(cnt / n),
    }


def process_jsonl(in_path, model_tag: str):
    """
    probe_*_alltraits.jsonl を読んで、1行を1サンプルとして特徴量を計算。
    model_tag は "base" / "instruct" など。
    """
    rows = []

    # 行数カウント（プログレスバー用）
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = None

    desc = f"{model_tag}: {os.path.basename(in_path)}"

    with open(in_path, "r", encoding="utf-8") as f:
        if total_lines is not None:
            iterator = tqdm(f, total=total_lines, desc=desc)
        else:
            iterator = tqdm(f, desc=desc)

        for line in iterator:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                # 壊れた行は無視
                continue

            text = rec.get("y", "") or ""
            tokens = simple_tokenize(text)

            rep = compute_repetition_features(tokens)
            tox = compute_toxic_features(tokens)

            row = {
                "model": model_tag,
                "trait": rec.get("trait"),
                "alpha_total": rec.get("alpha_total"),
                "i": rec.get("i"),
                "alpha_mode": rec.get("alpha_mode"),
                "alpha_per_layer": rec.get("alpha_per_layer"),
                "s_avg": rec.get("s_avg"),
                "s0_avg": rec.get("s0_avg"),
                "ds_avg": rec.get("ds_avg"),
                "text_len_tokens": len(tokens),
            }
            row.update(rep)
            row.update(tox)

            rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_json", required=True,
        help="probe_base_*.jsonl を cat したファイル"
    )
    ap.add_argument(
        "--instr_json", required=True,
        help="probe_instruct_*.jsonl を cat したファイル"
    )
    ap.add_argument(
        "--out_csv", required=True,
        help="出力 CSV パス"
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading & processing base_json: {args.base_json}")
    base_rows = process_jsonl(args.base_json, "base")

    print(f"[INFO] Loading & processing instr_json: {args.instr_json}")
    instr_rows = process_jsonl(args.instr_json, "instruct")

    all_rows = base_rows + instr_rows
    if not all_rows:
        print("[WARN] No rows collected. Check input files?")
    df = pd.DataFrame(all_rows)

    # ソートしておくと後で見やすい
    sort_cols = [c for c in ["trait", "model", "alpha_total", "i"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(args.out_csv, index=False)
    print(f"[Done] Saved repetition/toxic metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
