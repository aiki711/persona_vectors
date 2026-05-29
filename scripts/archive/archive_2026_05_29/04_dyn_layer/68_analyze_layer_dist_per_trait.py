#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 68_analyze_layer_dist_per_trait.py
#
# 集計結果のフォルダから性格特性（Trait）ごと、および選択手法（Method）ごとに、
# 動的に選択されたレイヤー（dyn_layer）の分布を集計して出力するスクリプト。
#

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

def analyze_per_trait(results_dir: Path):
    if not results_dir.exists():
        print(f"[ERROR] Directory {results_dir} does not exist.")
        return None
    
    # 構造: trait -> method -> layer -> count
    data_counts = defaultdict(lambda: defaultdict(Counter))
    data_totals = defaultdict(lambda: defaultdict(int))
    
    # 全ての jsonl ファイルを探索
    for jsonl_path in results_dir.glob("**/*.jsonl"):
        # ディレクトリ構造から性格特性名を取得
        # 例: exp_steering_dyn_layer_CnsZsc/results/neuroticism/logit_diff_Val10.0.jsonl
        # -> trait = "neuroticism"
        trait = jsonl_path.parent.name
        if trait not in TRAITS:
            continue
            
        filename = jsonl_path.name
        if "logit_diff" in filename:
            method = "logit_diff"
        elif "anti_alignment" in filename:
            method = "anti_alignment"
        elif filename.startswith("dyn_Val"):
            # 59_run_dynamic_layer_steering.py由来のデータ（Bhandari et al. の logit_diff 相当）
            method = "logit_diff"
        else:
            method = "unknown"
            
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "dyn_layer" in data:
                        L = data["dyn_layer"]
                        data_counts[trait][method][L] += 1
                        data_totals[trait][method] += 1
                except:
                    pass

    # 結果をパーセンテージに変換
    dists = {}
    for trait in TRAITS:
        dists[trait] = {}
        for method in ["logit_diff", "anti_alignment"]:
            counts = data_counts[trait][method]
            total = data_totals[trait][method]
            if total == 0:
                continue
            dist = {L: (count / total) * 100 for L, count in counts.items()}
            dists[trait][method] = dict(sorted(dist.items()))
            
    return dists

def print_table(dists, method):
    # すべての出現レイヤーを取得
    all_layers = set()
    for trait in TRAITS:
        if trait in dists and method in dists[trait]:
            all_layers.update(dists[trait][method].keys())
    
    if not all_layers:
        print(f"\nNo data found for method: {method}")
        return

    sorted_layers = sorted(list(all_layers))
    
    rows = []
    for L in sorted_layers:
        row = {"Layer": L}
        for trait in TRAITS:
            val = 0.0
            if trait in dists and method in dists[trait]:
                val = dists[trait][method].get(L, 0.0)
            row[trait.capitalize()] = f"{val:.1f}%"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print(f"\n=== Layer Selection Distribution per Trait (Method: {method}) ===")
    print(df.to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", "-d", default="exp_steering_dyn_layer_CnsZsc/results",
                    help="実験結果の保存先ディレクトリ")
    args = ap.parse_args()
    
    results_dir = Path(args.results_dir)
    print(f"Analyzing layer selection distribution per trait in: {results_dir}")
    
    dists = analyze_per_trait(results_dir)
    if not dists:
        return
        
    for method in ["logit_diff", "anti_alignment"]:
        print_table(dists, method)

if __name__ == "__main__":
    main()
