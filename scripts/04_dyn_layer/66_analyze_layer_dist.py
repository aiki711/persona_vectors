#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter
import pandas as pd

def analyze_dir(results_dir: Path):
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist.")
        return {}
    
    layer_counts = Counter()
    total = 0
    
    # Walk through all jsonl files
    for jsonl_path in results_dir.glob("**/*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "dyn_layer" in data:
                        layer_counts[data["dyn_layer"]] += 1
                        total += 1
                except Exception as e:
                    pass
                    
    if total == 0:
        return {}
        
    # Convert to percentages
    dist = {layer: (count / total) * 100 for layer, count in layer_counts.items()}
    return dict(sorted(dist.items()))

def main():
    dirs = {
        "Unconstrained": Path("exp_steering_dyn_layer/results"),
        "Constrained (12-24)": Path("exp_steering_dyn_layer_constrained/results"),
        "Z-score Normalized": Path("exp_steering_dyn_layer_zscore/results"),
        "Constrained Z-score (9-30)": Path("exp_steering_dyn_layer_CnsZsc/results")
    }
    
    all_dists = {}
    for name, path in dirs.items():
        print(f"Analyzing {name} ({path})...")
        all_dists[name] = analyze_dir(path)
        
    # Print summary table
    all_layers = sorted(list(set(
        L for dist in all_dists.values() for L in dist.keys()
    )))
    
    rows = []
    for L in all_layers:
        row = {"Layer": L}
        for name in dirs.keys():
            row[name] = f"{all_dists[name].get(L, 0.0):.1f}%"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print("\n=== Layer Selection Distribution Comparison ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
