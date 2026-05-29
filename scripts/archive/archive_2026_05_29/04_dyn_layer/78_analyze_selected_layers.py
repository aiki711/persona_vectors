#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 78_analyze_selected_layers.py
#

import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
METHODS = {
    "logit_diff": "DLS_logit_diff",
    "anti_alignment": "DLS_anti_align",
    "relative_anti_alignment": "DLS_relative"
}

def main():
    base_dir = Path("exp_steering_dyn_layer_all_layers/results")
    if not base_dir.exists():
        print(f"[ERROR] Directory {base_dir} does not exist.")
        return

    # Structure: trait -> method -> list of layers
    trait_method_layers = defaultdict(lambda: defaultdict(list))

    for trait in TRAITS:
        trait_dir = base_dir / trait
        if not trait_dir.exists():
            continue
        for jsonl_path in trait_dir.glob("*.jsonl"):
            filename = jsonl_path.name
            
            # Determine method
            matched_method = None
            for key, name in METHODS.items():
                if filename.startswith(f"{key}_Val"):
                    matched_method = name
                    break
            
            if not matched_method:
                continue

            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "dyn_layer" in data:
                            trait_method_layers[trait][matched_method].append(data["dyn_layer"])
                    except Exception:
                        pass

    # Print summary per trait
    print("# Layer Selection Analysis per Trait and Method\n")
    for trait in TRAITS:
        print(f"## {trait.capitalize()}")
        
        # Prepare table data
        methods_list = [METHODS[k] for k in ["logit_diff", "anti_alignment", "relative_anti_alignment"]]
        
        # We want to show top 3 selected layers with their percentage
        row_data = []
        for m in methods_list:
            layers = trait_method_layers[trait][m]
            total = len(layers)
            if total == 0:
                row_data.append({
                    "Method": m,
                    "Total Samples": 0,
                    "Distribution (Layer: %)": "No Data"
                })
                continue
            
            counts = Counter(layers)
            # sort by percentage descending
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            dist_str = ", ".join([f"L{L}: {c/total*100:.1f}%" for L, c in sorted_counts[:4]])
            row_data.append({
                "Method": m,
                "Total Samples": total,
                "Distribution (Layer: %; top 4)": dist_str
            })
            
        df = pd.DataFrame(row_data)
        print(df.to_markdown(index=False))
        print()

if __name__ == "__main__":
    main()
