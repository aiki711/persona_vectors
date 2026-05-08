#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# util_summarize_vocab_evolution.py
#
# Take the output of 36_adaptive_vocabulary_scan.csv and create a human-readable 
# summary of how personality tokens emerge and shift across layers.

import pandas as pd
import argparse
import os

def summarize_evolution(csv_path, out_txt):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    traits = df['Trait'].unique()
    
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("=== Vocabulary Evolution Analysis across Layers ===\n\n")
        
        for trait in traits:
            f.write(f"--- TRAIT: {trait.upper()} ---\n")
            trait_df = df[df['Trait'] == trait].sort_values('Layer')
            
            for _, row in trait_df.iterrows():
                layer = row['Layer']
                top = row['Top_Tokens (High)']
                bot = row['Bottom_Tokens (Low)']
                
                # Simple heuristic to see if the layer has "meaningful" words (len > 3)
                # This helps identify where the layer transitions from syntax to semantics
                meaningful_count = sum(1 for t in top.split(',') if len(t.strip().replace("'", "")) > 3)
                mark = "[SEMANTIC]" if meaningful_count > 5 else "[SYNTACTIC/NOISY]"
                
                f.write(f"Layer {layer:02d} {mark:18s} | Top: {top[:100]}...\n")
            f.write("\n" + "="*50 + "\n\n")

    print(f"Summary written to {out_txt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="exp_adaptive_steering/results/adaptive_vocab_scan.csv")
    parser.add_argument("--out", type=str, default="exp_adaptive_steering/results/layer_vocab_summary.txt")
    args = parser.parse_args()
    
    summarize_evolution(args.csv, args.out)

if __name__ == "__main__":
    main()
