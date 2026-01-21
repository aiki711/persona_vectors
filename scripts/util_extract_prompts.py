#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import json
import os
from collections import defaultdict
from datasets import load_dataset
from persona_vectors.live_axes import AXES

def extract_unseen_prompts(trait: str, count: int, vector_seed: int = 2025, vector_count: int = 1000):
    print(f"Loading big5-chat dataset to exclude first {vector_count} pairs (seed={vector_seed})...")
    
    # 1. Dataset loading
    ds_all = load_dataset("wenkai-li/big5_chat")
    if isinstance(ds_all, dict):
        split_name = next(iter(ds_all.keys()))
        ds = ds_all[split_name]
    else:
        ds = ds_all

    # 2. Logic to identify used indices (same as 00_prepare_vectors.py)
    buckets = defaultdict(lambda: {"high": [], "low": [], "input": []})
    
    for ex in ds:
        tr_raw = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr_raw not in AXES: 
            continue
            
        orig_idx = ex.get("original_index")
        if orig_idx is None:
            continue
            
        to = (ex.get("train_output") or "").strip()
        ti = (ex.get("train_input") or "").strip()
        
        if not to: continue
        
        if lv in ["high", "low"]:
            buckets[(tr_raw, orig_idx)][lv].append(to)
            buckets[(tr_raw, orig_idx)]["input"].append(ti)

    temp_pairs = {ax: [] for ax in AXES}
    
    for (tr, orig_idx), d in buckets.items():
        highs = d["high"]
        lows = d["low"]
        if not highs or not lows:
            continue
        
        inp_text = d["input"][0] if d["input"] else ""
        
        temp_pairs[tr].append({
            "orig_idx": orig_idx,
            "input": inp_text
        })
        
    # 3. Reproduce shuffle
    rng_state_backup = random.getstate()
    candidates = []
    
    try:
        random.seed(vector_seed)
        
        target_traits = AXES if trait == 'all' else [trait]
            
        # We must align random state by processing ALL axes in fixed order as done in 00_prepare_vectors
        # 00_prepare_vectors.py iterates over AXES_CANON (which is AXES)
        
        for ax in AXES:
            lst = temp_pairs[ax]
            random.shuffle(lst)
            
            # If this axis is one of our targets, extract unused
            if ax in target_traits:
                # Skip the used ones
                remaining = lst[vector_count:]
                print(f"  [trait={ax}] Total: {len(lst)}, Used: {vector_count}, Remaining: {len(remaining)}")
                
                for item in remaining:
                     if item["input"] and item["input"].strip():
                         candidates.append(item["input"])

    finally:
        random.setstate(rng_state_backup)

    # Now we are back to the caller's random state.
    # We can shuffle candidates here.
    random.shuffle(candidates)
    
    selected = candidates[:count]
    print(f"Collected total {len(candidates)} candidates from {len(target_traits)} traits.")
    print(f"Selected {len(selected)} prompts.")
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=str, required=True, help="Target trait or 'all'")
    parser.add_argument("--count", type=int, default=100, help="Number of prompts to extract")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--vector_seed", type=int, default=2025, help="Seed used in 00_prepare_vectors")
    parser.add_argument("--vector_count", type=int, default=1000, help="Number of vectors per axis used in 00_prepare_vectors")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling")
    
    args = parser.parse_args()
    
    if args.trait != 'all' and args.trait not in AXES:
        parser.error(f"--trait must be one of {AXES} or 'all'")
    
    random.seed(args.seed)
    
    prompts = extract_unseen_prompts(
        trait=args.trait,
        count=args.count,
        vector_seed=args.vector_seed,
        vector_count=args.vector_count
    )
    
    # Save as JSON list
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
        
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
