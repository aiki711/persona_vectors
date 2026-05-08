import pandas as pd
import numpy as np

def analyze(path, label):
    try:
        df = pd.read_csv(path)
        print(f"\n=== {label} ===")
        # Check available columns
        cols = ["alpha_total", "raw_score_extraversion"]
        if "ppl" in df.columns:
            cols.append("ppl")
        
        # Note: raw_score_extraversion might be different if we steer other traits,
        # but for comparative study on Extraversion, it's the main metric.
        summary = df.groupby("alpha_total")[cols[1:]].agg(["mean", "std"]).round(2)
        print(summary)
    except FileNotFoundError:
        print(f"File not found: {path}")

# Pilot results
# analyze("exp/adv_extra_golden_scaled_scores.csv", "Golden-Scaled (L1)")

# New Big5 experiment results (will be populated soon)
analyze("exp/big5_extra_golden_scores.csv", "Big5: Golden Layer (Baseline)")
analyze("exp/big5_extra_top5_scores.csv", "Big5: Top-5 Layers (Distributed)")
