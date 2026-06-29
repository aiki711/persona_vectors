import pandas as pd
from pathlib import Path
import numpy as np

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
METHODS = [
    ("DLS Logit-Diff", "logit_diff"),
    ("DLS Cos-Only", "cos_only"),
    ("DLS Rank-Only", "rank_only"),
    ("DLS Proj Cos-Only", "proj_cos_only"),
    ("DLS Proj Rank-Only", "proj_rank_only"),
    ("PDF Cos-Only", "masked_cos_only"),
    ("PDF Rank-Only", "masked_rank_only"),
    ("PDF Proj Cos-Only", "masked_proj_cos_only"),
    ("PDF Proj Rank-Only", "masked_proj_rank_only"),
]

print("--- PROMPT LEVEL DLS (ALPHA = 5.0) ---")
prompt_results = "exp_steering_dyn_layer_raw/results"
for name, key in METHODS:
    scores = []
    for trait in TRAITS:
        csv_path = Path(prompt_results) / trait / f"scores_{key}_Val5.0.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = "dyn_score" if "dyn_score" in df.columns else df.columns[2]
            scores.append(df[col].mean())
    if scores:
        print(f"{name:25s}: Mean={np.mean(scores):.3f} (Individual: {', '.join(f'{s:.2f}' for s in scores)})")

print("\n--- GEN TIME DLS (ALPHA = 5.0) ---")
gen_results = "exp_steering_dyn_gen_time_raw/results"
for name, key in METHODS[1:]:  # skip logit-diff
    scores = []
    for trait in TRAITS:
        csv_path = Path(gen_results) / trait / f"scores_{key}_Val5.0.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = "dyn_score" if "dyn_score" in df.columns else df.columns[2]
            scores.append(df[col].mean())
    if scores:
        print(f"{name:25s}: Mean={np.mean(scores):.3f} (Individual: {', '.join(f'{s:.2f}' for s in scores)})")
