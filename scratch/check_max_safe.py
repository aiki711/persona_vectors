import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
METHODS = [
    ("logit_diff", "DLS Logit-Diff"),
    ("cos_only", "DLS Cos-Only"),
    ("rank_only", "DLS Rank-Only"),
    ("proj_cos_only", "DLS Proj Cos-Only"),
    ("proj_rank_only", "DLS Proj Rank-Only"),
    ("masked_cos_only", "PDF Cos-Only"),
    ("masked_rank_only", "PDF Rank-Only"),
    ("masked_proj_cos_only", "PDF Proj Cos-Only"),
    ("masked_proj_rank_only", "PDF Proj Rank-Only"),
]

def calculate_repetition_rate(text: str, n: int) -> float:
    if not isinstance(text, str):
        return 0.0
    words = [w.strip(".,!?:;()\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    unique_ngrams = set(ngrams)
    return (len(ngrams) - len(unique_ngrams)) / len(ngrams)

def get_scores(results_dir: Path, trait: str, method: str, criteria: str):
    best_score = 0.0
    best_alpha = np.nan
    best_ppl = np.nan
    best_coherence = np.nan
    
    trait_dir = results_dir / trait
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        jsonl_path = trait_dir / f"{method}_Val{float(val)}.jsonl"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
        if not jsonl_path.exists():
            jsonl_path = trait_dir / f"{method}_Val{val}.jsonl"
            
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "dyn_score" in df.columns:
                    df["dyn_score"] = df["dyn_score"].replace(0, np.nan)
                mean_score = df["dyn_score"].mean()
                mean_ppl = df["dyn_ppl"].mean()
                max_ppl = df["dyn_ppl"].max() if "dyn_ppl" in df.columns else np.nan
                
                if "dyn_reason" in df.columns:
                    coherence_rate = df["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean()
                else:
                    coherence_rate = 1.0
                
                safe_ppl_rate = (df["dyn_ppl"] <= 25.0).mean() if "dyn_ppl" in df.columns else 1.0
                
                if criteria == "strict":
                    is_safe = (mean_ppl <= 25.0 and coherence_rate >= 0.8 and max_ppl <= 25.0)
                else: # practical
                    is_safe = (mean_ppl <= 20.0 and coherence_rate >= 0.8 and safe_ppl_rate >= 0.9)
                
                if is_safe:
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = val
                        best_ppl = mean_ppl
                        best_coherence = coherence_rate
            except Exception as e:
                pass
    return best_score, best_alpha

def main():
    results_dir = Path("exp_steering_dyn_layer_raw/results")
    
    print(f"{'Trait':<20} | {'Method':<22} | {'Strict Max Score (Alpha)':<25} | {'Practical Max Score (Alpha)':<25}")
    print("-" * 100)
    
    changes = 0
    total = 0
    for trait in TRAITS:
        for method_key, method_name in METHODS:
            s_score, s_alpha = get_scores(results_dir, trait, method_key, "strict")
            p_score, p_alpha = get_scores(results_dir, trait, method_key, "practical")
            
            strict_str = f"{s_score:.3f} (a={s_alpha})" if s_score > 0 else "N/A"
            practical_str = f"{p_score:.3f} (a={p_alpha})" if p_score > 0 else "N/A"
            
            if s_score != p_score or s_alpha != p_alpha:
                changes += 1
                flag = "*"
            else:
                flag = " "
                
            total += 1
            print(f"{trait:<20} | {method_name:<22} | {strict_str:<25} | {practical_str:<25} {flag}")
            
    print("-" * 100)
    print(f"Total combinations: {total}, Changed combinations: {changes}")

if __name__ == "__main__":
    main()
