import numpy as np
import pandas as pd
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
# Include all 9 methods to compare with the fixed-layer baselines
METHODS = [
    ("DLS Logit-Diff",        "logit_diff"),
    ("DLS Cos-Only",          "cos_only"),
    ("DLS Rank-Only",         "rank_only"),
    ("DLS Proj Cos-Only",     "proj_cos_only"),
    ("DLS Proj Rank-Only",    "proj_rank_only"),
    ("PDF Cos-Only",          "masked_cos_only"),
    ("PDF Rank-Only",         "masked_rank_only"),
    ("PDF Proj Cos-Only",     "masked_proj_cos_only"),
    ("PDF Proj Rank-Only",    "masked_proj_rank_only"),
]

def get_max_safe_score(results_dir: Path, trait: str, method: str) -> tuple[float, float, float, float]:
    best_score = 0.0
    best_alpha = np.nan
    best_ppl = np.nan
    best_coherence = np.nan
    
    trait_dir = results_dir / trait
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            
        if csv_path.exists():
            try:
                # Load CSV
                df = pd.read_csv(csv_path)
                if "dyn_score" in df.columns:
                    # In some earlier scripts, it might be called dyn_score or fusion_score
                    score_col = "dyn_score"
                elif "fusion_score" in df.columns:
                    score_col = "fusion_score"
                else:
                    score_col = df.columns[2] # Fallback
                
                df[score_col] = df[score_col].replace(0, np.nan)
                mean_score = df[score_col].mean()
                
                ppl_col = "dyn_ppl" if "dyn_ppl" in df.columns else "fusion_ppl"
                mean_ppl = df[ppl_col].mean()
                max_ppl = df[ppl_col].max()
                
                reason_col = "dyn_reason" if "dyn_reason" in df.columns else "fusion_reason"
                if reason_col in df.columns:
                    coherence_rate = df[reason_col].str.contains("Coherence: Yes", case=False, na=False).mean()
                else:
                    coherence_rate = 1.0
                
                # N-gram-free Strict Safety check
                is_safe = (
                    mean_ppl <= 25.0 and
                    coherence_rate >= 0.8 and
                    max_ppl <= 25.0
                )
                
                if is_safe:
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = val
                        best_ppl = mean_ppl
                        best_coherence = coherence_rate
            except Exception:
                pass
    return best_score, best_alpha, best_ppl, best_coherence

results_dir = Path("exp_steering_dyn_layer_raw/results")

print("--- [NEW SAFETY (N-gram-free)] Trait-by-Trait Max Safe Scores (Fixed-Layer DLS) ---")
all_results = {m[0]: [] for m in METHODS}
for trait in TRAITS:
    print(f"\nTrait: {trait.capitalize()}")
    for display_name, loader_key in METHODS:
        score, alpha, ppl, coherence = get_max_safe_score(results_dir, trait, loader_key)
        all_results[display_name].append(score)
        print(f"  {display_name:22s}: Score={score:.2f} (alpha={alpha}, PPL={ppl:.1f}, Coherence={coherence*100:.1f}%)")

print("\n--- [NEW SAFETY (N-gram-free)] Average Max Safe Scores Across All Traits (Fixed-Layer DLS) ---")
for display_name, scores in all_results.items():
    valid_scores = [s for s in scores if s > 0.0]
    avg_score = np.mean(valid_scores) if valid_scores else 0.0
    print(f"  {display_name:22s}: Avg Score={avg_score:.2f}")
