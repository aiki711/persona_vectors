import numpy as np
import pandas as pd
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
METHODS = ["cos_only", "rank_only", "logit_diff"]

def analyze(results_dir: Path):
    rows = []
    for trait in TRAITS:
        for method in METHODS:
            best_safe_score = -1
            best_safe_val = None
            best_safe_ppl = None
            
            for val in VALS:
                csv_path = results_dir / trait / f"scores_{method}_Val{float(val)}.csv"
                if not csv_path.exists():
                    csv_path = results_dir / trait / f"scores_{method}_Val{val}.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        mean_score = df["dyn_score"].mean()
                        mean_ppl = df["dyn_ppl"].mean()
                        
                        if mean_ppl <= 25.0:
                            if mean_score > best_safe_score:
                                best_safe_score = mean_score
                                best_safe_val = val
                                best_safe_ppl = mean_ppl
                    except Exception as e:
                        pass
            rows.append({
                "Trait": trait,
                "Method": method,
                "Best Safe Val": best_safe_val if best_safe_val is not None else "N/A",
                "Best Safe Score": round(best_safe_score, 3) if best_safe_score >= 0 else "N/A",
                "Best Safe PPL": round(best_safe_ppl, 2) if best_safe_ppl is not None else "N/A",
            })
    return pd.DataFrame(rows)

print("=== NORM RUN ANALYSIS ===")
norm_df = analyze(Path("exp_steering_dyn_layer_norm/results"))
print(norm_df.to_markdown(index=False))

print("\n=== RAW RUN ANALYSIS ===")
raw_df = analyze(Path("exp_steering_dyn_layer_raw/results"))
print(raw_df.to_markdown(index=False))
