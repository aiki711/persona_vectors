import pandas as pd
import numpy as np
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
METHODS = ["logit_diff", "cos_only", "rank_only", "cos_prior"]

base_dir = Path("exp_steering_dyn_layer_proj_prior/results_test_unseen")

print("Format: Score (Perplexity)")

for method in METHODS:
    print(f"\n==================== METHOD: {method} ====================")
    print("Val   | " + " | ".join(f"{t[:8]:8}" for t in TRAITS) + " | Average")
    print("-" * 75)
    for val in VALS:
        trait_scores = []
        trait_ppls = []
        parts = []
        for trait in TRAITS:
            p = base_dir / trait / f"scores_{method}_Val{val}.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    s = df["dyn_score"].mean()
                    ppl = df["dyn_ppl"].mean()
                    trait_scores.append(s)
                    trait_ppls.append(ppl)
                    parts.append(f"{s:.2f}({ppl:3.1f})")
                except Exception:
                    parts.append(" Error  ")
            else:
                parts.append("  N/A   ")
        
        if trait_scores:
            avg_s = np.mean(trait_scores)
            avg_ppl = np.mean(trait_ppls)
            avg_str = f"{avg_s:.2f}({avg_ppl:3.1f})"
        else:
            avg_str = "  N/A   "
        print(f"{val:4.1f}  | " + " | ".join(parts) + f" | {avg_str}")
