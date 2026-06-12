import numpy as np
import pandas as pd
from pathlib import Path

vals = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
new_results_dir = Path("exp_steering_dyn_layer_proj_prior/results")

print("| 強度 (Val) | rank_only (改善版) |")
print("|:---:|:---:|")

for val in vals:
    val_scores = []
    val_ppls = []
    for trait in traits:
        csv_path = new_results_dir / trait / f"scores_rank_only_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = new_results_dir / trait / f"scores_rank_only_Val{val}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "dyn_score" in df.columns:
                    df["dyn_score"] = df["dyn_score"].replace(0, 1)
                val_scores.append(df["dyn_score"].mean())
                val_ppls.append(df["dyn_ppl"].mean())
            except Exception as e:
                pass
    if val_scores:
        avg_score = np.mean(val_scores)
        avg_ppl = np.mean(val_ppls)
        print(f"| **{val}** | {avg_score:.2f} ({avg_ppl:.1f}) |")
