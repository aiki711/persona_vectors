import numpy as np
import pandas as pd
from pathlib import Path

vals = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

results_base_dir = Path("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen")
new_results_dir = Path("exp_steering_dyn_layer_proj_prior/results")

def get_avg_metrics(method_name):
    target_dir = results_base_dir if method_name == "logit_diff" else new_results_dir
    res = {}
    for val in vals:
        val_scores = []
        val_ppls = []
        for trait in traits:
            csv_path = target_dir / trait / f"scores_{method_name}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = target_dir / trait / f"scores_{method_name}_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "dyn_score" in df.columns:
                        df["dyn_score"] = df["dyn_score"].replace(0, 1)
                    val_scores.append(df["dyn_score"].mean())
                    val_ppls.append(df["dyn_ppl"].mean())
                except Exception:
                    pass
        if val_scores:
            res[val] = (np.mean(val_scores), np.mean(val_ppls))
    return res

ld_res = get_avg_metrics("logit_diff")
co_res = get_avg_metrics("cos_only")
ro_res = get_avg_metrics("rank_only")

print("| 強度 (Val) | logit_diff | cos_only | rank_only (改善版) |")
print("|:---:|:---:|:---:|:---:|")
for val in vals:
    ld_str = f"{ld_res[val][0]:.2f} ({ld_res[val][1]:.1f})" if val in ld_res else "N/A"
    co_str = f"{co_res[val][0]:.2f} ({co_res[val][1]:.1f})" if val in co_res else "N/A"
    ro_str = f"{ro_res[val][0]:.2f} ({ro_res[val][1]:.1f})" if val in ro_res else "N/A"
    print(f"| **{val}** | {ld_str} | {co_str} | {ro_str} |")
