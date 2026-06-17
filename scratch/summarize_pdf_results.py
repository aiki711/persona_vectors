import numpy as np
import pandas as pd
from pathlib import Path

vals = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

pdf_results_dir = Path("exp_steering_dyn_layer_pdf/results")
unmasked_results_dir = Path("exp_steering_dyn_layer_proj_prior/results_test_unseen")

def get_avg_metrics(results_dir, method_name):
    res = {}
    for val in vals:
        val_scores = []
        val_ppls = []
        for trait in traits:
            csv_path = results_dir / trait / f"scores_{method_name}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = results_dir / trait / f"scores_{method_name}_Val{val}.csv"
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

# Unmasked results
unmasked_cos = get_avg_metrics(unmasked_results_dir, "cos_only")
unmasked_rank = get_avg_metrics(unmasked_results_dir, "rank_only")
unmasked_proj_cos = get_avg_metrics(unmasked_results_dir, "proj_cos_only")
unmasked_proj_rank = get_avg_metrics(unmasked_results_dir, "proj_rank_only")

# Masked (PDF) results
masked_cos = get_avg_metrics(pdf_results_dir, "masked_cos_only")
masked_rank = get_avg_metrics(pdf_results_dir, "masked_rank_only")
masked_proj_cos = get_avg_metrics(pdf_results_dir, "masked_proj_cos_only")
masked_proj_rank = get_avg_metrics(pdf_results_dir, "masked_proj_rank_only")

print("=== Aligned Comparison: Unmasked vs. Masked (PDF K=500) ===")
print("\n--- 1. Proj-Rank-Only (Unmasked vs. Masked) ---")
print("| 強度 (Val) | Unmasked (proj_rank_only) | Masked PDF (masked_proj_rank_only) |")
print("|:---:|:---:|:---:|")
for val in vals:
    unm_str = f"{unmasked_proj_rank[val][0]:.2f} ({unmasked_proj_rank[val][1]:.1f})" if val in unmasked_proj_rank else "N/A"
    msk_str = f"{masked_proj_rank[val][0]:.2f} ({masked_proj_rank[val][1]:.1f})" if val in masked_proj_rank else "N/A"
    print(f"| **{val}** | {unm_str} | {msk_str} |")

print("\n--- 2. Proj-Cos-Only (Unmasked vs. Masked) ---")
print("| 強度 (Val) | Unmasked (proj_cos_only) | Masked PDF (masked_proj_cos_only) |")
print("|:---:|:---:|:---:|")
for val in vals:
    unm_str = f"{unmasked_proj_cos[val][0]:.2f} ({unmasked_proj_cos[val][1]:.1f})" if val in unmasked_proj_cos else "N/A"
    msk_str = f"{masked_proj_cos[val][0]:.2f} ({masked_proj_cos[val][1]:.1f})" if val in masked_proj_cos else "N/A"
    print(f"| **{val}** | {unm_str} | {msk_str} |")

print("\n--- 3. Cos-Only (Unmasked vs. Masked) ---")
print("| 強度 (Val) | Unmasked (cos_only) | Masked PDF (masked_cos_only) |")
print("|:---:|:---:|:---:|")
for val in vals:
    unm_str = f"{unmasked_cos[val][0]:.2f} ({unmasked_cos[val][1]:.1f})" if val in unmasked_cos else "N/A"
    msk_str = f"{masked_cos[val][0]:.2f} ({masked_cos[val][1]:.1f})" if val in masked_cos else "N/A"
    print(f"| **{val}** | {unm_str} | {msk_str} |")
