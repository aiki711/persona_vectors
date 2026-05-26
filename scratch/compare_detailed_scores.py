import pandas as pd
import numpy as np

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

output = []

for trait in TRAITS:
    output.append(f"\n==================== {trait.upper()} ====================")
    output.append(f"Val   | Logit_Diff (PPL) | Proj-Prior (PPL)")
    output.append("-" * 50)
    for val in VALS:
        p_logit = f"exp_steering_dyn_layer_all_layers_midpoint/results/{trait}/scores_logit_diff_Val{val}.csv"
        p_proj = f"exp_steering_dyn_layer_proj_prior/results/{trait}/scores_proj_prior_Val{val}.csv"
        
        logit_str = "N/A"
        proj_str = "N/A"
        
        try:
            df = pd.read_csv(p_logit)
            logit_str = f"{df['dyn_score'].mean():.2f} ({df['dyn_ppl'].mean():.1f})"
        except:
            pass
            
        try:
            df = pd.read_csv(p_proj)
            proj_str = f"{df['dyn_score'].mean():.2f} ({df['dyn_ppl'].mean():.1f})"
        except:
            pass
            
        output.append(f"{val:4.1f}  | {logit_str:16} | {proj_str:16}")

with open("scratch/comparison_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output) + "\n")
print("DONE")
