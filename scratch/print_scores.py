import pandas as pd
import glob

output = []

output.append("--- DLS logit_diff (midpoint normalized) Extraversion ---")
for val in [0.5, 1.0, 2.0, 4.0]:
    p = f"exp_steering_dyn_layer_all_layers_midpoint/results/extraversion/scores_logit_diff_Val{val}.csv"
    try:
        df = pd.read_csv(p)
        output.append(f"Val {val}: score={df['dyn_score'].mean():.2f}, ppl={df['dyn_ppl'].mean():.2f}")
    except Exception as e:
        output.append(f"Val {val}: Not found")

output.append("\n--- DLS anti_alignment (midpoint normalized) Extraversion ---")
for val in [0.5, 1.0, 2.0, 4.0]:
    p = f"exp_steering_dyn_layer_all_layers_midpoint/results/extraversion/scores_anti_alignment_Val{val}.csv"
    try:
        df = pd.read_csv(p)
        output.append(f"Val {val}: score={df['dyn_score'].mean():.2f}, ppl={df['dyn_ppl'].mean():.2f}")
    except Exception as e:
        output.append(f"Val {val}: Not found")

output.append("\n--- Single Layer Extraversion (Layer 15) ---")
for val in [0.5, 1.0, 2.0, 4.0]:
    p = f"exp_steering_layer_analysis/results/extraversion/scores_layer_15_Val{val}.csv"
    try:
        df = pd.read_csv(p)
        output.append(f"Val {val}: score={df['const_score'].mean():.2f}, ppl={df['const_ppl'].mean():.2f}")
    except Exception as e:
        output.append(f"Val {val}: Not found")

with open("scratch/print_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output) + "\n")
print("DONE")
