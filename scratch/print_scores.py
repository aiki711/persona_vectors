import pandas as pd
import numpy as np

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

output = []
output.append("=== Proj-Prior DLS Averages (All Traits Avg) ===")

rows_score = []
rows_ppl = []

for val in VALS:
    scores = []
    ppls = []
    for trait in TRAITS:
        p = f"exp_steering_dyn_layer_proj_prior/results/{trait}/scores_proj_prior_Val{val}.csv"
        try:
            df = pd.read_csv(p)
            scores.append(df['dyn_score'].mean())
            ppls.append(df['dyn_ppl'].mean())
        except Exception as e:
            pass
    avg_score = np.mean(scores) if scores else float("nan")
    avg_ppl = np.mean(ppls) if ppls else float("nan")
    output.append(f"Val {val:4.1f}: Score={avg_score:.2f}, PPL={avg_ppl:.2f}")

# Detail table per trait
output.append("\n=== Proj-Prior DLS Detailed Scores per Trait ===")
header = "Val   | " + " | ".join(t[:8] for t in TRAITS)
output.append(header)
output.append("-" * len(header))
for val in VALS:
    line = f"{val:4.1f} | "
    parts = []
    for trait in TRAITS:
        p = f"exp_steering_dyn_layer_proj_prior/results/{trait}/scores_proj_prior_Val{val}.csv"
        try:
            df = pd.read_csv(p)
            parts.append(f"{df['dyn_score'].mean():.2f} (PPL={df['dyn_ppl'].mean():.1f})")
        except:
            parts.append("N/A")
    line += " | ".join(parts)
    output.append(line)

with open("scratch/print_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output) + "\n")
print("DONE")
