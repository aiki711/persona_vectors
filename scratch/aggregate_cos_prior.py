import pandas as pd
import numpy as np
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def main():
    results_dir = Path("exp_steering_dyn_layer_proj_prior/results")
    
    # We want to aggregate:
    # 1. Per-trait score and ppl for each alpha
    # 2. Average score and ppl across all traits for each alpha
    
    aggregated = {val: {"scores": [], "ppls": []} for val in VALS}
    trait_data = {trait: {} for trait in TRAITS}
    
    for trait in TRAITS:
        trait_dir = results_dir / trait
        for val in VALS:
            csv_path = trait_dir / f"scores_cos_prior_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_cos_prior_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    score = df["dyn_score"].mean()
                    ppl = df["dyn_ppl"].mean()
                    trait_data[trait][val] = (score, ppl)
                    aggregated[val]["scores"].append(score)
                    aggregated[val]["ppls"].append(ppl)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
            else:
                print(f"File not found: {csv_path}")

    # Output markdown table for All Traits Average
    print("### 1. Cos-Prior DLS Averages (All Traits)")
    print("| Alpha | Score | PPL |")
    print("| :---: | :---: | :---: |")
    for val in VALS:
        scores = aggregated[val]["scores"]
        ppls = aggregated[val]["ppls"]
        avg_score = np.mean(scores) if scores else np.nan
        avg_ppl = np.mean(ppls) if ppls else np.nan
        print(f"| **{val}** | {avg_score:.2f} | {avg_ppl:.2f} |")
        
    print("\n### 2. Cos-Prior DLS Per-Trait Scores")
    header = "| Alpha | " + " | ".join(t.capitalize() for t in TRAITS) + " |"
    print(header)
    print("| :---: | " + " | ".join(":---:" for _ in TRAITS) + " |")
    for val in VALS:
        row = f"| **{val}**"
        for trait in TRAITS:
            val_data = trait_data[trait].get(val, (np.nan, np.nan))
            row += f" | {val_data[0]:.2f} (PPL={val_data[1]:.1f})"
        row += " |"
        print(row)

if __name__ == "__main__":
    main()
