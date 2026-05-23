import pandas as pd
from pathlib import Path
import numpy as np

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def main():
    all_layers_dir = Path("exp_steering_dyn_layer_all_layers/results")
    fusion_dir = Path("exp_steering_dyn_ic_fusion/results")
    
    methods = {
        "DLS_logit_diff": (all_layers_dir, "logit_diff"),
        "DLS_anti_align": (all_layers_dir, "anti_alignment"),
        "DLS_relative": (all_layers_dir, "relative_anti_alignment"),
        "Fusion_Sigmoid": (fusion_dir, "fusion_sigmoid", "scores_fusion_sigmoid"),
        "Fusion_Plateau": (fusion_dir, "fusion_soft_plateau", "scores_fusion_soft_plateau"),
    }
    
    summary_data = []
    
    for val in VALS:
        row = {"val": val}
        for name, info in methods.items():
            dir_path = info[0]
            scores = []
            ppls = []
            for trait in TRAITS:
                trait_dir = dir_path / trait
                if len(info) == 2:
                    csv_path = trait_dir / f"scores_{info[1]}_Val{val}.csv"
                    if not csv_path.exists():
                        csv_path = trait_dir / f"scores_{info[1]}_Val{float(val)}.csv"
                else:
                    csv_path = trait_dir / f"{info[2]}_Val{val}.csv"
                    if not csv_path.exists():
                        csv_path = trait_dir / f"{info[2]}_Val{float(val)}.csv"
                
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        scores.append(df["dyn_score"].mean())
                        ppls.append(df["dyn_ppl"].mean())
                    except Exception:
                        pass
            if scores:
                row[f"{name}_score"] = np.mean(scores)
                row[f"{name}_ppl"] = np.mean(ppls)
            else:
                row[f"{name}_score"] = np.nan
                row[f"{name}_ppl"] = np.nan
        summary_data.append(row)
        
    summary_df = pd.DataFrame(summary_data)
    print("### Average Scores (Higher is better, target range ~4.0-5.0)")
    score_cols = ["val"] + [f"{name}_score" for name in methods.keys()]
    print(summary_df[score_cols].to_markdown(index=False))
    
    print("\n### Average Perplexities (Lower is better, <= 25.0 is safe)")
    ppl_cols = ["val"] + [f"{name}_ppl" for name in methods.keys()]
    print(summary_df[ppl_cols].to_markdown(index=False))

if __name__ == "__main__":
    main()
