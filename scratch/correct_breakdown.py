import pandas as pd
from pathlib import Path

base_dir = Path("/home/s2550009/persona_vectors/exp_token_intensity")

for sweep_name in ["exp_v03_rise_sweep", "exp_v03_fall_sweep"]:
    sweep_dir = base_dir / sweep_name
    prefix = "Rise" if "rise" in sweep_name else "Fall"
    print(f"\n=================== Correct {prefix} Sweep Breakdown ===================")
    
    rows = []
    for trait in ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]:
        trait_dir = sweep_dir / trait
        for f in trait_dir.glob("*.csv"):
            # parse filename to get theta and k
            # e.g., scores_masked_proj_rank_theta_0.1_99.0_k_2.0_1.0_entropy_plateau_Val5.0.csv
            name = f.stem
            parts = name.split("_")
            try:
                if prefix == "Rise":
                    th = float(parts[4])
                    k_val = float(parts[7])
                else:
                    th = float(parts[5])
                    k_val = float(parts[8])
            except Exception:
                continue
            
            df = pd.read_csv(f)
            sc = df["dyn_score"].mean()
            ppl = df["dyn_ppl"][pd.Series(df["dyn_ppl"]).notna()].mean()
            rows.append({
                "sweep": prefix,
                "trait": trait,
                "theta": th,
                "k": k_val,
                "score": sc,
                "ppl": ppl,
                "filename": f.name
            })
            
    df_all = pd.DataFrame(rows)
    for trait in ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]:
        sub = df_all[df_all["trait"] == trait].sort_values(by=["score", "ppl"], ascending=[False, True])
        best = sub.iloc[0]
        print(f"{trait.capitalize():18s}: theta={best['theta']:.1f}, k={best['k']:.2f} => Score: {best['score']:.2f}, PPL: {best['ppl']:.2f} ({best['filename']})")
