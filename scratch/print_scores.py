import pandas as pd
df = pd.read_csv('exp_steering_dyn_ic_fusion_midpoint/results/neuroticism/scores_fusion_soft_plateau_Val1.5.csv')
with open('scratch/print_out.txt', 'w', encoding='utf-8') as f:
    f.write(df[['base_score', 'dyn_score', 'base_ppl', 'dyn_ppl']].head(10).to_string() + '\n')
print("DONE")
