import pandas as pd
df = pd.read_csv("exp/extraversion_scores.csv")
print(df.groupby("alpha_total")["raw_score_extraversion"].mean())
print(df.groupby("alpha_total")["raw_score_extraversion"].std())
