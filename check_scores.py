
import pandas as pd
import sys
import os

def check_scores():
    csv_path = "exp_personality_L10-30/mistral_7b/scores/personality_scores_llm_openness.csv"
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("\n=== Score Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for expected columns
    if 'alpha_total' in df.columns and 'score_llm' in df.columns:
        print("\n=== Score Stats by Alpha ===")
        # Group by alpha and calculate mean/std/count
        stats = df.groupby('alpha_total')['score_llm'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(stats)
        
        # Check correlation
        correlation = df['alpha_total'].corr(df['score_llm'])
        print(f"\nCorrelation (Alpha vs Score): {correlation:.4f}")
    else:
        print("\nUsing columns from CSV:")
        print(df.columns.tolist())
        print(df.head())

if __name__ == "__main__":
    check_scores()
