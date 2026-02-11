
import pandas as pd
import glob

# Files to check
files = [
    'exp_L10-30/mistral_7b/results_advice/mistral_7b_base_personality_scores.csv',
    'exp_L10-30/falcon3_7b/results_advice/falcon3_7b_base_personality_scores.csv'
]

for f in files:
    print(f"\n--- Checking {f} ---")
    try:
        # load full file but only needed cols to be fast? 
        # Actually file size is small enough (~1000 rows). 
        # The previous script might have hung on something else or just was silent.
        df = pd.read_csv(f)
        
        # Rename if needed
        if 'score_LABEL_0' in df.columns:
             df.rename(columns={'score_LABEL_0': 'score_extraversion', 
                                'score_LABEL_1': 'score_neuroticism', 
                                'score_LABEL_2': 'score_agreeableness', 
                                'score_LABEL_3': 'score_conscientiousness', 
                                'score_LABEL_4': 'score_openness'}, inplace=True)

        traits = df['trait'].unique()
        print(f"Traits: {traits}")
        
        # Check one trait
        target_trait = traits[0] 
        print(f"Analyzing trait: {target_trait}")
        
        subset = df[df['trait'] == target_trait]
        target_score_col = f"score_{target_trait}"
        
        print(f"Mean {target_score_col} by alpha:")
        print(subset.groupby('alpha_total')[target_score_col].mean())
        
    except Exception as e:
        print(f"Error: {e}")
