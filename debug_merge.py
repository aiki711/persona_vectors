import pandas as pd
import glob
import os

DIST_GLOB = "exp/mistral_7b/asst_pairwise_results/mistral_7b_base_edit_distance.csv"
SCORE_GLOB = "exp/mistral_7b/asst_pairwise_results/mistral_7b_base_personality_scores.csv"

def load_and_debug():
    print("Loading dist...")
    df_dist = pd.read_csv(DIST_GLOB)
    
    print("Loading score...")
    df_score = pd.read_csv(SCORE_GLOB)
    
    # Apply column rename fix
    if 'x' in df_score.columns and 'prompt' not in df_score.columns:
        print("Renaming 'x' to 'prompt'")
        df_score.rename(columns={'x': 'prompt'}, inplace=True)
        
    if 'generated_text' in df_dist.columns and 'y' not in df_dist.columns:
        print("Renaming 'generated_text' to 'y'")
        df_dist.rename(columns={'generated_text': 'y'}, inplace=True)

    merge_keys = ['trait', 'prompt', 'alpha_total']
    
    print(f"Dist columns: {df_dist.columns.tolist()}")
    print(f"Score columns: {df_score.columns.tolist()}")
    
    # Check sample values
    print("\nSample Dist Keys:")
    print(df_dist[merge_keys].head(3))
    print(df_dist[merge_keys].dtypes)
    
    # Find a specific prompt to check
    sample_prompt = df_dist['prompt'].iloc[0]
    sample_trait = df_dist['trait'].iloc[0]
    sample_alpha = df_dist['alpha_total'].iloc[0]
    
    print(f"\nSearching for match in Score for: {sample_trait}, {sample_prompt}, {sample_alpha}")
    
    mask = (df_score['prompt'] == sample_prompt) & (df_score['alpha_total'] == sample_alpha)
    msg_matches = df_score[mask]
    
    if msg_matches.empty:
        print("No matches for PROMPT and ALPHA found in Score.")
        # Check prompt existence
        if sample_prompt in df_score['prompt'].values:
            print("Prompt exists in Score.")
        else:
            print("Prompt DOES NOT exist in Score.")
            print(f"Dist prompt: '{sample_prompt}'")
            print(f"Score sample prompt: '{df_score['prompt'].iloc[0]}'")
    else:
        print(f"Found matches for Prompt+Alpha. Traits are: {msg_matches['trait'].unique()}")
        if sample_trait in msg_matches['trait'].values:
            print("Trait also matches! Merge should work.")
        else:
            print(f"Trait mismatch. Dist has '{sample_trait}', Score has {msg_matches['trait'].unique()}")

    # Try merge
    df_merged = pd.merge(df_dist, df_score, on=merge_keys, how='inner')
    print(f"\nMerged shape: {df_merged.shape}")

if __name__ == "__main__":
    load_and_debug()
