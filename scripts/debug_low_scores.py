
import pandas as pd
import argparse
import os
import json

def analyze_model(tag, trait, mode="base", base_dir="exp_pca_L10-30"):
    print(f"=== Analyzing {tag} / {mode} / {trait} ===")
    
    # Paths
    results_dir = os.path.join(base_dir, tag, "results_writing")
    jsonl_path = os.path.join(results_dir, f"{tag}_{mode}_{trait}_probe_results.jsonl")
    metrics_path = os.path.join(results_dir, f"{tag}_{mode}_text_metrics.csv")
    scores_path = os.path.join(results_dir, f"{tag}_{mode}_personality_scores.csv")
    
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    # Load Data
    print("Loading data...")
    try:
        df_text = pd.read_json(jsonl_path, lines=True)
        # Metrics and Scores are usually for ALL output, need to merge carefully or filter
        # actually metrics/scores csvs are usually aligned by index if generated sequentially
        # But let's check columns.
        
        df_met = pd.read_csv(metrics_path)
        df_scr = pd.read_csv(scores_path)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # The CSVs might contain all traits mixed if run_text_analysis was run on "alltraits.jsonl"
    # We need to filter df_met and df_scr for the rows corresponding to our trait if they contain a 'trait' column
    # or just merge by 'generation' text if possible? 
    # Usually they have 'trait' and 'alpha_total' columns.
    
    # Filter for current trait
    if 'trait' in df_met.columns:
        df_met = df_met[df_met['trait'] == trait].copy()
    if 'trait' in df_scr.columns:
        df_scr = df_scr[df_scr['trait'] == trait].copy()
        
    print(f"Data shapes: Text={df_text.shape}, Met={df_met.shape}, Scr={df_scr.shape}")
    
    # Merge (assuming aligned by sort or index, but safer to merge on prompt + alpha if available)
    # df_text has 'alpha_total', 'prompt', 'generation'
    # df_met has 'alpha_total', 'prompt', 'generation' (maybe) output csv from 13_text_metrics has prompt/generation
    
    # Let's try to concat if lengths match, distinct files are per trait usually for probe_results
    # actually metrics/scores were generated from _alltraits.jsonl which is concatenated
    # So we need to match carefully.
    
    # For simplicity, let's look at df_scr (which comes from alltraits) for this trait
    # and print high alpha stats.
    
    # Group by Alpha
    stats = df_scr.groupby('alpha_total')[f'score_{trait}'].mean()
    print("\n--- Score vs Alpha ---")
    print(stats)
    
    # Now check Perplexity for this trait
    print("\n--- PPL vs Alpha ---")
    if 'perplexity' in df_met.columns:
        ppl_stats = df_met.groupby('alpha_total')['perplexity'].mean()
        print(ppl_stats)
    elif 'ppl' in df_met.columns:
        ppl_stats = df_met.groupby('alpha_total')['ppl'].mean()
        print(ppl_stats)
        
    # Correlation
    # We need to align them. 
    # Let's just grab high alpha rows from df_text (which has the text)
    # and print samples.
    
    print("\n--- Samples (High Positive Alpha) ---")
    max_alpha = df_text['alpha_total'].max()
    high_alpha_df = df_text[df_text['alpha_total'] == max_alpha]
    
    for i, row in high_alpha_df.head(3).iterrows():
        # Handle dict or list prompt
        prompt_txt = row.get('x', '') # Use 'x' as discovered
        gen_txt = row.get('y', '')    # Use 'y' as discovered

        print(f"\n[Alpha {row['alpha_total']}] Prompt: {prompt_txt[:50]}...")
        print(f"Gen: {gen_txt[:200]}...")

    print("\n--- Samples (Baseline Alpha 0.0) ---")
    baseline_df = df_text[df_text['alpha_total'].abs() < 0.01]
    
    for i, row in baseline_df.head(3).iterrows():
        # Handle dict or list prompt
        prompt_txt = row.get('x', '')
        gen_txt = row.get('y', '')

        print(f"\n[Alpha {row['alpha_total']}] Prompt: {prompt_txt[:50]}...")
        print(f"Gen: {gen_txt[:200]}...")

    print("\n--- Samples (Low Negative Alpha) ---")
    min_alpha = df_text['alpha_total'].min()
    low_alpha_df = df_text[df_text['alpha_total'] == min_alpha]
    
    for i, row in low_alpha_df.head(3).iterrows():
        # Handle dict or list prompt
        prompt_txt = row.get('x', '')
        gen_txt = row.get('y', '')

        print(f"\n[Alpha {row['alpha_total']}] Prompt: {prompt_txt[:50]}...")
        print(f"Gen: {gen_txt[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--trait", type=str, default="openness")
    parser.add_argument("--mode", type=str, default="base")
    args = parser.parse_args()
    
    analyze_model(args.tag, args.trait, args.mode)
