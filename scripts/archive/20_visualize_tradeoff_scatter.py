import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import seaborn as sns

def get_alpha_max(model_name):
    # Same map as other scripts for consistency
    ALPHA_MAX_MAP = {
        "mistral_7b": 5.5,
        "llama3_8b": 20,
        "olmo3_7b": 50,
        "qwen25_7b": 160,
        "gemma2_9b": 500,
        "falcon3_7b": 500
    }
    for key, val in ALPHA_MAX_MAP.items():
        if key in model_name:
            return val
    return 1.0

def load_data(score_path, distance_path):
    print(f"Loading scores: {score_path}")
    df_score = pd.read_csv(score_path)
    print(f"Loading distances: {distance_path}")
    df_dist = pd.read_csv(distance_path)
    
    # Standardize 'x' to 'prompt' in score df
    if 'x' in df_score.columns and 'prompt' not in df_score.columns:
        df_score.rename(columns={'x': 'prompt'}, inplace=True)
        
    # Standardize 'score' columns
    label_mapping = {
        'score_LABEL_0': 'score_extraversion',
        'score_LABEL_1': 'score_neuroticism',
        'score_LABEL_2': 'score_agreeableness',
        'score_LABEL_3': 'score_conscientiousness',
        'score_LABEL_4': 'score_openness'
    }
    df_score.rename(columns=label_mapping, inplace=True)
    
    # Merge
    # keys: trait, prompt, alpha_total
    # Check if 'split' is in columns, if so include it
    merge_keys = ['trait', 'prompt', 'alpha_total']
    if 'split' in df_score.columns and 'split' in df_dist.columns:
        merge_keys.append('split')
        
    # Ensure alpha_total is float
    df_score['alpha_total'] = df_score['alpha_total'].astype(float)
    df_dist['alpha_total'] = df_dist['alpha_total'].astype(float)
    
    # Identify target score columns
    score_cols = [c for c in df_score.columns if c.startswith('score_')]
    
    # Merge (inner join to keep valid pairs)
    merged = pd.merge(df_score, df_dist, on=merge_keys, how='inner', suffixes=('', '_dist'))
    
    return merged, score_cols

def plot_dual_axis(df, trait, model_name, split, output_path):
    """
    X: Alpha
    Y1 (Left): Score
    Y2 (Right): Normalized Distance
    """
    target_score_col = f"score_{trait}"
    if target_score_col not in df.columns:
        return

    subset = df[df['trait'] == trait].copy()
    if subset.empty:
        return
        
    # Sort by alpha
    subset.sort_values('alpha_total', inplace=True)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Shared X axis
    x = subset['alpha_total']
    
    # --- Y1: Score (Left) ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Alpha (Steering Intensity)')
    ax1.set_ylabel(f'Personality Score ({trait})', color=color1)
    
    # Scatter points for Score
    ax1.scatter(x, subset[target_score_col], color=color1, alpha=0.3, s=20, label='Score Samples')
    
    # Mean trend line
    mean_scores = subset.groupby('alpha_total')[target_score_col].mean()
    ax1.plot(mean_scores.index, mean_scores.values, color=color1, linewidth=2, label='Mean Score')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    # Remove hardcoded limit, let it autoscale or set based on min/max
    # ax1.set_ylim(0, 1.0) 
    ax1.grid(True, alpha=0.3)

    # --- Y2: Distance (Right) ---
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:red'
    ax2.set_ylabel('Normalized Edit Distance', color=color2)  # we already handled the x-label with ax1
    
    # Scatter points for Distance
    ax2.scatter(x, subset['normalized_distance'], color=color2, alpha=0.3, s=20, marker='x', label='Distance Samples')
    
    # Mean trend line
    mean_dist = subset.groupby('alpha_total')['normalized_distance'].mean()
    ax2.plot(mean_dist.index, mean_dist.values, color=color2, linewidth=2, linestyle='--', label='Mean Distance')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    # Distance is strictly 0-1, but autoscaling is safer or use 0-1.1
    # ax2.set_ylim(0, 1.1) 

    # Vertical line at Alpha=0
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)

    plt.title(f"Alpha vs Score & Edit Distance\nModel: {model_name} ({split}) | Trait: {trait.capitalize()}")
    
    # Legends
    # Ensure legends from both axes are shown
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_tradeoff(df, trait, model_name, split, output_path):
    """
    X: Normalized Distance
    Y: Score
    Color: Alpha
    """
    target_score_col = f"score_{trait}"
    if target_score_col not in df.columns:
        return

    subset = df[df['trait'] == trait].copy()
    if subset.empty:
        return
        
    plt.figure(figsize=(8, 6))
    
    # Plot scatter with colormap
    sc = plt.scatter(
        subset['normalized_distance'], 
        subset[target_score_col], 
        c=subset['alpha_total'], 
        cmap='coolwarm', 
        alpha=0.6, 
        edgecolors='k', 
        linewidth=0.3
    )
    
    plt.colorbar(sc, label='Alpha (Steering Intensity)')
    
    plt.xlabel('Normalized Edit Distance (Cost)')
    plt.ylabel(f'Personality Score ({trait})')
    plt.title(f"Trade-off Curve: Score vs Distance\nModel: {model_name} ({split}) | Trait: {trait.capitalize()}")
    
    # Remove hardcoded limits to show all points
    # plt.ylim(0, 1.0)
    # plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add annotations for extreme alphas (max/min)
    # Find point with max alpha
    max_alpha_row = subset.loc[subset['alpha_total'].idxmax()]
    plt.annotate(f"α={max_alpha_row['alpha_total']}", 
                 (max_alpha_row['normalized_distance'], max_alpha_row[target_score_col]),
                 xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="exp", help="Root search directory")
    parser.add_argument("--suffix", default="", help="Suffix for results dir (e.g. _L10-30)")
    args = parser.parse_args()
    
    # Find all score files
    score_pattern = os.path.join(args.root_dir, f"*/asst_pairwise_results{args.suffix}/*_personality_scores.csv")
    score_files = glob.glob(score_pattern)
    
    print(f"Found {len(score_files)} score files.")
    
    for score_file in score_files:
        dirname = os.path.dirname(score_file)
        basename = os.path.basename(score_file)
        
        # Infer Distance File Path
        # Naming convention: {model}_{split}_personality_scores.csv -> {model}_{split}_edit_distance.csv
        dist_basename = basename.replace("_personality_scores.csv", "_edit_distance.csv")
        dist_file = os.path.join(dirname, dist_basename)
        
        if not os.path.exists(dist_file):
            print(f"Skipping {basename}: Matching distance file not found: {dist_basename}")
            continue
            
        # Metadata Inference
        model_dir = os.path.dirname(dirname)
        model_name = os.path.basename(model_dir)
        
        if "_base_" in basename:
            split = "base"
        elif "_instruct_" in basename:
            split = "instruct"
        else:
            split = "unknown"
            
        # Load and Merge
        try:
            merged_df, score_cols = load_data(score_file, dist_file)
            if merged_df.empty:
                print(f"Skipping {basename}: Merged dataframe is empty.")
                continue
                
            # Create Plot Directories
            plot_dir_dual = os.path.join(dirname, "plots", "alpha_vs_score_dist")
            plot_dir_tradeoff = os.path.join(dirname, "plots", "tradeoff_dist_score")
            os.makedirs(plot_dir_dual, exist_ok=True)
            os.makedirs(plot_dir_tradeoff, exist_ok=True)
            
            # Identify traits present
            traits = merged_df['trait'].unique()
            
            for trait in traits:
                # Plot 1: Dual Axis
                out_dual = os.path.join(plot_dir_dual, f"{model_name}_{split}_{trait}_dual_axis.png")
                plot_dual_axis(merged_df, trait, model_name, split, out_dual)
                
                # Plot 2: Tradeoff
                out_trade = os.path.join(plot_dir_tradeoff, f"{model_name}_{split}_{trait}_tradeoff.png")
                plot_tradeoff(merged_df, trait, model_name, split, out_trade)
                
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
