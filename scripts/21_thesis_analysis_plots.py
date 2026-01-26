
import pandas as pd
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(metrics_glob, score_glob):
    """
    Loads and merges personality scores and text metrics.
    """
    # 1. Load Metrics
    metric_files = glob.glob(metrics_glob)
    df_metrics_list = []
    print(f"Found {len(metric_files)} metric files.")
    for f in metric_files:
        try:
            df = pd.read_csv(f)
            # Infer split/model from filename if possible, otherwise rely on existing columns
            # Assuming standard naming: {model}_{split}_text_metrics.csv
            basename = os.path.basename(f)
            if "_base_" in basename:
                df['split'] = 'base'
            elif "_instruct_" in basename:
                df['split'] = 'instruct'
            else:
                df['split'] = 'unknown'
            
            # Extract model name (parent dir name usually)
            # dirname = exp/{model_name}/...
            model_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
            df['model_name'] = model_name
            
            df_metrics_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_metrics_list:
        print("No metric files loaded.")
        return pd.DataFrame()
    
    df_metrics = pd.concat(df_metrics_list, ignore_index=True)
    
    # 2. Load Scores
    score_files = glob.glob(score_glob)
    df_scores_list = []
    print(f"Found {len(score_files)} score files.")
    for f in score_files:
        try:
            df = pd.read_csv(f)
            if 'x' in df.columns and 'prompt' not in df.columns:
                df.rename(columns={'x': 'prompt'}, inplace=True)
            
            basename = os.path.basename(f)
            if "_base_" in basename:
                df['split'] = 'base'
            elif "_instruct_" in basename:
                df['split'] = 'instruct'
            else:
                df['split'] = 'unknown'

            model_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
            df['model_name'] = model_name

            df_scores_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_scores_list:
        print("No score files loaded.")
        return pd.DataFrame()

    df_scores = pd.concat(df_scores_list, ignore_index=True)
    
    # Rename score columns
    label_mapping = {
        'score_LABEL_0': 'score_extraversion',
        'score_LABEL_1': 'score_neuroticism',
        'score_LABEL_2': 'score_agreeableness',
        'score_LABEL_3': 'score_conscientiousness',
        'score_LABEL_4': 'score_openness'
    }
    df_scores.rename(columns=label_mapping, inplace=True)

    # 3. Merge
    # Keys: model_name, split, trait, prompt, alpha_total
    # Ensure alpha_total is float
    df_metrics['alpha_total'] = df_metrics['alpha_total'].astype(float)
    df_scores['alpha_total'] = df_scores['alpha_total'].astype(float)
    
    merge_keys = ['model_name', 'split', 'trait', 'prompt', 'alpha_total']
    
    # Check for missing keys
    missing_keys = [k for k in merge_keys if k not in df_metrics.columns or k not in df_scores.columns]
    if missing_keys:
        print(f"Error: Missing merge keys: {missing_keys}")
        print("Metrics cols:", df_metrics.columns)
        print("Scores cols:", df_scores.columns)
        return pd.DataFrame()

    # Pre-merge check: drop duplicates
    df_metrics = df_metrics.drop_duplicates(subset=merge_keys)
    df_scores = df_scores.drop_duplicates(subset=merge_keys)

    print("Merging data...")
    df_merged = pd.merge(df_scores, df_metrics, on=merge_keys, how='inner')
    print(f"Merged Data Shape: {df_merged.shape}")
    
    return df_merged

def plot_breaking_point(df, output_dir):
    """
    1. Breaking Point Analysis
    Graph: Dual Axis Line Plot
    X: Alpha
    Y1: Personality Score (External Score)
    Y2: Perplexity (PPL)
    """
    print("Generating Breaking Point Plots...")
    os.makedirs(os.path.join(output_dir, "breaking_point"), exist_ok=True)
    
    # Analyze per Model / Split / Trait
    groups = df.groupby(['model_name', 'split', 'trait'])
    
    for (model, split, trait), group in groups:
        # Calculate means per alpha
        agg_data = group.groupby('alpha_total').agg({
            f'score_{trait}': 'mean',
            'perplexity': 'mean'
        }).reset_index()
        
        if agg_data.empty:
            continue
            
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Steering Intensity (α)')
        ax1.set_ylabel('Personality Score', color=color1, fontsize=12)
        ax1.plot(agg_data['alpha_total'], agg_data[f'score_{trait}'], color=color1, marker='o', label='Score')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Perplexity (PPL)', color=color2, fontsize=12)
        ax2.plot(agg_data['alpha_total'], agg_data['perplexity'], color=color2, marker='x', linestyle='--', label='PPL')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f"Breaking Point Analysis: {model} ({split}) - {trait.capitalize()}")
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        out_path = os.path.join(output_dir, "breaking_point", f"{model}_{split}_{trait}_breaking_point.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def plot_disentanglement(df, output_dir):
    """
    2. Disentanglement Analysis
    Graph: Scatter Plot
    X: Semantic Similarity
    Y: Delta Score (Score - Score at alpha=0)
    """
    print("Generating Disentanglement Plots...")
    os.makedirs(os.path.join(output_dir, "disentanglement"), exist_ok=True)

    # Calculate Delta Score
    # We need baseline (alpha=0) score for each prompt
    # Pivot to get alpha=0 score
    base_scores = df[df['alpha_total'] == 0].set_index(['model_name', 'split', 'trait', 'prompt'])
    
    # We need access to score_{trait}, but trait is variable.
    # Let's iterate groups first to simplify
    groups = df.groupby(['model_name', 'split', 'trait'])

    for (model, split, trait), group in groups:
        score_col = f'score_{trait}'
        
        # Filter 0 alpha for this group
        zeros = group[group['alpha_total'] == 0].set_index('prompt')[score_col]
        
        # Calculate Delta
        # Map baseline score to the group rows
        group['baseline_score'] = group['prompt'].map(zeros)
        group['delta_score'] = group[score_col] - group['baseline_score']
        
        # Filter out alpha=0 points for the plot (or keep them at 0,0)
        # We include negative alphas now as requested.
        plot_data = group[group['alpha_total'] != 0].copy()
        
        if plot_data.empty:
            continue

        plt.figure(figsize=(8, 8))
        
        # Scatter
        scatter = plt.scatter(
            plot_data['semantic_similarity'], 
            plot_data['delta_score'], 
            c=plot_data['alpha_total'], 
            cmap='viridis', 
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
        
        plt.colorbar(scatter, label='Alpha')
        
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(1.0, color='gray', linestyle='--') # Ideal similarity
        
        plt.xlabel('Semantic Similarity (1.0 = Same Meaning)')
        plt.ylabel('Δ Personality Score')
        plt.title(f"Disentanglement: {model} ({split}) - {trait.capitalize()}\n(Right-Up is Ideal)")
        
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlim(0, 1.1)
        # plt.ylim(-1, 1) # Auto limit usually better unless user specified fixed range
        
        # Quadrant Annotation
        plt.text(0.1, plot_data['delta_score'].max() * 0.9, "Failure:\nChanged Context", color='red', fontsize=10)
        plt.text(0.8, plot_data['delta_score'].max() * 0.9, "Success:\nPreserved Meaning", color='green', fontsize=10)

        out_path = os.path.join(output_dir, "disentanglement", f"{model}_{split}_{trait}_disentanglement.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def plot_efficiency(df, output_dir):
    """
    3. Efficiency Analysis
    Graph: Trade-off Curve
    X: Normalized Edit Distance
    Y: Personality Score
    """
    print("Generating Efficiency Plots...")
    os.makedirs(os.path.join(output_dir, "efficiency"), exist_ok=True)
    
    groups = df.groupby('trait') # Compare models on the same plot?
    # User said "Compare models". So one plot per Trait, multiple lines for models.
    
    for trait, group in groups:
        plt.figure(figsize=(10, 7))
        
        score_col = f'score_{trait}'
        
        # Aggregate by model, split, alpha
        agg = group.groupby(['model_name', 'split', 'alpha_total']).agg({
            'normalized_distance': 'mean',
            score_col: 'mean'
        }).reset_index()
        
        # For each model/split configuration, plot a line
        configs = agg[['model_name', 'split']].drop_duplicates()
        
        for _, row in configs.iterrows():
            m, s = row['model_name'], row['split']
            subset = agg[(agg['model_name'] == m) & (agg['split'] == s)].sort_values('normalized_distance')
            
            label = f"{m} ({s})"
            # Use alpha values to mark points if needed, but for trade-off curve just line is good
            plt.plot(subset['normalized_distance'], subset[score_col], marker='o', markersize=4, label=label)

        plt.xlabel('Normalized Edit Distance (Cost)')
        plt.ylabel('Personality Score (Benefit)')
        plt.title(f"Efficiency Trade-off: {trait.capitalize()}\n(Steeper Slope = Better Efficiency)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = os.path.join(output_dir, "efficiency", f"all_models_{trait}_efficiency.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_glob", required=True, help="Glob pattern for score files")
    parser.add_argument("--metrics_glob", required=True, help="Glob pattern for metrics files")
    parser.add_argument("--out_dir", default="analysis_results/thesis_plots")
    args = parser.parse_args()
    
    df = load_data(args.metrics_glob, args.score_glob)
    if df.empty:
        print("Aborting: No data found.")
        return
        
    print(f"Loaded {len(df)} rows.")
    
    plot_breaking_point(df, args.out_dir)
    plot_disentanglement(df, args.out_dir)
    plot_efficiency(df, args.out_dir)
    
    print(f"All plots saved to {args.out_dir}")

if __name__ == "__main__":
    main()
