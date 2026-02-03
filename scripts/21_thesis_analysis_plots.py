
import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
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
            basename = os.path.basename(f)
            if "_base_" in basename:
                df['split'] = 'base'
            elif "_instruct_" in basename:
                df['split'] = 'instruct'
            else:
                df['split'] = 'unknown'
            
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
    df_metrics['alpha_total'] = df_metrics['alpha_total'].astype(float)
    df_scores['alpha_total'] = df_scores['alpha_total'].astype(float)
    
    merge_keys = ['model_name', 'split', 'trait', 'prompt', 'alpha_total']
    
    # Check for missing keys
    missing_keys = [k for k in merge_keys if k not in df_metrics.columns or k not in df_scores.columns]
    if missing_keys:
        print(f"Error: Missing merge keys: {missing_keys}")
        return pd.DataFrame()

    # Pre-merge check
    df_metrics = df_metrics.drop_duplicates(subset=merge_keys)
    df_scores = df_scores.drop_duplicates(subset=merge_keys)

    print("Merging data...")
    df_merged = pd.merge(df_scores, df_metrics, on=merge_keys, how='inner')
    print(f"Merged Data Shape: {df_merged.shape}")
    
    return df_merged

def load_internal_probe_data(jsonl_glob):
    """
    Load raw probe results from JSONL files to get real Cosine Similarity (s_avg).
    """
    files = glob.glob(jsonl_glob)
    print(f"Found {len(files)} probe JSONL files.")
    data_list = []
    
    for f in files:
        try:
            # Extract metadata from filename/path
            # Expected path: exp/{model}/results_*/{tag}_{split}_{trait}_probe_results.jsonl
            basename = os.path.basename(f)
            path_parts = f.split(os.sep)
            
            model_name = "unknown"
            if 'exp' in path_parts:
                idx = path_parts.index('exp')
                if idx + 1 < len(path_parts):
                    model_name = path_parts[idx+1]
            
            # Infer split
            split = "unknown"
            if "_base_" in basename:
                split = "base"
            elif "_instruct_" in basename:
                split = "instruct"
                
            # Infer trait (optional, but good for verification)
            # trait is usually in the filename
            
            # Read JSONL
            with open(f, 'r') as fh:
                for line in fh:
                    if not line.strip(): continue
                    try:
                        row = json.loads(line)
                        # We need: alpha_total, s_avg, prompt (x), trait
                        # row keys: i, trait, layers, alpha_total, alpha_mode, x, y, s_avg, s0_avg...
                        
                        rec = {
                            'model_name': model_name,
                            'split': split,
                            'trait': row.get('trait'),
                            'prompt': row.get('x'),
                            'alpha_total': float(row.get('alpha_total', 0)),
                            'internal_score': float(row.get('s_avg', 0))
                        }
                        data_list.append(rec)
                    except Exception:
                        continue
                        
        except Exception as e:
            print(f"Error reading JSONL file {f}: {e}")
            
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    return df

def plot_breaking_point(df, output_dir):
    """
    1. Breaking Point Analysis (External Score vs PPL)
    """
    print("Generating Breaking Point Plots...")
    os.makedirs(os.path.join(output_dir, "breaking_point"), exist_ok=True)
    
    groups = df.groupby(['model_name', 'split', 'trait'])
    
    for (model, split, trait), group in groups:
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
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        out_path = os.path.join(output_dir, "breaking_point", f"{model}_{split}_{trait}_breaking_point.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def plot_breaking_point_internal(df, output_dir):
    """
    Breaking Point Analysis: Internal Score (Cosine Sim) vs Perplexity
    """
    print("Generating Internal Breaking Point Plots...")
    out_subdir = os.path.join(output_dir, "breaking_point_internal")
    os.makedirs(out_subdir, exist_ok=True)
    
    groups = df.groupby(['model_name', 'split', 'trait'])
    
    for (model, split, trait), group in groups:
        if 'internal_score' not in group.columns or group['internal_score'].isnull().all():
            continue
            
        agg_data = group.groupby('alpha_total').agg({
            'internal_score': 'mean',
            'perplexity': 'mean'
        }).reset_index()
        
        if agg_data.empty:
            continue
            
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = 'tab:purple'
        ax1.set_xlabel('Steering Intensity (α)')
        ax1.set_ylabel('Internal Similarity (Cosine)', color=color1, fontsize=12)
        ax1.plot(agg_data['alpha_total'], agg_data['internal_score'], color=color1, marker='^', label='Internal Sim')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Perplexity (PPL)', color=color2, fontsize=12)
        ax2.plot(agg_data['alpha_total'], agg_data['perplexity'], color=color2, marker='x', linestyle='--', label='PPL')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f"Internal Breaking Point: {model} ({split}) - {trait.capitalize()}")
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        out_path = os.path.join(out_subdir, f"{model}_{split}_{trait}_breaking_point_internal.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def plot_disentanglement(df, output_dir):
    """
    2. Disentanglement Analysis
    """
    print("Generating Disentanglement Plots...")
    os.makedirs(os.path.join(output_dir, "disentanglement"), exist_ok=True)

    groups = df.groupby(['model_name', 'split', 'trait'])

    for (model, split, trait), group in groups:
        score_col = f'score_{trait}'
        
        # Safe baseline retrieval: Get alpha=0 rows, map prompt->score
        baseline_map = {}
        # Ensure we have alpha=0
        zeros = group[group['alpha_total'] == 0]
        if not zeros.empty:
            # Drop duplicates if any unique prompt
            zeros = zeros.drop_duplicates(subset=['prompt'])
            baseline_map = zeros.set_index('prompt')[score_col].to_dict()
        
        if not baseline_map:
            continue
            
        # Calculate Delta
        # This lambda might fail if prompt not in map (e.g. slight mismatch?)
        # Use map with fillna? or merge
        
        # Let's use merge for robustness
        baseline_df = zeros[['prompt', score_col]].rename(columns={score_col: 'baseline_score'})
        
        # We merge back to group
        # Note: 'group' already contains the zeros rows, but that's fine.
        merged_group = pd.merge(group, baseline_df, on='prompt', how='inner')
        merged_group['delta_score'] = merged_group[score_col] - merged_group['baseline_score']
        
        plot_data = merged_group[merged_group['alpha_total'] != 0].copy()
        
        if plot_data.empty:
            continue

        plt.figure(figsize=(8, 8))
        
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
        plt.axvline(1.0, color='gray', linestyle='--') 
        
        plt.xlabel('Semantic Similarity (1.0 = Same Meaning)')
        plt.ylabel('Δ Personality Score')
        plt.title(f"Disentanglement: {model} ({split}) - {trait.capitalize()}\n(Right-Up is Ideal)")
        
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlim(0, 1.1)
        
        stats_max = plot_data['delta_score'].max() if not plot_data['delta_score'].empty else 1.0
        plt.text(0.1, stats_max * 0.9, "Failure:\nChanged Context", color='red', fontsize=10)
        plt.text(0.8, stats_max * 0.9, "Success:\nPreserved Meaning", color='green', fontsize=10)

        out_path = os.path.join(output_dir, "disentanglement", f"{model}_{split}_{trait}_disentanglement.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def plot_efficiency(df, output_dir):
    """
    3. Efficiency Analysis
    """
    print("Generating Efficiency Plots...")
    os.makedirs(os.path.join(output_dir, "efficiency"), exist_ok=True)
    
    groups = df.groupby('trait') 
    
    for trait, group in groups:
        plt.figure(figsize=(10, 7))
        
        score_col = f'score_{trait}'
        
        agg = group.groupby(['model_name', 'split', 'alpha_total']).agg({
            'normalized_distance': 'mean',
            score_col: 'mean'
        }).reset_index()
        
        configs = agg[['model_name', 'split']].drop_duplicates()
        
        for _, row in configs.iterrows():
            m, s = row['model_name'], row['split']
            subset = agg[(agg['model_name'] == m) & (agg['split'] == s)].sort_values('normalized_distance')
            
            label = f"{m} ({s})"
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
    parser.add_argument("--jsonl_glob", required=False, help="Glob pattern for PROBE JSONL files (optional)")
    parser.add_argument("--out_dir", default="analysis_results/thesis_plots")
    args = parser.parse_args()
    
    df = load_data(args.metrics_glob, args.score_glob)
    if df.empty:
        print("Aborting: No data found.")
        return

    if args.jsonl_glob:
        print("Loading internal probe data (JSONL)...")
        df_internal = load_internal_probe_data(args.jsonl_glob)
        if not df_internal.empty:
            print(f"Merging internal data. Internal shape: {df_internal.shape}")
            # Key: model_name, split, trait, prompt, alpha_total
            # We must aggregate duplicates in internal (layer-wise?) 
            # Wait, probe JSONL usually has one row per sample per alpha?
            # It has 's_avg' which IS the average across layers.
            # So one row per (prompt, alpha).
            
            # Check unique keys
            keys = ['model_name', 'split', 'trait', 'prompt', 'alpha_total']
            # Ensure float alpha
            df_internal['alpha_total'] = df_internal['alpha_total'].astype(float)
            
            # Drop duplicates if any (e.g. rerun)
            df_internal = df_internal.drop_duplicates(subset=keys)
            
            df = pd.merge(df, df_internal, on=keys, how='left')
        else:
            print("No internal data loaded.")
        
    print(f"Loaded {len(df)} rows.")
    
    plot_breaking_point(df, args.out_dir)
    if 'internal_score' in df.columns:
        plot_breaking_point_internal(df, args.out_dir)
        
    plot_disentanglement(df, args.out_dir)
    plot_efficiency(df, args.out_dir)
    
    print(f"All plots saved to {args.out_dir}")

if __name__ == "__main__":
    main()
