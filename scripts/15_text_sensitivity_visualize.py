import pandas as pd
import numpy as np
import argparse
import os
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import pi

def get_split_from_filename(filepath):
    """ファイル名から base / instruct を判定する"""
    filename = os.path.basename(filepath)
    if "_base_" in filename:
        return "base"
    elif "_instruct_" in filename:
        return "instruct"
    return "unknown"

def load_data(metrics_dist_pattern, score_pattern):
    """CSVファイルを読み込み、split情報を付与して結合します"""
    
    # メトリクスデータ (edit_dist, perplexity, similarity)
    dist_files = glob.glob(metrics_dist_pattern)
    df_dist_list = []
    for f in dist_files:
        try:
            tmp = pd.read_csv(f)
            tmp['split'] = get_split_from_filename(f)
            df_dist_list.append(tmp)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_dist_list:
        print("No metrics files found.")
        return pd.DataFrame()
    
    df_dist = pd.concat(df_dist_list, ignore_index=True)

    # 性格スコアデータ
    score_files = glob.glob(score_pattern)
    df_score_list = []
    for f in score_files:
        try:
            tmp = pd.read_csv(f)
            tmp['split'] = get_split_from_filename(f)
            if 'x' in tmp.columns and 'prompt' not in tmp.columns:
                tmp.rename(columns={'x': 'prompt'}, inplace=True)
            df_score_list.append(tmp)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_score_list:
        print("No score files found.")
        return pd.DataFrame()

    df_score = pd.concat(df_score_list, ignore_index=True)

    # Rename score columns
    label_mapping = {
        'score_LABEL_0': 'score_extraversion',
        'score_LABEL_1': 'score_neuroticism',
        'score_LABEL_2': 'score_agreeableness',
        'score_LABEL_3': 'score_conscientiousness',
        'score_LABEL_4': 'score_openness'
    }
    df_score.rename(columns=label_mapping, inplace=True)

    # Merge
    merge_keys = ['trait', 'prompt', 'alpha_total', 'split']
    score_cols = [c for c in df_score.columns if c.startswith('score_')]
    df_score_subset = df_score[merge_keys + score_cols].drop_duplicates(subset=merge_keys)
    
    # Check if necessary columns exist in df_dist
    required_metrics = ['normalized_distance', 'perplexity', 'semantic_similarity']
    existing_metrics = [m for m in required_metrics if m in df_dist.columns]
    
    if not existing_metrics:
        print(f"Warning: None of the expected metrics {required_metrics} found in text metrics file.")
    
    df_merged = pd.merge(df_dist, df_score_subset, on=merge_keys, how='inner')
    
    return df_merged, existing_metrics

def calculate_slopes(df, metric_cols):
    """
    trait と split ごとに、指定されたmetricの傾きを計算する
    """
    groups = df.groupby(['trait', 'split'])
    results = []

    print(f"Calculating slopes for {len(groups)} groups...")
    
    for (trait, split), group in groups:
        # ターゲット性格スコア列の特定
        target_score_col = None
        for col in df.columns:
            if col.startswith('score_') and trait.lower() in col.lower():
                target_score_col = col
                break
        
        if target_score_col is None:
            continue

        lr = LinearRegression()

        # Score Sensitivity (alpha vs score)
        X = group[['alpha_total']].values.reshape(-1, 1)
        y_score = group[target_score_col].values
        lr.fit(X, y_score)
        score_slope = lr.coef_[0]

        row = {
            'trait': trait,
            'split': split,
            'score_sensitivity': score_slope,
            'target_col': target_score_col
        }

        # Calculate slopes for each metric
        for metric in metric_cols:
            if metric not in group.columns or group[metric].isnull().all():
                row[f'{metric}_slope'] = np.nan
                continue
            
            # Use absolute alpha for distance-like metrics if appropriate?
            # normalized_distance usually increases with |alpha|.
            # perplexity usually increases with |alpha|.
            # semantic_similarity usually decreases with |alpha|.
            # fitting against |alpha| assumes symmetry.
            
            X_abs = group[['alpha_total']].abs().values.reshape(-1, 1)
            y_metric = group[metric].fillna(0).values # Handle NaNs temporarily
            
            lr.fit(X_abs, y_metric)
            slope = lr.coef_[0]
            row[f'{metric}_slope'] = slope
            
            # Alias for backward compatibility if needed
            if metric == 'normalized_distance':
                row['text_change_slope'] = slope

        results.append(row)

    return pd.DataFrame(results)

def plot_radar(slope_df, output_dir, title_prefix="", metric_col='text_change_slope', metric_name='Text Change'):
    """splitごとにレーダーチャートを描画"""
    if slope_df.empty or metric_col not in slope_df.columns:
        return

    splits = slope_df['split'].unique()
    
    for split in splits:
        split_data = slope_df[slope_df['split'] == split].copy()
        
        # Big 5 Order
        order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        order = [t for t in order if t in split_data['trait'].unique()]
        
        if not order:
            continue
            
        split_data = split_data.set_index('trait').reindex(order).reset_index()

        N = len(order)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], order, color='grey', size=12)
        
        # 1. Score Sensitivity
        values1 = [abs(v) if pd.notnull(v) else 0 for v in split_data['score_sensitivity'].tolist()]
        values1 += values1[:1]
        ax.plot(angles, values1, linewidth=2, linestyle='solid', label='Score Sensitivity')
        ax.fill(angles, values1, alpha=0.1)

        # 2. Metric Slope
        values2 = [abs(v) if pd.notnull(v) else 0 for v in split_data[metric_col].tolist()]
        values2 += values2[:1]
        
        # 類似度(Similarity)の場合、値が減少するのが普通なので、傾きは負になることが多い。
        # 変化の大きさを見たいので絶対値をとるか、そのままか。
        # ここでは変化の大きさ(Magnitude)として絶対値をプロットする方針にします。
        
        ax.plot(angles, values2, linewidth=2, linestyle='dashed', label=f'{metric_name} Slope')
        ax.fill(angles, values2, alpha=0.1)

        safe_metric_name = metric_name.replace(" ", "_").lower()
        plt.title(f"{title_prefix} ({split})\nSensitivity vs {metric_name}", size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        out_path = os.path.join(output_dir, f"{title_prefix}_{split}_radar_{safe_metric_name}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved radar chart for {split} ({metric_name}) to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_glob", type=str, required=True, help="Glob pattern for text metrics csv")
    parser.add_argument("--score_glob", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="analysis_plots")
    parser.add_argument("--tag", type=str, default="Experiment")
    
    # Backward compatibility args (if needed, but script 99 is updated)
    parser.add_argument("--dist_glob", type=str, help="Deprecated: use --metrics_glob")

    args = parser.parse_args()

    metrics_glob = args.metrics_glob if args.metrics_glob else args.dist_glob
    
    df, metrics_found = load_data(metrics_glob, args.score_glob)
    if df.empty:
        print("No data loaded.")
        return

    slope_df = calculate_slopes(df, metrics_found)
    
    os.makedirs(args.out_dir, exist_ok=True)
    slope_df.to_csv(os.path.join(args.out_dir, f"{args.tag}_text_sensitivities.csv"), index=False)

    # Plot Radar for each found metric
    # Map metrics to display names
    metric_names = {
        'normalized_distance': 'Edit Distance',
        'perplexity': 'Perplexity',
        'semantic_similarity': 'Similarity'
    }

    for metric in metrics_found:
        if f'{metric}_slope' in slope_df.columns:
            disp_name = metric_names.get(metric, metric)
            plot_radar(slope_df, args.out_dir, title_prefix=args.tag, 
                      metric_col=f'{metric}_slope', metric_name=disp_name)

if __name__ == "__main__":
    main()