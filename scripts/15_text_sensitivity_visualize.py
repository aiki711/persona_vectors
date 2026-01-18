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

def load_data(edit_dist_pattern, score_pattern):
    """CSVファイルを読み込み、split情報を付与して結合します"""
    
    # 編集距離データの読み込み
    dist_files = glob.glob(edit_dist_pattern)
    df_dist_list = []
    for f in dist_files:
        tmp = pd.read_csv(f)
        tmp['split'] = get_split_from_filename(f) # split情報を付与
        df_dist_list.append(tmp)
    
    if not df_dist_list:
        print("No edit distance files found.")
        return pd.DataFrame()
    
    df_dist = pd.concat(df_dist_list, ignore_index=True)

    # 性格スコアデータの読み込み
    score_files = glob.glob(score_pattern)
    df_score_list = []
    for f in score_files:
        tmp = pd.read_csv(f)
        tmp['split'] = get_split_from_filename(f) # split情報を付与
        if 'x' in tmp.columns and 'prompt' not in tmp.columns:
            tmp.rename(columns={'x': 'prompt'}, inplace=True)
        df_score_list.append(tmp)

    if not df_score_list:
        print("No score files found.")
        return pd.DataFrame()

    df_score = pd.concat(df_score_list, ignore_index=True)

    # Rename score columns based on Minej/bert-base-personality mapping
    # 0: extraversion, 1: neuroticism, 2: agreeableness, 3: conscientiousness, 4: openness
    label_mapping = {
        'score_LABEL_0': 'score_extraversion',
        'score_LABEL_1': 'score_neuroticism',
        'score_LABEL_2': 'score_agreeableness',
        'score_LABEL_3': 'score_conscientiousness',
        'score_LABEL_4': 'score_openness'
    }
    df_score.rename(columns=label_mapping, inplace=True)

    # 結合 (trait, prompt, alpha_total, split がキー)
    merge_keys = ['trait', 'prompt', 'alpha_total', 'split']
    
    # スコア列の抽出
    score_cols = [c for c in df_score.columns if c.startswith('score_')]
    df_score_subset = df_score[merge_keys + score_cols].drop_duplicates(subset=merge_keys)
    
    # マージ
    df_merged = pd.merge(df_dist, df_score_subset, on=merge_keys, how='inner')
    
    return df_merged

def calculate_slopes(df):
    """
    trait と split ごとに傾きを計算する
    """
    # trait と split の組み合わせでグループ化
    groups = df.groupby(['trait', 'split'])
    results = []

    print(f"Calculating slopes for {len(groups)} groups...")
    
    for (trait, split), group in groups:
        # ターゲットカラムの特定
        target_score_col = None
        for col in df.columns:
            if col.startswith('score_') and trait.lower() in col.lower():
                target_score_col = col
                break
        
        if target_score_col is None:
            continue

        lr = LinearRegression()

        # 1. Text Change Slope (|alpha| vs dist)
        X_abs = group[['alpha_total']].abs().values.reshape(-1, 1)
        y_dist = group['normalized_distance'].values
        lr.fit(X_abs, y_dist)
        dist_slope = lr.coef_[0]

        # 2. Score Sensitivity (alpha vs score)
        X = group[['alpha_total']].values.reshape(-1, 1)
        y_score = group[target_score_col].values
        lr.fit(X, y_score)
        score_slope = lr.coef_[0]

        results.append({
            'trait': trait,
            'split': split,  # split情報を保持
            'text_change_slope': dist_slope,
            'score_sensitivity': score_slope,
            'target_col': target_score_col
        })

    return pd.DataFrame(results)

def plot_radar(slope_df, output_dir, title_prefix=""):
    """splitごとにレーダーチャートを描画"""
    if slope_df.empty:
        return

    splits = slope_df['split'].unique()
    
    for split in splits:
        split_data = slope_df[slope_df['split'] == split].copy()
        
        # Big 5順序固定
        order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        order = [t for t in order if t in split_data['trait'].unique()]
        
        split_data = split_data.set_index('trait').reindex(order).reset_index()

        N = len(order)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], order, color='grey', size=12)
        
        # Score Sensitivity
        values1 = [abs(v) for v in split_data['score_sensitivity'].tolist()]
        values1 += values1[:1]
        ax.plot(angles, values1, linewidth=2, linestyle='solid', label='Score Sensitivity')
        ax.fill(angles, values1, alpha=0.1)

        # Text Change Slope
        values2 = split_data['text_change_slope'].tolist()
        values2 += values2[:1]
        ax.plot(angles, values2, linewidth=2, linestyle='dashed', label='Text Change Slope')
        ax.fill(angles, values2, alpha=0.1)

        plt.title(f"{title_prefix} ({split}) Sensitivity", size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # ファイル名にsplitを含める
        out_path = os.path.join(output_dir, f"{title_prefix}_{split}_radar_sensitivity.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved radar chart for {split} to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_glob", type=str, required=True)
    parser.add_argument("--score_glob", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="analysis_plots")
    parser.add_argument("--tag", type=str, default="Experiment")

    args = parser.parse_args()

    df = load_data(args.dist_glob, args.score_glob)
    if df.empty:
        print("No data loaded.")
        return

    slope_df = calculate_slopes(df)
    
    os.makedirs(args.out_dir, exist_ok=True)
    slope_df.to_csv(os.path.join(args.out_dir, f"{args.tag}_text_sensitivities.csv"), index=False)

    plot_radar(slope_df, args.out_dir, title_prefix=args.tag)

if __name__ == "__main__":
    main()