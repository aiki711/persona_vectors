import pandas as pd
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# 各モデルの実験で使用した最大Alpha値（Impact計算用）
# シェルスクリプトの設定値に基づきます
ALPHA_MAX_MAP = {
    "mistral_7b": 5.5,
    "llama3_8b": 20,
    "olmo3_7b": 50,
    "qwen25_7b": 160,
    "gemma2_9b": 500,
    "falcon3_7b": 500
}

def load_all_metrics(root_dir, glob_pattern):
    """
    指定されたパターンに一致する全モデルのCSVを読み込み、結合する。
    """
    search_path = os.path.join(root_dir, glob_pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"No files found for pattern: {search_path}")
        return pd.DataFrame()
    
    df_list = []
    for f in files:
        try:
            # パスからモデル名を抽出
            parts = os.path.normpath(f).split(os.sep)
            if 'exp' in parts:
                idx = parts.index('exp')
                tag = parts[idx+1]
            else:
                tag = "unknown"
            
            df = pd.read_csv(f)
            df['model_tag'] = tag
            
            # splitカラム名のゆらぎ吸収 (split_x -> split)
            if 'split' not in df.columns and 'split_x' in df.columns:
                df.rename(columns={'split_x': 'split'}, inplace=True)
            
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def calculate_impact(df):
    """
    Slope（傾き）に MaxAlpha を掛けて、Total Impact（最大変化量）に変換する。
    これにより、Alphaスケールの異なるモデル間でも公平に比較できる。
    """
    def get_max_alpha(row):
        return ALPHA_MAX_MAP.get(row['model_tag'], 1.0) # 未知のモデルは1.0倍

    # 各行に対してMaxAlphaを取得
    max_alphas = df.apply(get_max_alpha, axis=1)

    # Impact = Slope * MaxAlpha (絶対値で計算)
    df['internal_impact'] = df['internal_sensitivity'].abs() * max_alphas
    df['score_impact'] = df['score_sensitivity'].abs() * max_alphas
    df['text_change_impact'] = df['text_change_slope'].abs() * max_alphas
    
    return df

def plot_heatmap(df, value_col, title, output_path, cmap="Viridis"):
    """
    ヒートマップ描画 (Impactベース)
    """
    splits = df['split'].unique()
    
    for split in splits:
        subset = df[df['split'] == split]
        if subset.empty:
            continue
            
        # ピボットテーブル (行: モデル, 列: 特性)
        pivot = subset.pivot_table(index='model_tag', columns='trait', values=value_col, aggfunc='mean')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, linewidths=.5)
        plt.title(f"{title} ({split})\n(Normalized by Alpha Range)")
        plt.tight_layout()
        
        save_file = output_path.replace(".png", f"_{split}.png")
        plt.savefig(save_file)
        plt.close()
        print(f"Saved heatmap to {save_file}")

def plot_efficiency_bar(df, output_path):
    """
    効率性（Score Slope / Text Slope）の比較
    ※ 効率は比率なので、Alphaスケールはキャンセルされるため、元のSlopeを使って計算して良い。
    """
    # 0除算回避
    df['efficiency'] = df['score_sensitivity'].abs() / (df['text_change_slope'].abs() + 1e-9)
    
    splits = df['split'].unique()
    for split in splits:
        subset = df[df['split'] == split]
        if subset.empty:
            continue
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=subset, x='trait', y='efficiency', hue='model_tag')
        
        plt.title(f"Steering Efficiency (Score / TextChange) - {split}")
        plt.ylabel("Efficiency (Score gain per unit of text disruption)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_file = output_path.replace(".png", f"_{split}.png")
        plt.savefig(save_file)
        plt.close()
        print(f"Saved efficiency chart to {save_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="exp", help="Root directory of experiments")
    parser.add_argument("--pattern", default="*/asst_pairwise_results/plots/*_combined_metrics.csv", help="Glob pattern")
    parser.add_argument("--out_dir", default="exp/_all/comparison_plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading all model metrics...")
    df = load_all_metrics(args.root_dir, args.pattern)
    
    if df.empty:
        print("No data found.")
        return

    # スケール補正（Impact計算）
    df = calculate_impact(df)

    # 保存
    df.to_csv(os.path.join(args.out_dir, "all_models_metrics_impact.csv"), index=False)
    print(f"Saved processed data to {os.path.join(args.out_dir, 'all_models_metrics_impact.csv')}")

    # 1. Internal Impact (Mechanism)
    plot_heatmap(df, 'internal_impact', 
                 "Internal Impact (Total Vector Change)", 
                 os.path.join(args.out_dir, "heatmap_internal_impact.png"), cmap="Blues")

    # 2. Score Impact (Effect)
    plot_heatmap(df, 'score_impact', 
                 "External Score Impact (Total Score Shift)", 
                 os.path.join(args.out_dir, "heatmap_score_impact.png"), cmap="Greens")

    # 3. Text Change Impact (Side-effect)
    plot_heatmap(df, 'text_change_impact', 
                 "Text Change Impact (Total Edit Distance)", 
                 os.path.join(args.out_dir, "heatmap_text_change_impact.png"), cmap="Reds")

    # 4. Efficiency (Cost Performance) - これは補正なしの比率でOK
    plot_efficiency_bar(df, os.path.join(args.out_dir, "bar_efficiency.png"))

if __name__ == "__main__":
    main()