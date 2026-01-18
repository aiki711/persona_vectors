import pandas as pd
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# 各モデルの実験で使用した最大Alpha値（Impact計算用）
ALPHA_MAX_MAP = {
    "mistral_7b": 5.5,
    "llama3_8b": 20,
    "olmo3_7b": 50,
    "qwen25_7b": 160,
    "gemma2_9b": 500,
    "falcon3_7b": 500
}

# 03_slopes_visualize.py と同じ特性順序
TRAIT_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

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
            
            # splitカラム名のゆらぎ吸収
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
    """
    def get_max_alpha(row):
        return ALPHA_MAX_MAP.get(row['model_tag'], 1.0)

    max_alphas = df.apply(get_max_alpha, axis=1)

    df['internal_impact'] = df['internal_sensitivity'].abs() * max_alphas
    df['score_impact'] = df['score_sensitivity'].abs() * max_alphas
    df['text_change_impact'] = df['text_change_slope'].abs() * max_alphas
    
    return df

def plot_heatmap(df, value_col, title, output_path, cmap="Viridis"):
    """ヒートマップ描画"""
    splits = df['split'].unique()
    
    for split in splits:
        subset = df[df['split'] == split]
        if subset.empty:
            continue
            
        pivot = subset.pivot_table(index='model_tag', columns='trait', values=value_col, aggfunc='mean')
        # 特性順序を並べ替え
        existing_traits = [t for t in TRAIT_ORDER if t in pivot.columns]
        pivot = pivot[existing_traits]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, linewidths=.5)
        plt.title(f"{title} ({split})\n(Normalized by Alpha Range)")
        plt.tight_layout()
        
        save_file = output_path.replace(".png", f"_{split}.png")
        plt.savefig(save_file)
        plt.close()
        print(f"Saved heatmap to {save_file}")

def plot_efficiency_bar(df, output_path):
    """効率性（Score / TextChange）の棒グラフ"""
    # 0除算回避
    df['efficiency'] = df['score_sensitivity'].abs() / (df['text_change_slope'].abs() + 1e-9)
    
    splits = df['split'].unique()
    for split in splits:
        subset = df[df['split'] == split].copy()
        if subset.empty:
            continue
        
        plt.figure(figsize=(12, 6))
        # 特性順序を固定するためにカテゴリ型に変換してもよいが、ここではTRAIT_ORDER順にソート
        subset['trait'] = pd.Categorical(subset['trait'], categories=TRAIT_ORDER, ordered=True)
        subset = subset.sort_values('trait')

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

def plot_radar_comparison(df, value_col, title, output_path):
    """
    モデル間比較用のレーダーチャートを描画
    splitごとに作成し、全モデルを重ねて表示する
    """
    splits = df['split'].unique()
    
    # カラーパレットの準備（モデル用）
    models = df['model_tag'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color_map = dict(zip(models, colors))

    for split in splits:
        subset = df[df['split'] == split]
        if subset.empty:
            continue

        # 軸の設定（特性）
        categories = [t for t in TRAIT_ORDER if t in subset['trait'].unique()]
        N = len(categories)
        if N < 3:
            print(f"Not enough traits to plot radar for {split} (needs >= 3)")
            continue

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1] # 閉じるため

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], categories, color='black', size=12)
        
        # 軸のメモリ設定（データの最大値に合わせて調整したい場合は自動でもよい）
        # ここでは自動スケールに任せつつ、グリッドを見やすくする
        ax.yaxis.grid(True, linestyle='--')

        # モデルごとにプロット
        for model in models:
            model_data = subset[subset['model_tag'] == model]
            if model_data.empty:
                continue
            
            # 特性順序に合わせて値を抽出
            # データがない特性は0埋めするかスキップするかだが、ここでは再インデックスして0埋め
            pivot = model_data.set_index('trait')
            values = []
            for t in categories:
                if t in pivot.index:
                    val = pivot.loc[t, value_col]
                    # ピボットの結果がSeriesになる場合とスカラーになる場合をケア（念のため平均）
                    if isinstance(val, pd.Series):
                        val = val.mean()
                    values.append(abs(val)) # Impactは絶対値で比較
                else:
                    values.append(0.0)
            
            values += values[:1] # 閉じる
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=model_color_map[model])
            ax.fill(angles, values, alpha=0.05, color=model_color_map[model])

        plt.title(f"{title} ({split})", size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        save_file = output_path.replace(".png", f"_{split}.png")
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
        print(f"Saved radar chart to {save_file}")

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

    # --- 1. ヒートマップ (Heatmap) ---
    plot_heatmap(df, 'internal_impact', 
                 "Internal Impact (Total Vector Change)", 
                 os.path.join(args.out_dir, "heatmap_internal_impact.png"), cmap="Blues")

    plot_heatmap(df, 'score_impact', 
                 "External Score Impact (Total Score Shift)", 
                 os.path.join(args.out_dir, "heatmap_score_impact.png"), cmap="Greens")

    plot_heatmap(df, 'text_change_impact', 
                 "Text Change Impact (Total Edit Distance)", 
                 os.path.join(args.out_dir, "heatmap_text_change_impact.png"), cmap="Reds")

    # --- 2. 棒グラフ (Efficiency Bar) ---
    plot_efficiency_bar(df, os.path.join(args.out_dir, "bar_efficiency.png"))

    # --- 3. レーダーチャート (Radar Chart) ---
    # Internal Impact
    plot_radar_comparison(df, 'internal_impact',
                          "Internal Impact Comparison",
                          os.path.join(args.out_dir, "radar_internal_impact.png"))
    
    # Score Impact
    plot_radar_comparison(df, 'score_impact',
                          "External Score Impact Comparison",
                          os.path.join(args.out_dir, "radar_score_impact.png"))

    # Text Change Impact
    plot_radar_comparison(df, 'text_change_impact',
                          "Text Change Impact Comparison",
                          os.path.join(args.out_dir, "radar_text_change_impact.png"))

if __name__ == "__main__":
    main()