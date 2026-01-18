import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from math import pi

def max_scale(series):
    """
    最大値を1.0とするスケーリング (x / max)
    差が小さいデータでも0に潰れず、比率として可視化できる
    """
    # 絶対値をとる（向きは無視して「大きさ」を見るため）
    s_abs = series.abs()
    max_val = s_abs.max()
    if max_val == 0:
        return s_abs
    return s_abs / max_val

def load_and_process_internal_slopes(file_path):
    df = pd.read_csv(file_path)
    
    # model列の値(base/instructなど)をsplit列として正規化
    if 'model' in df.columns:
        df['split'] = df['model'].apply(lambda x: 'base' if 'base' in str(x).lower() else ('instruct' if 'instruct' in str(x).lower() else str(x)))
    else:
        df['split'] = 'unknown'

    # traitとsplitごとに層平均をとる
    internal_df = df.groupby(['trait', 'split'])['slope_delta_score_vs_alpha'].mean().reset_index()
    internal_df.rename(columns={'slope_delta_score_vs_alpha': 'internal_sensitivity'}, inplace=True)
    return internal_df

def plot_combined_radar(merged_df, output_path, title="Sensitivity Comparison"):
    if merged_df.empty:
        return

    # Big 5の順序を固定（見やすくするため）
    order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    # データにあるものだけでフィルタ
    order = [t for t in order if t in merged_df['trait'].unique()]
    
    # 並べ替え
    merged_df = merged_df.set_index('trait').reindex(order).reset_index()

    categories = merged_df['trait'].tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='black', size=12)

    # 軸の範囲を0-1に固定
    ax.set_ylim(0, 1.0)
    # グリッド線を設定
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="grey", size=10)

    # --- データの準備 (Max Scaling) ---
    
    # 1. Internal Sensitivity (内部変化)
    val_internal = max_scale(merged_df['internal_sensitivity']).tolist()
    val_internal += val_internal[:1]
    
    # 2. Score Sensitivity (外部性格スコア変化)
    val_score = max_scale(merged_df['score_sensitivity']).tolist()
    val_score += val_score[:1]
    
    # 3. Text Change (文章破壊度)
    val_text = max_scale(merged_df['text_change_slope']).tolist()
    val_text += val_text[:1]

    # --- プロット ---
    # 内部感度 (青・実線)
    ax.plot(angles, val_internal, linewidth=2, linestyle='solid', color='blue', label='Internal Sensitivity')
    ax.fill(angles, val_internal, 'blue', alpha=0.1)

    # 外部スコア感度 (緑・実線)
    ax.plot(angles, val_score, linewidth=2, linestyle='solid', color='green', label='Ext. Score Sensitivity')
    ax.fill(angles, val_score, 'green', alpha=0.1)

    # テキスト変化/破壊度 (赤・点線)
    ax.plot(angles, val_text, linewidth=2, linestyle='dashed', color='red', label='Text Change Slope')
    # 点線は見にくい場合があるので、fillは薄く
    
    # 凡例とタイトル
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(title, size=15, y=1.1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved radar chart to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal_csv", required=True)
    parser.add_argument("--external_csv", required=True)
    parser.add_argument("--out_dir", default="analysis_plots")
    parser.add_argument("--tag", default="Comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    df_internal = load_and_process_internal_slopes(args.internal_csv)
    df_external = pd.read_csv(args.external_csv)

    # splitの共通部分を探す
    splits = set(df_internal['split'].unique()) & set(df_external['split'].unique())
    
    if not splits:
        print("Warning: No common splits found.")
        print(f"Internal: {df_internal['split'].unique()}")
        print(f"External: {df_external['split'].unique()}")
        return

    for split in splits:
        print(f"Processing split: {split}")
        sub_int = df_internal[df_internal['split'] == split]
        sub_ext = df_external[df_external['split'] == split]
        
        merged = pd.merge(sub_int, sub_ext, on='trait', how='inner')
        
        if not merged.empty:
            # プロット
            out_name = f"{args.tag}_{split}_combined_radar.png"
            plot_combined_radar(
                merged, 
                os.path.join(args.out_dir, out_name), 
                title=f"{args.tag} ({split}) Comparison"
            )
            # CSV保存
            merged.to_csv(os.path.join(args.out_dir, f"{args.tag}_{split}_combined_metrics.csv"), index=False)

if __name__ == "__main__":
    main()