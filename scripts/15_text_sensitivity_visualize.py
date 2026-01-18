import pandas as pd
import numpy as np
import argparse
import os
import glob
import re
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import pi

def load_data(edit_dist_pattern, score_pattern):
    """CSVファイルを読み込み、結合してDataFrameを返します"""
    
    # 編集距離データの読み込み
    dist_files = glob.glob(edit_dist_pattern)
    df_dist_list = []
    for f in dist_files:
        tmp = pd.read_csv(f)
        # ファイル名からメタデータを推測（必要であれば実装）
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
        if 'x' in tmp.columns and 'prompt' not in tmp.columns:
            print(f"Renaming 'x' to 'prompt' in score data from {f}")
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

    # 結合 (trait, prompt, alpha_total がキーになると仮定)
    # 重複カラムを避けるための処理
    # Safe merge keys:
    merge_keys = ['trait', 'prompt', 'alpha_total']
    
    print(f"Merging on {merge_keys}...")
    
    # Check if keys exist in both
    for key in merge_keys:
        if key not in df_score.columns:
             raise KeyError(f"Key '{key}' not in score data columns: {df_score.columns.tolist()}")
        if key not in df_dist.columns:
             raise KeyError(f"Key '{key}' not in dist data columns: {df_dist.columns.tolist()}")
    
    # df_scoreから必要なスコア列だけ抜き出す
    score_cols = [c for c in df_score.columns if c.startswith('score_')]
    df_score_subset = df_score[merge_keys + score_cols].drop_duplicates(subset=merge_keys)
    
    # マージ
    df_merged = pd.merge(df_dist, df_score_subset, on=merge_keys, how='inner')
    
    return df_merged

def calculate_slopes(df):
    """
    各特性ごとに、alphaに対する各指標の傾き（感度）を計算します。
    """
    traits = df['trait'].unique()
    results = []

    print("Calculating slopes...")
    
    for trait in traits:
        # その特性のデータのみ抽出
        trait_data = df[df['trait'] == trait]
        
        # ターゲットとなるスコアカラムを特定 (例: score_Agreeableness)
        # trait文字列とカラム名のマッピングが必要。ここでは簡易的に検索
        target_score_col = None
        for col in df.columns:
            if col.startswith('score_') and trait.lower() in col.lower():
                target_score_col = col
                break
        
        if target_score_col is None:
            # マッピングできない場合、Extraversionなどの代表的なものを探すかスキップ
            continue

        # 回帰モデル
        lr = LinearRegression()

        # 1. Text Change Sensitivity (編集距離の感度)
        # 編集距離はalphaの絶対値に対して増える傾向があるため、|alpha| で回帰
        X_abs = trait_data[['alpha_total']].abs().values.reshape(-1, 1)
        y_dist = trait_data['normalized_distance'].values
        lr.fit(X_abs, y_dist)
        dist_slope = lr.coef_[0]

        # 2. Steering Effectiveness (性格スコアの感度)
        # スコアはalphaに対して正負の相関があるはずなので、そのままalphaで回帰
        X = trait_data[['alpha_total']].values.reshape(-1, 1)
        y_score = trait_data[target_score_col].values
        lr.fit(X, y_score)
        score_slope = lr.coef_[0]

        # R2スコアなども本来は見るべきだが、ここでは傾きのみ取得

        results.append({
            'trait': trait,
            'text_change_slope': dist_slope, # 文章の変化しやすさ
            'score_sensitivity': score_slope, # 性格の変わりやすさ
            'target_col': target_score_col
        })

    return pd.DataFrame(results)

def plot_radar(slope_df, output_dir, title_prefix=""):
    """レーダーチャートを描画します"""
    if slope_df.empty:
        return

    # Big 5の順序を固定したい場合
    order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    # データにあるものだけでフィルタ
    order = [t for t in order if t in slope_df['trait'].unique()]
    
    if not order:
        order = slope_df['trait'].unique().tolist()

    # データを並べ替え
    slope_df = slope_df.set_index('trait').reindex(order).reset_index()

    # 項目数
    N = len(order)

    # 角度計算
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # 閉じるために最初に戻る

    # プロット準備
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 軸のラベル設定
    plt.xticks(angles[:-1], order, color='grey', size=12)
    
    # --- データ系列1: 性格スコア感度 (Steering Effectiveness) ---
    values1 = slope_df['score_sensitivity'].tolist()
    # 負の傾き（意図と逆）もあり得るので絶対値にするか、あるいはそのまま出すか。
    # 「変わりやすさ」を見るなら絶対値が分かりやすいですが、制御失敗を見たいなら生の値。
    # ここでは「感度」として絶対値でプロットし、負の場合は色を変えるなどの処理も考えられますが、シンプルに絶対値にします。
    values1 = [abs(v) for v in values1]
    values1 += values1[:1]
    
    ax.plot(angles, values1, linewidth=2, linestyle='solid', label='Score Sensitivity (Effectiveness)')
    ax.fill(angles, values1, alpha=0.1)

    # --- データ系列2: テキスト変化感度 (Text Change) ---
    # スケールが違う可能性が高いので、第2軸を使うか、正規化が必要。
    # ここでは単純に重ねてプロットし、スケールは自動調整に任せます。
    values2 = slope_df['text_change_slope'].tolist()
    values2 += values2[:1]
    
    ax.plot(angles, values2, linewidth=2, linestyle='dashed', label='Text Change Slope (Disruption)')
    ax.fill(angles, values2, alpha=0.1)

    # タイトルと凡例
    plt.title(f"{title_prefix} Text & Score Sensitivity by Trait", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{title_prefix}_radar_sensitivity.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved radar chart to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize text metrics sensitivity (slopes) as radar charts.")
    parser.add_argument("--dist_glob", type=str, required=True, help="Glob pattern for edit distance CSVs (e.g. 'exp/*/results/*_edit_distance.csv')")
    parser.add_argument("--score_glob", type=str, required=True, help="Glob pattern for personality score CSVs")
    parser.add_argument("--out_dir", type=str, default="analysis_plots", help="Directory to save plots")
    parser.add_argument("--tag", type=str, default="Experiment", help="Tag for the plot title")

    args = parser.parse_args()

    # データ読み込み
    print("Loading data...")
    df = load_data(args.dist_glob, args.score_glob)
    
    if df.empty:
        print("No data loaded. Check glob patterns.")
        return

    # 傾き計算
    slope_df = calculate_slopes(df)
    print("Slopes calculated:")
    print(slope_df)
    
    # CSVとしても保存
    os.makedirs(args.out_dir, exist_ok=True)
    slope_df.to_csv(os.path.join(args.out_dir, f"{args.tag}_text_sensitivities.csv"), index=False)

    # 可視化
    print("Plotting...")
    plot_radar(slope_df, args.out_dir, title_prefix=args.tag)

if __name__ == "__main__":
    main()