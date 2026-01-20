import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re
from scipy.stats import linregress
from math import pi
import seaborn as sns

# --- 設定: モデルごとの最大Alpha値 ---
# 17_cross_model_comparison.py と整合性を保持
ALPHA_MAX_MAP = {
    "mistral_7b": 5.5,
    "llama3_8b": 20,
    "olmo3_7b": 50,
    "qwen25_7b": 160,
    "gemma2_9b": 500,
    "falcon3_7b": 500
}

# --- 設定: 正しいラベルマッピング (ENACO順) ---
# モデルの出力IDと性格特性の対応
LABEL_MAP = {
    'score_LABEL_0': 'score_extraversion',
    'score_LABEL_1': 'score_neuroticism',
    'score_LABEL_2': 'score_agreeableness',
    'score_LABEL_3': 'score_conscientiousness',
    'score_LABEL_4': 'score_openness'
}

# 表示順序（レーダーチャートの時計回り）
# 03_slopes_visualize.py, 17_cross_model_comparison.py と整合
TRAIT_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

def get_alpha_max(model_name):
    """
    モデル名からalpha_maxを推定
    完全一致 -> 前方一致 -> デフォルト(1.0) の順で検索
    """
    if model_name in ALPHA_MAX_MAP:
        return ALPHA_MAX_MAP[model_name]
    
    # 部分一致検索 (e.g. mistral_7b_v0.1 -> mistral_7b)
    for key, val in ALPHA_MAX_MAP.items():
        if key in model_name:
            return val
            
    print(f"Warning: Alpha Max not found for {model_name}, using default 1.0")
    return 1.0

def calculate_external_impact(csv_path, alpha_max):
    """外部スコアCSVからImpact（傾き * alpha_max）を計算"""
    try:
        df = pd.read_csv(csv_path)
        
        # ラベルのリネーム
        df.rename(columns=LABEL_MAP, inplace=True)
        
        # Alphaカラムの特定
        alpha_col = 'alpha' if 'alpha' in df.columns else 'alpha_total'
        if alpha_col not in df.columns:
            return {}

        impacts = {}
        for trait in TRAIT_ORDER:
            # その特性への介入データのみを抽出
            # (traitカラムがない場合は全データを使うが、通常はあるはず)
            if 'trait' in df.columns:
                subset = df[df['trait'] == trait]
            else:
                subset = df
            
            if subset.empty:
                continue

            # 対象のスコアカラム（例: score_openness）
            target_col = f"score_{trait}"
            if target_col not in subset.columns:
                continue
            
            # 線形回帰で傾きを算出
            # NAを除去
            subset = subset.dropna(subset=[alpha_col, target_col])
            if len(subset) < 2:
                continue

            slope, intercept, r_value, p_value, std_err = linregress(subset[alpha_col], subset[target_col])
            
            # Impact = |Slope * Alpha_Max|
            # 絶対値をとることで「変化の大きさ」に統一
            impacts[trait] = abs(slope * alpha_max)
            
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return {}
    
    return impacts

def load_internal_slopes(slopes_path):
    """
    内部感度(Slopes)ファイルを読み込み、trait/splitごとに平均化して返す。
    Internal Impact計算用。
    """
    if not os.path.exists(slopes_path):
        print(f"Warning: Internal slopes file not found: {slopes_path}")
        return None

    try:
        df = pd.read_csv(slopes_path)
        # 必要なカラムがあるか確認
        if 'slope_delta_score_vs_alpha' not in df.columns:
             print(f"Warning: 'slope_delta_score_vs_alpha' column not found in {slopes_path}")
             return None
        
        # model列をsplitとして扱う (base/instruct)
        if 'model' in df.columns:
            df['split'] = df['model'].apply(lambda x: 'base' if 'base' in str(x).lower() else ('instruct' if 'instruct' in str(x).lower() else str(x)))
        else:
            print(f"Warning: 'model' column not found in {slopes_path}")
            return None

        # trait, split ごとに層平均をとる
        df['trait'] = df['trait'].str.lower()
        
        internal_df = df.groupby(['trait', 'split'])['slope_delta_score_vs_alpha'].mean().reset_index()
        internal_df.rename(columns={'slope_delta_score_vs_alpha': 'internal_slope'}, inplace=True)
        return internal_df

    except Exception as e:
        print(f"Error loading internal slopes from {slopes_path}: {e}")
        return None

def plot_radar_comparison(external_impacts_base, external_impacts_instruct, internal_df, family_name, save_path):
    """
    Base vs Instruct の比較レーダーチャートを描画 (2段組み: External / Internal)
    """
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 16), subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(hspace=0.4)
    
    splits = ['base', 'instruct']
    colors = {'base': '#1f77b4', 'instruct': '#ff7f0e'}  # Blue, Orange
    
    angles = [n / float(len(TRAIT_ORDER)) * 2 * pi for n in range(len(TRAIT_ORDER))]
    angles += [angles[0]]  # 閉じるために先頭を追加
    
    # --- Plot 1: External Score Impact ---
    ax1 = axes[0]
    
    # データを整える
    ext_data = {'base': [], 'instruct': []}
    for trait in TRAIT_ORDER:
        ext_data['base'].append(external_impacts_base.get(trait, 0))
        ext_data['instruct'].append(external_impacts_instruct.get(trait, 0))
    
    # max scale calc
    all_ext_vals = ext_data['base'] + ext_data['instruct']
    max_ext = max(all_ext_vals) if all_ext_vals else 1.0
    max_ext = max_ext * 1.1 if max_ext > 0 else 1.0

    # Label setup
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([t.capitalize() for t in TRAIT_ORDER], fontsize=12)
    
    for split in splits:
        values = ext_data[split]
        values += [values[0]] # close polygon
        ax1.plot(angles, values, linewidth=2, linestyle='solid', label=f"{family_name} {split}", color=colors[split])
        ax1.fill(angles, values, color=colors[split], alpha=0.1)
    
    ax1.set_ylim(0, max_ext)
    ax1.set_title(f"External Score Impact", fontsize=16, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # --- Plot 2: Internal Impact ---
    ax2 = axes[1]
    
    if internal_df is not None and not internal_df.empty:
        # Internal DF contains 'internal_impact' calculated in main
        # Prepare data
        int_data = {'base': [], 'instruct': []}
        
        for split in splits:
            subset = internal_df[internal_df['split'] == split]
            vals = []
            for trait in TRAIT_ORDER:
                # 該当trait行を取得
                row = subset[subset['trait'] == trait]
                if not row.empty:
                    vals.append(row['internal_impact'].iloc[0])
                else:
                    vals.append(0)
            int_data[split] = vals
            
        all_int_vals = int_data['base'] + int_data['instruct']
        max_int = max(all_int_vals) if all_int_vals else 1.0
        max_int = max_int * 1.1 if max_int > 0 else 1.0

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([t.capitalize() for t in TRAIT_ORDER], fontsize=12)

        for split in splits:
            values = int_data[split]
            values += [values[0]]
            ax2.plot(angles, values, linewidth=2, linestyle='solid', label=f"{family_name} {split}", color=colors[split])
            ax2.fill(angles, values, color=colors[split], alpha=0.1)
            
        ax2.set_ylim(0, max_int)
        ax2.set_title(f"Internal Impact", fontsize=16, pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    else:
        ax2.text(0.5, 0.5, "Internal Impact Data Not Found/Available", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title("Internal Impact (Data Missing)", fontsize=16)

    # 全体保存
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Alignment Impact (Base vs Instruct)")
    parser.add_argument("--root_dir", default="exp", help="Root directory of experiments")
    parser.add_argument("--suffix", default="", help="Result folder suffix (e.g. _L10-30)")
    args = parser.parse_args()

    # 1. 全てのスコアCSVを探索
    pattern = os.path.join(args.root_dir, f"*/asst_pairwise_results{args.suffix}/*_personality_scores.csv")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} score files matching pattern: {pattern}")
    
    # 2. モデルファミリーごとにグループ化
    # ファイル名から判定する: {model}_{split}_personality_scores.csv
    groups = {}
    
    for f in files:
        filename = os.path.basename(f)
        
        # Split (Base/Instruct) の判定
        if "_instruct_" in filename.lower() or "_chat_" in filename.lower() or "instruct" in filename.lower():
            split = "instruct"
        elif "_base_" in filename.lower() or "base" in filename.lower():
            split = "base"
        else:
            print(f"Skipping file (unknown split): {filename}")
            continue
            
        # Family Name の抽出
        # 一般的な命名規則: "mistral_7b_base_..." -> "mistral_7b"
        # "_base" や "_instruct" より前の部分をモデル名とする
        match = re.search(r'(.+?)(_|-)(base|instruct|chat)', filename, re.IGNORECASE)
        if match:
            family_name = match.group(1)
        else:
            # フォールバック: ディレクトリ名などから推定
            print(f"Warning: Could not extract family name from {filename}, trying directory.")
            family_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        
        # グループに追加
        if family_name not in groups:
            groups[family_name] = {"base": None, "instruct": None}
            
        groups[family_name][split] = f

    # 3. ペアが見つかったものについて処理
    # 3. ペアが見つかったものについて処理
    for family, pairs in groups.items():
        if pairs["base"] and pairs["instruct"]:
            print(f"Processing comparison for: {family}...")
            
            # Alpha Maxの取得
            alpha_max = get_alpha_max(family)
            print(f"  Alpha Max for {family}: {alpha_max}")
            
            # Baseデータの計算 (External)
            ext_impacts_base = calculate_external_impact(pairs["base"], alpha_max)
            # Instructデータの計算 (External)
            ext_impacts_instruct = calculate_external_impact(pairs["instruct"], alpha_max)
            
            if not ext_impacts_base and not ext_impacts_instruct:
                print(f"Skipping {family}: Could not calculate external impacts.")
                continue

            # Internal Impactの計算
            # base_fileパスからSlopesパスを推定
            base_file = pairs["base"]
            # base_file is inside asst_pairwise_results usually, e.g. exp/model/asst_pairwise_results/file.csv
            results_dir = os.path.dirname(base_file) 
            slopes_file = os.path.join(results_dir, "slopes", f"slopes_{family}_asst_pairwise.csv")
            
            # gemma2_9bなどの例外パス対応
            if not os.path.exists(slopes_file):
                alt_slopes_file = os.path.join(results_dir, f"slopes_{family}_asst_pairwise.csv")
                if os.path.exists(alt_slopes_file):
                    slopes_file = alt_slopes_file
                else:
                    # さらに汎用的な名前
                    alt_slopes_file_2 = os.path.join(results_dir, "slopes", "slopes_asst_pairwise.csv")
                    if os.path.exists(alt_slopes_file_2):
                        slopes_file = alt_slopes_file_2
            
            internal_df = load_internal_slopes(slopes_file)
            
            if internal_df is not None:
                # Impact = |Slope| * AlphaMax
                internal_df['internal_impact'] = internal_df['internal_slope'].abs() * alpha_max
            else:
                print(f"  Internal slopes not found or failed to load for {family}")

            # プロット出力
            out_dir = os.path.dirname(pairs["instruct"]) # Instruct側のフォルダに保存
            out_path = os.path.join(out_dir, "plots", f"{family}_alignment_comparison_radar.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            plot_radar_comparison(
                ext_impacts_base, 
                ext_impacts_instruct,
                internal_df,
                family, 
                out_path
            )
            
        else:
            # デバッグ情報
            status = []
            if pairs["base"]: status.append("Base found")
            else: status.append("Base MISSING")
            
            if pairs["instruct"]: status.append("Instruct found")
            else: status.append("Instruct MISSING")
            
            if pairs["base"] or pairs["instruct"]: # 片方でもある場合のみ表示
                print(f"Skipping {family}: Pair incomplete ({', '.join(status)})")

if __name__ == "__main__":
    main()