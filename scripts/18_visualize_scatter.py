import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import json
from tqdm import tqdm

def get_color(alpha):
    if alpha > 0:
        return 'red'
    elif alpha < 0:
        return 'blue'
    else:
        return 'gray'

def plot_scatter(x, y, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(6, 5))
    
    # 色のリスト作成
    colors = [get_color(a) for a in x]
    
    # 散布図
    plt.scatter(x, y, c=colors, alpha=0.6, edgecolors='k', linewidth=0.3)
    
    # グリッドと基準線
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()

def process_external_scores(root_dir, suffix):
    """
    *_personality_scores.csv を探して散布図を描く (Robust版)
    """
    pattern = os.path.join(root_dir, f"*/asst_pairwise_results{suffix}/*_personality_scores.csv")
    files = glob.glob(pattern)
    
    print(f"[External] Found {len(files)} score CSVs.")
    
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # --- カラム名の揺らぎ吸収 ---
            # Alphaカラムの特定
            alpha_col = None
            if 'alpha' in df.columns:
                alpha_col = 'alpha'
            elif 'alpha_total' in df.columns:
                alpha_col = 'alpha_total'
            
            # Label Mapping for Score columns
            # 0: extraversion, 1: neuroticism, 2: agreeableness, 3: conscientiousness, 4: openness
            label_mapping = {
                'score_LABEL_0': 'score_extraversion',
                'score_LABEL_1': 'score_neuroticism',
                'score_LABEL_2': 'score_agreeableness',
                'score_LABEL_3': 'score_conscientiousness',
                'score_LABEL_4': 'score_openness'
            }
            df.rename(columns=label_mapping, inplace=True)

            # Scoreカラムの特定
            # "score" 単独カラムがある場合と、上記のように score_xxx に分かれている場合がある
            score_cols_found = [c for c in df.columns if c.startswith('score_')]
            
            # チェック
            if alpha_col is None or not score_cols_found:
                # "score" カラムも "score_xxx" カラムもない場合
                if 'score' not in df.columns and 'probability' not in df.columns:
                    print(f"[Skip] {os.path.basename(f)}: Missing columns. Found: {list(df.columns)}")
                    continue
            
            # 単独scoreカラムのフォールバック
            if not score_cols_found:
                if 'score' in df.columns:
                    score_cols_found = ['score']
                elif 'probability' in df.columns:
                    # probabilityをscoreとして扱う
                    df['score'] = df['probability']
                    score_cols_found = ['score']

            # --- メタデータ抽出 ---
            basename = os.path.basename(f)
            dirname = os.path.dirname(f)
            
            # model_tag と split の推定
            # ディレクトリ構成: exp/{model_tag}/asst_pairwise_results{suffix}/...
            model_dir = os.path.dirname(os.path.dirname(f))
            model_tag = os.path.basename(model_dir)
            
            # split はファイル名から推定
            if "base" in basename:
                split = "base"
            elif "instruct" in basename:
                split = "instruct"
            else:
                split = "unknown"

            # Plot出力先
            plot_dir = os.path.join(dirname, "plots", "external")
            os.makedirs(plot_dir, exist_ok=True)
            
            # --- Plotting ---
            # traitカラムがある場合 (まだ分割されていないデータなど)
            if 'trait' in df.columns:
                traits = df['trait'].unique()
                for t in traits:
                    subset = df[df['trait'] == t]
                    if subset.empty: continue
                    
                    # 対応するスコアカラムを探す
                    # trait名がカラム名に含まれているかチェック (case-insensitive)
                    target_col = None
                    for c in score_cols_found:
                        if t.lower() in c.lower():
                            target_col = c
                            break
                    
                    # 見つからなければ 'score' を使う (単一カラムの場合)
                    if target_col is None and 'score' in df.columns:
                        target_col = 'score'
                        
                    if target_col is None:
                        # traitに対応するスコアがない場合はスキップ
                        continue

                    out_name = f"{model_tag}_{split}_{t}_scatter_external.png"
                    plot_scatter(
                        subset[alpha_col], subset[target_col], 
                        f"External Score: {t}\n({model_tag} / {split})",
                        "Alpha", f"BERT Score ({target_col})",
                        os.path.join(plot_dir, out_name)
                    )
            else:
                # traitカラムがない場合、すべての score_xxx カラムについてプロット
                for col in score_cols_found:
                    trait_name = col.replace("score_", "")
                    out_name = f"{model_tag}_{split}_{trait_name}_scatter_external.png"
                    plot_scatter(
                        df[alpha_col], df[col], 
                        f"External Score: {trait_name}\n({model_tag} / {split})",
                        "Alpha", f"BERT Score ({trait_name})",
                        os.path.join(plot_dir, out_name)
                    )

        except Exception as e:
            print(f"[Error] Processing {f}: {e}")

def process_internal_sensitivity(root_dir, suffix):
    """
    *_with_rms.jsonl を探して散布図を描く
    """
    pattern = os.path.join(root_dir, f"*/asst_pairwise_results{suffix}/*_with_rms.jsonl")
    files = glob.glob(pattern)
    
    print(f"[Internal] Found {len(files)} JSONL files.")
    
    for f in tqdm(files, desc="Processing Internal Metrics"):
        try:
            data = []
            with open(f, 'r') as fin:
                for line in fin:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    if 'alpha_total' in rec and 's_avg' in rec:
                        data.append(rec)
            
            if not data:
                continue
                
            df = pd.DataFrame(data)
            
            # メタデータ抽出
            dirname = os.path.dirname(f)
            basename = os.path.basename(f)
            
            model_dir = os.path.dirname(os.path.dirname(f))
            model_tag = os.path.basename(model_dir)
            
            # split & trait extraction
            parts = basename.replace("_with_rms.jsonl", "").split('_')
            
            if "base" in parts:
                idx = parts.index("base")
                split = "base"
                trait = "_".join(parts[idx+1:])
            elif "instruct" in parts:
                idx = parts.index("instruct")
                split = "instruct"
                trait = "_".join(parts[idx+1:])
            else:
                split = "unknown"
                trait = "unknown"
            
            plot_dir = os.path.join(dirname, "plots", "internal")
            os.makedirs(plot_dir, exist_ok=True)
            
            out_name = f"{model_tag}_{split}_{trait}_scatter_internal.png"
            
            plot_scatter(
                df['alpha_total'], df['s_avg'],
                f"Internal Sensitivity: {trait}\n({model_tag} / {split})",
                "Alpha", "Cosine Similarity (s_avg)",
                os.path.join(plot_dir, out_name)
            )

        except Exception as e:
            print(f"Error processing {f}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="exp", help="Root directory")
    parser.add_argument("--suffix", default="_L10-30", help="Suffix of results dir (e.g. _L10-30 or empty)")
    args = parser.parse_args()
    
    print(f"Generating scatter plots for results in '{args.suffix}' folders...")
    
    process_external_scores(args.root_dir, args.suffix)
    process_internal_sensitivity(args.root_dir, args.suffix)
    
    print("Done.")

if __name__ == "__main__":
    main()