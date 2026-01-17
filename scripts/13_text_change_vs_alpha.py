import json
import argparse
import pandas as pd
import Levenshtein
import os
import sys

def load_data(file_path):
    """
    JSONまたはJSONL形式のファイルを読み込み、DataFrameとして返します。
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        sys.exit(1)

    data = []
    try:
        # JSONL (1行1JSON) の場合
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass # 空行や不正な行はスキップ
        
        # もしJSONLとして読み込めなかった（リストが空）場合、通常のJSONリストとして試行
        if not data:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    return pd.DataFrame(data)

def calculate_metrics(df, baseline_alpha):
    """
    プロンプト(x)と特性(trait)ごとにグループ化し、
    基準となるalpha値の生成文に対する編集距離を計算します。
    """
    results = []

    # プロンプトと特性でグループ化（同じ条件でのalpha違いを比較するため）
    # x: プロンプト, trait: 性格特性
    grouped = df.groupby(['trait', 'x'])

    print(f"Processing {len(grouped)} groups (trait + prompt combinations)...")

    for (trait, prompt), group in grouped:
        # 基準となる行を取得 (例: alpha_total == 0)
        baseline_row = group[group['alpha_total'] == baseline_alpha]

        if baseline_row.empty:
            # 指定されたalphaがこのグループに存在しない場合はスキップ、または最も近い値を基準にするなどの処理
            # ここではスキップし、警告を出します
            # print(f"Warning: Baseline alpha {baseline_alpha} not found for trait '{trait}' and prompt '{prompt[:20]}...'. Skipping.")
            continue
        
        # 基準テキスト
        base_text = baseline_row.iloc[0]['y']
        
        for _, row in group.iterrows():
            target_text = row['y']
            current_alpha = row['alpha_total']
            
            # レーベンシュタイン距離 (挿入・削除・置換の最小回数)
            dist = Levenshtein.distance(base_text, target_text)
            
            # 正規化レーベンシュタイン距離 (0.0〜1.0)
            # 0: 完全一致, 1: 全く異なる
            # 計算式: dist / max(len(base), len(target))
            max_len = max(len(base_text), len(target_text))
            norm_dist = dist / max_len if max_len > 0 else 0.0
            
            # 類似度 (1 - 正規化距離) 
            similarity = 1.0 - norm_dist

            # 文字数差
            len_diff = len(target_text) - len(base_text)

            results.append({
                "trait": trait,
                "prompt": prompt,
                "alpha_total": current_alpha,
                "levenshtein_distance": dist,
                "normalized_distance": norm_dist,
                "similarity_ratio": similarity,
                "length_diff": len_diff,
                "base_text_len": len(base_text),
                "target_text_len": len(target_text),
                "generated_text": target_text # 確認用（必要なければ削除可）
            })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Calculate Levenshtein distance for LLM generated texts across different steering alphas.")
    
    # 引数設定
    parser.add_argument("input_file", type=str, help="Path to the input JSON/JSONL file containing generation results.")
    parser.add_argument("--output", "-o", type=str, default="13_distance_results.csv", help="Path to save the output CSV file.")
    parser.add_argument("--baseline", "-b", type=float, default=0.0, help="The alpha value to use as the baseline for comparison (default: 0.0).")
    
    args = parser.parse_args()

    # データ読み込み
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    # 必要なカラムの存在確認
    required_cols = ['trait', 'x', 'y', 'alpha_total']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input data must contain columns: {required_cols}")
        sys.exit(1)

    # 計算実行
    print(f"Calculating distances (Baseline Alpha = {args.baseline})...")
    result_df = calculate_metrics(df, args.baseline)

    if result_df.empty:
        print("No comparisons could be made. Check if the baseline alpha exists in your data.")
        sys.exit(1)

    # 結果保存
    print(f"Saving results to {args.output}...")
    result_df.sort_values(by=['trait', 'prompt', 'alpha_total'], inplace=True)
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig') # Excel互換のためutf-8-sig推奨
    print("Done.")

if __name__ == "__main__":
    main()