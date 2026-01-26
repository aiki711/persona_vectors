import json
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import sys
import os

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
                        pass
        
        # もしJSONLとして読み込めなかった（リストが空）場合、通常のJSONリストとして試行
        if not data:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    return pd.DataFrame(data)

def get_personality_scores(texts, model_name, batch_size=16):
    """
    Hugging Faceのモデルを使用してテキストの性格スコアを算出します。
    """
    
    # デバイス設定 (GPUがあれば使用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルとトークナイザーの読み込み
    print(f"Loading model: {model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # ラベル情報の取得 (モデルが持つラベル名を取得: Agreeableness, Opennessなど)
    id2label = model.config.id2label
    if not id2label:
        # ラベル情報がない場合のフォールバック（モデルによりますが、一般的なBig5の順序を仮定）
        print("Warning: id2label not found in config. Using indices as columns.")
        id2label = {i: f"dim_{i}" for i in range(model.config.num_labels)}
    
    print(f"Model labels: {list(id2label.values())}")

    all_scores = []
    
    # バッチ処理
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print("Calculating scores...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, unit="batch", disable=True):
            batch_texts = texts[i : i + batch_size]
            
            # トークナイズ
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            # 推論
            outputs = model(**inputs)
            
            # ロジットを取得し、正規化（ここではSigmoidまたはSoftmaxを適用して0-1の確率/スコアにする）
            # 性格特性は独立したラベルであることが多いため、Sigmoidを使用する場合が多いですが、
            # モデルの学習方法によります。ここでは汎用的にロジットをそのまま出力するか、確率にするか。
            # 直感的な理解のために softmax/sigmoid を適用します。
            # Minej/bert-base-personality はマルチラベル回帰的な挙動をするため、ここではそのままロジットをとるか
            # 比較のためスコア化します。
            
            scores = outputs.logits.cpu()
            
            # 辞書形式に変換してリストに追加
            for score_tensor in scores:
                # tensorをnumpy/listに変換
                row_scores = score_tensor.tolist()
                score_dict = {id2label[idx]: val for idx, val in enumerate(row_scores)}
                all_scores.append(score_dict)

    return pd.DataFrame(all_scores)

def main():
    parser = argparse.ArgumentParser(description="Calculate Big Five personality scores using a BERT model.")
    
    parser.add_argument("input_file", type=str, help="Path to the input JSON/JSONL file containing generated texts.")
    parser.add_argument("--output", "-o", type=str, default="14_personality_scores.csv", help="Path to save the output CSV file.")
    parser.add_argument("--model", "-m", type=str, default="Minej/bert-base-personality", help="Hugging Face model name to use.")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size for inference (adjust based on VRAM).")
    
    args = parser.parse_args()

    # データ読み込み
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    if 'y' not in df.columns:
        print("Error: Input data must contain a 'y' column (generated text).")
        sys.exit(1)

    # テキストリストの作成（空文字などは置換）
    texts = df['y'].fillna("").astype(str).tolist()

    # スコア計算
    scores_df = get_personality_scores(texts, args.model, args.batch_size)

    # 元のデータと結合
    print("Merging results...")
    # カラム名が被らないようにprefixをつける
    scores_df = scores_df.add_prefix("score_")
    result_df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

    # 結果保存
    print(f"Saving results to {args.output}...")
    
    # 分析しやすいように、trait, alpha, y, そしてスコア列を先頭に持ってくる並べ替え（任意）
    cols = list(result_df.columns)
    priority_cols = ['trait', 'alpha_total', 'x', 'y'] + [c for c in cols if c.startswith('score_')]
    other_cols = [c for c in cols if c not in priority_cols]
    result_df = result_df[priority_cols + other_cols]

    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print("Done.")

if __name__ == "__main__":
    main()