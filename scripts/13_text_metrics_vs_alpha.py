import json
import argparse
import pandas as pd
import Levenshtein
import os
import sys
import torch
import math
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util

def load_data(file_path):
    """
    JSONまたはJSONL形式のファイルを読み込み、DataFrameとして返します。
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        sys.exit(1)

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        if not data:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    return pd.DataFrame(data)

def load_models(device, mode="all"):
    """
    Perplexity用のGPT2と、類似度用のSentence-BERTを読み込みます。
    mode="distance_only" の場合はNoneを返します。
    """
    if mode == "distance_only":
        return None, None, None

    print("Loading evaluation models...")
    
    # Perplexity (GPT-2)
    # gpt2-large or gpt2-xl is better, but using gpt2 (small) for speed/memory as a default
    ppl_model_id = 'gpt2' 
    ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).to(device)
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)
    ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
    ppl_model.eval()

    # Similarity (Sentence-BERT)
    sim_model_id = 'all-MiniLM-L6-v2'
    sim_model = SentenceTransformer(sim_model_id, device=device)
    
    return ppl_model, ppl_tokenizer, sim_model

def calculate_perplexity_batch(model, tokenizer, texts, device, batch_size=32):
    """
    GPT-2を使用してPerplexityをバッチ計算します。
    """
    if model is None:
        return [float('nan')] * len(texts)

    ppls = []
    
    # リファレンス実装として、バッチ処理は少し複雑になるため（padding maskの考慮など）、
    # ここではシンプルに1つずつ計算するが、torch.no_grad()内で回すことで高速化を図る方針にします。
    
    model.eval()
    results = []
    for text in texts:
        if not text.strip():
            results.append(float('nan'))
            continue
            
        encodings = tokenizer(text, return_tensors='pt').to(device)
        input_ids = encodings.input_ids
        
        if input_ids.size(1) == 0:
            results.append(float('nan'))
            continue
            
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss)
            results.append(ppl.item())
            
    return results

def calculate_metrics_batch(df, baseline_alpha, device="cuda", mode="all"):
    """
    編集距離、Perplexity、類似度を一括計算します。
    mode="distance_only" の場合、Perplexityと類似度は計算しません。
    """
    
    # モデルロード
    ppl_model, ppl_tokenizer, sim_model = load_models(device, mode)
    
    results = []
    grouped = df.groupby(['trait', 'x'])
    print(f"Processing {len(grouped)} groups...")

    # 進捗表示
    for (trait, prompt), group in tqdm(grouped):
        baseline_row = group[group['alpha_total'] == baseline_alpha]
        if baseline_row.empty:
            continue
            
        base_text = baseline_row.iloc[0]['y']
        
        # Target texts
        target_texts = group['y'].tolist()
        alphas = group['alpha_total'].tolist()
        
        # 1. Edit Distance (CPU)
        distances = []
        norm_distances = []
        similarities_ratio = []
        lengths = []
        
        for t_text in target_texts:
            d = Levenshtein.distance(base_text, t_text)
            max_len = max(len(base_text), len(t_text))
            norm_d = d / max_len if max_len > 0 else 0.0
            
            distances.append(d)
            norm_distances.append(norm_d)
            similarities_ratio.append(1.0 - norm_d)
            lengths.append(len(t_text) - len(base_text))

        # 2. Semantic Similarity (GPU Batch)
        sem_sims = [float('nan')] * len(target_texts)
        if mode == "all" and sim_model is not None:
             # Base text embedding
            base_emb = sim_model.encode(base_text, convert_to_tensor=True, show_progress_bar=False)
            target_embs = sim_model.encode(target_texts, convert_to_tensor=True, show_progress_bar=False)
            sem_sims = util.cos_sim(base_emb, target_embs)[0].cpu().numpy().tolist()
        
        # 3. Perplexity (GPU Sequential/Small Batch)
        ppls = [float('nan')] * len(target_texts)
        if mode == "all" and ppl_model is not None:
             ppls = calculate_perplexity_batch(ppl_model, ppl_tokenizer, target_texts, device)
        
        # 結果格納
        for i in range(len(target_texts)):
            results.append({
                "trait": trait,
                "prompt": prompt,
                "alpha_total": alphas[i],
                "levenshtein_distance": distances[i],
                "normalized_distance": norm_distances[i],
                "similarity_ratio": similarities_ratio[i],
                "length_diff": lengths[i],
                "base_text_len": len(base_text),
                "target_text_len": len(target_texts[i]),
                "perplexity": ppls[i],
                "semantic_similarity": sem_sims[i]
            })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Calculate Text Metrics (Distance, PPL, Similarity).")
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output", "-o", type=str, default="13_text_metrics.csv")
    parser.add_argument("--baseline", "-b", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", type=str, choices=["all", "distance_only"], default="all", help="Calculation mode: 'all' (default) or 'distance_only'.")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    print(f"Calculating metrics on {args.device} in '{args.mode}' mode...")
    result_df = calculate_metrics_batch(df, args.baseline, args.device, args.mode)
    
    print(f"Saving results to {args.output}...")
    result_df.sort_values(by=['trait', 'prompt', 'alpha_total'], inplace=True)
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig') # with BOM
    print("Done.")

if __name__ == "__main__":
    main()
