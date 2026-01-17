#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 00_prepare_vectors.py (All-Layers Edition)
#
# 目的:
# 1. exp.yaml からモデルを1つロードする。
# 2. ★ 効率化: モデルのフォワードパスを1回実行するだけで、
#    「全レイヤー」の隠れ状態を取得し、レイヤーごとに v_diff を計算する。
# 3. 全レイヤーの v_diff を正規化し、(L, ax) のペアで .npz (axes_bank) に保存する。

import json, yaml, numpy as np, random
import os, shutil, time
from pathlib import Path
from typing import List, Dict
import torch
from huggingface_hub import login

# live_axes_and_hook.py から必要なヘルパーをインポート
from live_axes_and_hook import (
    load_model_and_tokenizer, get_layer_stack,
    _infer_main_device, _is_bnb_quantized,
    _ensure_pad_token, _ensure_dialog_tokens,
    _mean_pool_last_token,  # ★ build_axes_for_model の代わりにこれを使う
    AXES as AXES_CANON
)
# (datasets が必要)
try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP  = ROOT / "exp"
CFG  = EXP / "configs" / "big5_vectors.yaml"

# .hf_token 読み込み (02_compare... と同じロジック)
try:
    token_path = ROOT / ".hf_token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        login(token=token)
        print("[login] Logged in to HuggingFace Hub (token found)")
    else:
        print("[login] .hf_token not found, proceeding without login.")
except Exception as e:
    print(f"[login] Failed to login to HuggingFace Hub: {e}")

# Big5Chat 側の列名 → ライブラリ側の正規ラベル名への対応表
AX_MAP = {"O":"openness","C":"conscientiousness","E":"extraversion","A":"agreeableness","N":"neuroticism"}
AXES = ["O","C","E","A","N"]  # Big5Chat の列名

def extract_big5_texts_from_hf(per_axis=1000):
    """
    HuggingFace の Big5Chat から各軸の上位/下位サンプルを抽出して pos/neg を返す。
    (この関数は 02_compare... と同じ)
    """
    if not _HAS_DATASETS:
        raise RuntimeError("datasets が無いため Big5Chat を読み込めません。pip install datasets してください。")

    ds_all = load_dataset("wenkai-li/big5_chat")
    ds = ds_all[next(iter(ds_all.keys()))] if isinstance(ds_all, dict) else ds_all

    POS = {ax: [] for ax in AXES_CANON}
    NEG = {ax: [] for ax in AXES_CANON}

    for ex in ds:
        tr = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr not in AXES_CANON or lv not in {"high","low"}:
            continue
        to = (ex.get("train_output") or "").strip()
        if not to: continue
        text = f"<asst> {to}"
        (POS if lv == "high" else NEG)[tr].append(text)

    for ax in AXES_CANON:
        random.shuffle(POS[ax]); random.shuffle(NEG[ax])
        if per_axis > 0:
            POS[ax] = POS[ax][:per_axis]
            NEG[ax] = NEG[ax][:per_axis]
        print(f"[big5chat] {ax}: +{len(POS[ax])} / -{len(NEG[ax])} (dialog)")
    return POS, NEG


def main():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    
    model_name = cfg["model_name"]
    print(f"--- Preparing ALL-LAYER vectors for model: {model_name} ---")

    # --- 1. テキストデータをロード ---
    PER_AXIS= int(cfg.get("per_axis", 1000))
    POS_TEXTS, NEG_TEXTS = extract_big5_texts_from_hf(PER_AXIS)
    
    # (ローカルファイルフォールバックは、簡潔のため省略。必要なら 02 から移植)
    
    # --- 2. モデルをロードし、全レイヤー情報を取得 ---
    model, tok = load_model_and_tokenizer(
        model_name, quant=cfg.get("quant","auto"), device_map="auto"
    )
    
    _stack, N_layers, kind = get_layer_stack(model)
    # ★ 全レイヤー (0=emb, 1..N=layers) を対象にする
    layer_indices = list(range(N_layers)) 
    print(f"Targeting ALL {len(layer_indices)} hidden states (L=0 to L={N_layers}) (Kind={kind})")
    
    H_dim = model.config.hidden_size
    print(f"Model Hidden Dimension (H): {H_dim}")
    
    device = _infer_main_device(model)
    if _is_bnb_quantized(model):
        model.eval()
    else:
        model.to(device).eval()
        
    tok = _ensure_pad_token(tok, model)
    pooling = os.getenv("POOLING", cfg.get("pooling", "asst"))
    print(f"[INFO] Using pooling mode = {pooling}")
    if pooling == "asst":
        tok = _ensure_dialog_tokens(tok, model)
        
    batch_size = int(cfg.get("batch_size", 8))
    max_length = int(cfg.get("max_length", 160))

    # --- 3. 全レイヤー一括計算 (02_compare... と同じロジック) ---
    @torch.no_grad()
    def calculate_vectors_for_axis(
        texts: List[str]
    ) -> Dict[int, torch.Tensor]:
        """
        指定テキストリストに対し、全レイヤーの「平均」ベクトル {L: (H,)} を返す。
        (注: この時点では正規化しない)
        """
        layer_vecs_sum = {L: torch.zeros(H_dim, device=device, dtype=torch.float32) for L in layer_indices}
        total_samples = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tok_d = tok(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length
            ).to(device)
            
            out = model(**tok_d, output_hidden_states=True, use_cache=False)
            attn = tok_d["attention_mask"]
            
            for L, hs in enumerate(out.hidden_states): # L = 0, 1, ..., N
                if L not in layer_indices: continue
                v_batch = _mean_pool_last_token(hs, attn, mode=pooling) 
                layer_vecs_sum[L] += v_batch.to(torch.float32) * len(batch)
            total_samples += len(batch)
            
        if total_samples == 0:
            return {L: torch.zeros(H_dim, dtype=torch.float32) for L in layer_indices}
            
        final_vecs = {L: (v_sum / total_samples) for L, v_sum in layer_vecs_sum.items()}
        return final_vecs
    # --- インナー関数定義終わり ---
    
    print("Calculating vectors for ALL layers/axes (this may take a while)...")
    axes_by_layer = {} # 最終的な保存コンテナ
    
    for ax in AXES_CANON:
        print(f"  Axis: {ax} (pos)...")
        v_pos_all_layers = calculate_vectors_for_axis(POS_TEXTS[ax])
        
        print(f"  Axis: {ax} (neg)...")
        v_neg_all_layers = calculate_vectors_for_axis(NEG_TEXTS[ax])
        
        # 3. 差分を計算し保存
        for L in layer_indices:
            # L=0 (埋め込み層) もベクトルとして保存する
            v_pos = v_pos_all_layers[L]
            v_neg = v_neg_all_layers[L]
            
            v_diff = (v_pos - v_neg) # まだ (H,) tensor
            
            # (L, ax) のキーで、numpy 配列として保存
            axes_by_layer[(L, ax)] = v_diff.cpu().to(torch.float32).numpy()
    
    del model, tok
    torch.cuda.empty_cache()
    
    # --- 4. .npz ファイル (axes_bank) に保存 ---
    bank_path_str = os.getenv("AX_BANK")
    bank_path = Path(bank_path_str)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    
    # np.savez は (key=str, value=np.ndarray) の辞書を要求する
    npz_dict = {
        f"{L}|{ax}": vec 
        for (L, ax), vec in axes_by_layer.items()
    }
    
    np.savez_compressed(bank_path, **npz_dict) # 圧縮保存
    print(f"\n[Done] Saved ALL-LAYER axes bank to: {bank_path}")
    print(f"  Total vectors saved: {len(npz_dict)}")

if __name__ == "__main__":
    main()